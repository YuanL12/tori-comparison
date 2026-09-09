#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

template <typename scalar_t>
__device__ bool point_in_triangle(
    const scalar_t* __restrict__ tri,
    scalar_t px,
    scalar_t py,
    scalar_t eps,
    scalar_t* l0,
    scalar_t* l1,
    scalar_t* l2) {
  const scalar_t ax = tri[0];
  const scalar_t ay = tri[1];
  const scalar_t bx = tri[2];
  const scalar_t by = tri[3];
  const scalar_t cx = tri[4];
  const scalar_t cy = tri[5];

  const scalar_t denom = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
  const scalar_t tiny = scalar_t(1e-30);
  if (denom > -tiny && denom < tiny) return false;

  *l0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / denom;
  *l1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / denom;
  *l2 = scalar_t(1) - *l0 - *l1;
  return *l0 >= -eps && *l1 >= -eps && *l2 >= -eps;
}

template <typename scalar_t>
__global__ void locate_uv_kernel(
    const scalar_t* __restrict__ points, // restricted so that it will not overlap with other variables
    const scalar_t* __restrict__ tri_uv,
    int64_t n_points,
    int64_t n_triangles,
    scalar_t eps,
    int64_t* __restrict__ out_face,
    scalar_t* __restrict__ out_bary) {
  
  // Each thread handles one point
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_points) return;

  const scalar_t px = points[2 * i + 0];
  const scalar_t py = points[2 * i + 1];
  out_face[i] = -1;
  out_bary[3 * i + 0] = scalar_t(0);
  out_bary[3 * i + 1] = scalar_t(0);
  out_bary[3 * i + 2] = scalar_t(0);

  for (int64_t f = 0; f < n_triangles; ++f) {
    scalar_t l0, l1, l2;
    if (point_in_triangle(tri_uv + 6 * f, px, py, eps, &l0, &l1, &l2)) {
      out_face[i] = f;
      out_bary[3 * i + 0] = l0;
      out_bary[3 * i + 1] = l1;
      out_bary[3 * i + 2] = l2;
      return;
    }
  }
}

template <typename scalar_t>
__global__ void locate_uv_grid_kernel(
    const scalar_t* __restrict__ points,
    const scalar_t* __restrict__ tri_uv,
    const int64_t* __restrict__ cell_starts,
    const int64_t* __restrict__ cell_tris,
    int64_t n_points,
    int64_t n_triangles,
    int64_t grid_res,
    scalar_t eps,
    int64_t* __restrict__ out_face,
    scalar_t* __restrict__ out_bary) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_points) return;

  const scalar_t px = points[2 * i + 0];
  const scalar_t py = points[2 * i + 1];
  out_face[i] = -1;
  out_bary[3 * i + 0] = scalar_t(0);
  out_bary[3 * i + 1] = scalar_t(0);
  out_bary[3 * i + 2] = scalar_t(0);

  int64_t gx = static_cast<int64_t>(px * static_cast<scalar_t>(grid_res));
  int64_t gy = static_cast<int64_t>(py * static_cast<scalar_t>(grid_res));
  if (gx < 0) gx = 0;
  if (gy < 0) gy = 0;
  if (gx >= grid_res) gx = grid_res - 1;
  if (gy >= grid_res) gy = grid_res - 1;

  const int64_t cell = gy * grid_res + gx;
  const int64_t start = cell_starts[cell];
  const int64_t end = cell_starts[cell + 1];
  for (int64_t k = start; k < end; ++k) {
    const int64_t f = cell_tris[k];
    scalar_t l0, l1, l2;
    if (point_in_triangle(tri_uv + 6 * f, px, py, eps, &l0, &l1, &l2)) {
      out_face[i] = f;
      out_bary[3 * i + 0] = l0;
      out_bary[3 * i + 1] = l1;
      out_bary[3 * i + 2] = l2;
      return;
    }
  }

  // ponytail: brute fallback keeps boundary/roundoff misses correct; remove after grid coverage is proven.
  for (int64_t f = 0; f < n_triangles; ++f) {
    scalar_t l0, l1, l2;
    if (point_in_triangle(tri_uv + 6 * f, px, py, eps, &l0, &l1, &l2)) {
      out_face[i] = f;
      out_bary[3 * i + 0] = l0;
      out_bary[3 * i + 1] = l1;
      out_bary[3 * i + 2] = l2;
      return;
    }
  }
}

std::vector<torch::Tensor> locate_uv(torch::Tensor points, torch::Tensor tri_uv, double eps) {
  CHECK_CUDA(points);
  CHECK_CUDA(tri_uv);
  TORCH_CHECK(points.scalar_type() == tri_uv.scalar_type(), "points and tri_uv must have the same dtype");
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 2, "points must have shape (N, 2)");
  TORCH_CHECK(tri_uv.dim() == 3 && tri_uv.size(1) == 3 && tri_uv.size(2) == 2,
              "tri_uv must have shape (F, 3, 2)");
  
  // make them compact row-major
  points = points.contiguous();
  tri_uv = tri_uv.contiguous();
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(tri_uv);

  auto out_face = torch::empty({points.size(0)}, points.options().dtype(torch::kInt64));
  auto out_bary = torch::empty({points.size(0), 3}, points.options());

  const int threads = 256;
  const int64_t n_points = points.size(0);
  const int blocks = static_cast<int>((n_points + threads - 1) / threads);
  if (n_points == 0) return {out_face, out_bary};

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "locate_uv", [&] {
    locate_uv_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<scalar_t>(),
        tri_uv.data_ptr<scalar_t>(),
        n_points,
        tri_uv.size(0),
        static_cast<scalar_t>(eps),
        out_face.data_ptr<int64_t>(),
        out_bary.data_ptr<scalar_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out_face, out_bary};
}

std::vector<torch::Tensor> locate_uv_grid(
    torch::Tensor points,
    torch::Tensor tri_uv,
    torch::Tensor cell_starts,
    torch::Tensor cell_tris,
    int64_t grid_res,
    double eps) {
  CHECK_CUDA(points);
  CHECK_CUDA(tri_uv);
  CHECK_CUDA(cell_starts);
  CHECK_CUDA(cell_tris);
  TORCH_CHECK(points.scalar_type() == tri_uv.scalar_type(), "points and tri_uv must have the same dtype");
  TORCH_CHECK(cell_starts.scalar_type() == torch::kInt64, "cell_starts must be int64");
  TORCH_CHECK(cell_tris.scalar_type() == torch::kInt64, "cell_tris must be int64");
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 2, "points must have shape (N, 2)");
  TORCH_CHECK(tri_uv.dim() == 3 && tri_uv.size(1) == 3 && tri_uv.size(2) == 2,
              "tri_uv must have shape (F, 3, 2)");
  TORCH_CHECK(grid_res > 0, "grid_res must be positive");
  TORCH_CHECK(cell_starts.dim() == 1 && cell_starts.size(0) == grid_res * grid_res + 1,
              "cell_starts must have shape (grid_res * grid_res + 1,)");
  TORCH_CHECK(cell_tris.dim() == 1, "cell_tris must be 1D");

  points = points.contiguous();
  tri_uv = tri_uv.contiguous();
  cell_starts = cell_starts.contiguous();
  cell_tris = cell_tris.contiguous();
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(tri_uv);
  CHECK_CONTIGUOUS(cell_starts);
  CHECK_CONTIGUOUS(cell_tris);

  auto out_face = torch::empty({points.size(0)}, points.options().dtype(torch::kInt64));
  auto out_bary = torch::empty({points.size(0), 3}, points.options());

  const int threads = 256;
  const int64_t n_points = points.size(0);
  const int blocks = static_cast<int>((n_points + threads - 1) / threads);
  if (n_points == 0) return {out_face, out_bary};

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "locate_uv_grid", [&] {
    locate_uv_grid_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<scalar_t>(),
        tri_uv.data_ptr<scalar_t>(),
        cell_starts.data_ptr<int64_t>(),
        cell_tris.data_ptr<int64_t>(),
        n_points,
        tri_uv.size(0),
        grid_res,
        static_cast<scalar_t>(eps),
        out_face.data_ptr<int64_t>(),
        out_bary.data_ptr<scalar_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out_face, out_bary};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("locate_uv", &locate_uv, "Brute-force CUDA UV point locator");
  m.def("locate_uv_grid", &locate_uv_grid, "Uniform-grid CUDA UV point locator");
}
