from __future__ import annotations

from typing import Any, Dict


class Logger:
    def start(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        tags: list[str],
        group: str | None,
        run_id: str | None = None,
        resume: str | None = None,
    ) -> None:
        raise NotImplementedError

    def log(self, data: Dict[str, float], *, step: int | None = None) -> None:
        raise NotImplementedError

    def update_config(self, updates: Dict[str, Any]) -> None:
        raise NotImplementedError

    def details_log(self, data: Dict[str, Any]) -> None:
        """Optional trajectory / artifact storage (used when saving UV paths)."""
        _ = data

    def finish(self) -> None:
        raise NotImplementedError


class NoOpLogger(Logger):
    def __init__(self) -> None:
        self._details: Dict[str, Any] = {}

    def start(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        tags: list[str],
        group: str | None,
        run_id: str | None = None,
        resume: str | None = None,
    ) -> None:
        _ = (project, name, config, tags, group, run_id, resume)

    def log(self, data: Dict[str, float], *, step: int | None = None) -> None:
        _ = data
        _ = step

    def update_config(self, updates: Dict[str, Any]) -> None:
        _ = updates

    def details_log(self, data: Dict[str, Any]) -> None:
        self._details.update(data)

    def finish(self) -> None:
        return None


class WandbLogger(Logger):
    def __init__(self) -> None:
        import wandb  # local import to avoid hard dependency when not used

        self._wandb = wandb
        self._started: bool = False
        self._details: Dict[str, Any] = {}

    def start(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        tags: list[str],
        group: str | None,
        run_id: str | None = None,
        resume: str | None = None,
    ) -> None:
        self._wandb.login()
        init_kwargs: Dict[str, Any] = {
            "project": project,
            "name": name,
            "config": config,
            "tags": tags,
            "group": group,
        }
        if run_id is not None:
            init_kwargs["id"] = run_id
        if resume is not None:
            init_kwargs["resume"] = resume
        self._wandb.init(**init_kwargs)
        self._started = True

    def log(self, data: Dict[str, float], *, step: int | None = None) -> None:
        if self._started:
            if step is None:
                self._wandb.log(data)
            else:
                self._wandb.log(data, step=int(step))

    def update_config(self, updates: Dict[str, Any]) -> None:
        if self._started and updates:
            self._wandb.config.update(updates, allow_val_change=True)

    def details_log(self, data: Dict[str, Any]) -> None:
        self._details.update(data)

    def finish(self) -> None:
        if self._started:
            self._wandb.finish()
            self._started = False
