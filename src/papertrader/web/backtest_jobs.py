"""Runs backtests from the dashboard as background jobs.

A backtest against a large universe/date range can take minutes (network
fetch per symbol, then a day-by-day replay) -- far too long for a single
HTTP request. Instead the dashboard starts a job in a background thread and
polls a small in-memory status dict for progress, the same pattern as the
autonomous engine already runs in its own thread (see `serve` in cli.py).
This is deliberately process-local, single-user state: fine for a personal
dashboard, not meant to survive a restart or be shared across processes.
"""
from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..config import Config

_jobs: dict[str, "BacktestJob"] = {}
_lock = threading.Lock()


@dataclass
class BacktestJob:
    id: str
    start: str
    end: str
    universe_file: str
    status: str = "running"  # running | done | error
    stage: str = "starting"  # starting | fetch | simulate
    progress_current: int = 0
    progress_total: int = 0
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)


def start_backtest_job(cfg: Config, start: str, end: str, universe_file: str | None = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    bt_cfg = cfg
    if universe_file:
        # Don't mutate the live app's Config -- a backtest run from the
        # dashboard shouldn't affect what the running engine/other panels see.
        bt_cfg = Config(raw=copy.deepcopy(cfg.raw), path=cfg.path)
        bt_cfg.raw.setdefault("universe", {})["file"] = universe_file

    job = BacktestJob(id=job_id, start=start, end=end, universe_file=bt_cfg.universe_file)
    with _lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_run, args=(job_id, bt_cfg, start, end), name=f"backtest-{job_id}", daemon=True)
    thread.start()
    return job_id


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return asdict(job) if job is not None else None


def _run(job_id: str, cfg: Config, start: str, end: str) -> None:
    from ..backtest.engine import Backtester

    def on_progress(stage: str, current: int, total: int) -> None:
        with _lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.stage = stage
                job.progress_current = current
                job.progress_total = total

    try:
        bt = Backtester(cfg, start=start, end=end, on_progress=on_progress)
        with _lock:
            _jobs[job_id].progress_total = len(bt.universe)
        result = bt.run()
        with _lock:
            job = _jobs[job_id]
            job.status = "done"
            job.result = asdict(result)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the polling client, don't crash the thread silently
        with _lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.status = "error"
                job.error = str(exc)
