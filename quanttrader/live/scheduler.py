"""
live/scheduler.py
──────────────────
APScheduler-based strategy scheduler for live trading.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz

logger = logging.getLogger(__name__)


class TradingScheduler:
    def __init__(self, timezone: str = "America/New_York"):
        self.tz = pytz.timezone(timezone)
        self.scheduler = BackgroundScheduler(timezone=self.tz)
        self._jobs: dict = {}

    def add_strategy_job(
        self,
        func: Callable,
        job_id: str,
        cron_expr: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Add a strategy execution job.

        cron_expr: e.g. "0 9 * * 1-5"  (9am Mon-Fri)
        interval_seconds: e.g. 3600 (every hour)
        """
        if cron_expr:
            trigger = CronTrigger.from_crontab(cron_expr, timezone=self.tz)
        elif interval_seconds:
            trigger = IntervalTrigger(seconds=interval_seconds)
        else:
            raise ValueError("Provide cron_expr or interval_seconds.")

        self.scheduler.add_job(func, trigger=trigger, id=job_id,
                               replace_existing=True, **kwargs)
        self._jobs[job_id] = func
        logger.info("Scheduled job '%s'", job_id)

    def start(self) -> None:
        self.scheduler.start()
        logger.info("Scheduler started (%d jobs)", len(self._jobs))

    def stop(self) -> None:
        self.scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")

    def pause_job(self, job_id: str) -> None:
        self.scheduler.pause_job(job_id)

    def resume_job(self, job_id: str) -> None:
        self.scheduler.resume_job(job_id)
