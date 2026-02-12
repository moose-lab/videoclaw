"""Bilibili publisher (placeholder -- requires bilibili API credentials)."""

from __future__ import annotations

import logging

from videoclaw.publishers.base import PublishRequest, PublishResult, PublishStatus

logger = logging.getLogger(__name__)


class BilibiliPublisher:
    platform_name = "bilibili"

    def __init__(self, sessdata: str | None = None) -> None:
        self._sessdata = sessdata

    async def publish(self, request: PublishRequest) -> PublishResult:
        logger.warning("Bilibili publish not yet implemented, returning stub")
        return PublishResult(
            platform=self.platform_name,
            status=PublishStatus.FAILED,
            error="Bilibili publisher not yet implemented.",
        )

    async def get_status(self, platform_id: str) -> PublishResult:
        return PublishResult(
            platform=self.platform_name,
            status=PublishStatus.FAILED,
            error="Not implemented",
        )

    async def health_check(self) -> bool:
        return False
