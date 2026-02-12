"""YouTube publisher (placeholder -- requires google-api-python-client)."""

from __future__ import annotations

import logging

from videoclaw.publishers.base import PublishRequest, PublishResult, PublishStatus

logger = logging.getLogger(__name__)


class YouTubePublisher:
    platform_name = "youtube"

    def __init__(self, credentials_path: str | None = None) -> None:
        self._credentials_path = credentials_path

    async def publish(self, request: PublishRequest) -> PublishResult:
        # TODO: Implement with google-api-python-client
        logger.warning("YouTube publish not yet implemented, returning stub")
        return PublishResult(
            platform=self.platform_name,
            status=PublishStatus.FAILED,
            error="YouTube publisher not yet implemented. See docs/publisher-guide.md",
        )

    async def get_status(self, platform_id: str) -> PublishResult:
        return PublishResult(
            platform=self.platform_name,
            status=PublishStatus.FAILED,
            error="Not implemented",
        )

    async def health_check(self) -> bool:
        return False
