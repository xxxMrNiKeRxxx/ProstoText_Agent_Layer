import logging
from typing import Iterable

import httpx

from app.models import SegmentRequest

logger = logging.getLogger(__name__)


class TransportError(RuntimeError):
    """Ошибка доставки сегмента на транспортный уровень."""


class TransportClient:
    def __init__(self, transfer_url: str, timeout_sec: float = 15.0) -> None:
        self.transfer_url = transfer_url
        self.timeout_sec = timeout_sec

    async def send_segment(
        self,
        client: httpx.AsyncClient,
        segment: SegmentRequest,
    ) -> None:
        """Отправляет один сегмент на транспортный уровень (POST /simplify/transfer)."""
        try:
            response = await client.post(
                self.transfer_url,
                json=segment.model_dump(),
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            logger.info(
                "Отправлен сегмент %d/%d (sendTime=%s, bytes=%d)",
                segment.segmentNumber,
                segment.totalSegments,
                segment.sendTime,
                len(segment.payload.encode("utf-8")),
            )
        except httpx.HTTPError as exc:
            raise TransportError(
                f"Ошибка отправки сегмента {segment.segmentNumber}/{segment.totalSegments} "
                f"на {self.transfer_url}: {exc}"
            ) from exc

    async def send_segments(
        self,
        client: httpx.AsyncClient,
        segments: Iterable[SegmentRequest],
    ) -> None:
        """Последовательно отправляет все сегменты одной части ответа."""
        for segment in segments:
            await self.send_segment(client, segment)