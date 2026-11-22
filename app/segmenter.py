import re
from datetime import datetime, timedelta
from typing import List

from app.models import SegmentRequest, SimplifiedPartMessage

# Сокращения, после которых точка не означает конец предложения
_ABBREVIATIONS = (
    "т.д",
    "т.п",
    "т.е",
    "т.к",
    "др",
    "пр",
    "см",
    "им",
    "г",
    "гг",
    "вв",
    "рис",
    "табл",
    "стр",
    "с",
    "напр",
    "и т.д",
    "и т.п",
)

# Граница предложения: знак конца предложения + пробельный символ
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+")


def _ends_with_abbreviation(fragment: str) -> bool:
    cleaned = fragment.rstrip(". ").lower()
    for abbr in _ABBREVIATIONS:
        if cleaned == abbr or cleaned.endswith(f" {abbr}") or cleaned.endswith(f"({abbr}"):
            return True
    # Одиночная буква перед точкой (например, инициалы «Н. И.»)
    if re.search(r"(?:^|\s)[а-яёa-z]$", cleaned):
        return True
    return False


class SentenceAccumulator:
    """Накапливает потоковые токены от LLM и выдаёт готовые части по N предложений."""

    def __init__(self, sentences_per_part: int = 2) -> None:
        if sentences_per_part < 1:
            raise ValueError("sentences_per_part должен быть >= 1")
        self.sentences_per_part = sentences_per_part
        self._buffer: str = ""
        self._ready_sentences: List[str] = []

    def feed(self, token: str) -> List[str]:
        """Добавляет очередной фрагмент текста и возвращает список готовых частей."""
        if not token:
            return []

        self._buffer += token
        self._extract_complete_sentences()

        parts: List[str] = []
        while len(self._ready_sentences) >= self.sentences_per_part:
            chunk = self._ready_sentences[: self.sentences_per_part]
            del self._ready_sentences[: self.sentences_per_part]
            parts.append(" ".join(chunk).strip())
        return parts

    def flush(self) -> List[str]:
        """Возвращает оставшийся текст после завершения потока генерации."""
        tail = self._buffer.strip()
        self._buffer = ""
        if tail:
            self._ready_sentences.append(tail)

        if not self._ready_sentences:
            return []

        remaining = " ".join(self._ready_sentences).strip()
        self._ready_sentences.clear()
        return [remaining] if remaining else []

    def _extract_complete_sentences(self) -> None:
        raw_parts = _SENTENCE_BOUNDARY_RE.split(self._buffer)
        if len(raw_parts) <= 1:
            return

        # Последний элемент может быть незавершённым предложением — оставляем в буфере
        pending = raw_parts[-1]
        completed = raw_parts[:-1]

        merged: List[str] = []
        carry = ""
        for item in completed:
            candidate = f"{carry} {item}".strip() if carry else item.strip()
            if not candidate:
                continue
            if _ends_with_abbreviation(candidate):
                carry = candidate
            else:
                merged.append(candidate)
                carry = ""

        if carry:
            pending = f"{carry} {pending}".strip() if pending else carry

        self._ready_sentences.extend(merged)
        self._buffer = pending


def split_utf8_by_bytes(text: str, max_bytes: int = 1000) -> List[str]:
    """
    Разбивает строку на сегменты так, чтобы каждый сегмент был корректной строкой UTF-8
    и его размер в байтах не превышал max_bytes.
    """
    if max_bytes < 4:
        raise ValueError("max_bytes должен быть не менее 4 байт для поддержки UTF-8")
    if not text:
        return [""]

    segments: List[str] = []
    current_chars: List[str] = []
    current_bytes = 0

    for char in text:
        char_bytes = len(char.encode("utf-8"))
        if current_bytes + char_bytes > max_bytes:
            segments.append("".join(current_chars))
            current_chars = [char]
            current_bytes = char_bytes
        else:
            current_chars.append(char)
            current_bytes += char_bytes

    if current_chars:
        segments.append("".join(current_chars))

    return segments


class UniqueSendTimeGenerator:
    """
    Генерирует уникальную метку времени sendTime для каждой отправляемой части,
    исключая коллизии при сборке сегментов на транспортном уровне.
    """

    def __init__(self) -> None:
        self._last_dt: datetime | None = None

    def next_time(self) -> str:
        now = datetime.now()
        now = now.replace(microsecond=(now.microsecond // 1000) * 1000)
        if self._last_dt is not None and now <= self._last_dt:
            now = self._last_dt + timedelta(milliseconds=1)
        self._last_dt = now
        return now.strftime("%H:%M:%S.%f")[:-3]


def build_segments_for_part(
    part: SimplifiedPartMessage,
    send_time: str,
    max_bytes: int = 1000,
) -> List[SegmentRequest]:
    """Сериализует часть ответа в компактный JSON и формирует список SegmentRequest."""
    serialized = part.model_dump_json()
    payload_chunks = split_utf8_by_bytes(serialized, max_bytes=max_bytes)
    total = len(payload_chunks)

    return [
        SegmentRequest(
            sendTime=send_time,
            totalSegments=total,
            segmentNumber=idx,
            payload=chunk,
        )
        for idx, chunk in enumerate(payload_chunks, start=1)
    ]