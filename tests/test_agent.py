import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main as agent_main
from app.models import SimplifiedPartMessage
from app.segmenter import (
    SentenceAccumulator,
    build_segments_for_part,
    split_utf8_by_bytes,
)


def test_utf8_segmentation_respects_1000_byte_limit():
    # Кириллица занимает по 2 байта на символ в UTF-8
    long_text = "Термодинамика изучает превращения энергии и тепловые процессы. " * 40
    part = SimplifiedPartMessage(
        sender="Иван",
        requestId="14:25:10",
        partNumber=1,
        simplifiedText=long_text,
        isFinal=False,
    )

    segments = build_segments_for_part(part, send_time="14:25:12.001", max_bytes=1000)
    assert len(segments) > 1

    for idx, seg in enumerate(segments, start=1):
        assert seg.segmentNumber == idx
        assert seg.totalSegments == len(segments)
        assert len(seg.payload.encode("utf-8")) <= 1000

    reassembled = "".join(seg.payload for seg in segments)
    decoded = json.loads(reassembled)
    assert decoded["simplifiedText"] == long_text
    assert decoded["sender"] == "Иван"


def test_split_utf8_does_not_corrupt_multibyte_chars():
    sample = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" * 5
    chunks = split_utf8_by_bytes(sample, max_bytes=11)
    assert all(len(c.encode("utf-8")) <= 11 for c in chunks)
    assert "".join(chunks) == sample


def test_sentence_accumulator_groups_sentences_and_handles_abbreviations():
    acc = SentenceAccumulator(sentences_per_part=2)
    tokens = [
        "При нагревании ",
        "частицы движутся быстрее, т.е. их энергия растёт. ",
        "Чем выше температура, тем сильнее движение! ",
        "Это правило работает для газов и жидкостей.",
    ]

    emitted = []
    for tok in tokens:
        emitted.extend(acc.feed(tok))
    emitted.extend(acc.flush())

    assert len(emitted) == 2
    assert "т.е." in emitted[0]
    assert emitted[0].endswith("движение!")
    assert emitted[1] == "Это правило работает для газов и жидкостей."


def test_transfer_validation_returns_400():
    with TestClient(agent_main.app) as client:
        resp = client.post(
            "/transfer",
            json={"sender": "   ", "sendTime": "14:25:10", "complexText": ""},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"] is True


def test_transfer_returns_500_when_ollama_unavailable(monkeypatch):
    monkeypatch.setattr(
        agent_main.ollama_client,
        "check_availability",
        AsyncMock(return_value=False),
    )
    with TestClient(agent_main.app) as client:
        resp = client.post(
            "/transfer",
            json={
                "sender": "Иван",
                "sendTime": "14:25:10",
                "complexText": "Энтропия замкнутой системы не убывает.",
            },
        )
        assert resp.status_code == 500
        assert resp.json()["error"] is True


def test_transfer_streams_parts_and_final_marker(monkeypatch):
    sent_segments = []

    async def fake_stream(_client, _text):
        yield "В замкнутой системе беспорядок растёт. "
        yield "Поэтому чай сам остывает в комнате. "
        yield "Обратный процесс сам по себе не идёт."

    async def fake_send_segment(_client, segment):
        sent_segments.append(segment)

    monkeypatch.setattr(
        agent_main.ollama_client,
        "check_availability",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(agent_main.ollama_client, "stream_simplification", fake_stream)
    monkeypatch.setattr(agent_main.transport_client, "send_segment", fake_send_segment)

    with TestClient(agent_main.app) as client:
        resp = client.post(
            "/transfer",
            json={
                "sender": "Иван",
                "sendTime": "14:25:10",
                "complexText": "Энтропия замкнутой системы не убывает.",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    # Проверяем, что части собрались корректно и последняя имеет isFinal=True
    assembled_parts = {}
    for seg in sent_segments:
        assembled_parts.setdefault(seg.sendTime, []).append(seg.payload)

    messages = [json.loads("".join(chunks)) for chunks in assembled_parts.values()]
    assert len(messages) == 3
    assert messages[0]["partNumber"] == 1
    assert messages[0]["isFinal"] is False
    assert messages[1]["partNumber"] == 2
    assert messages[1]["isFinal"] is False
    assert messages[2]["partNumber"] == 3
    assert messages[2]["simplifiedText"] == ""
    assert messages[2]["isFinal"] is True