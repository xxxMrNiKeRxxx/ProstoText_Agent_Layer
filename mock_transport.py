import json
from collections import defaultdict
from typing import Dict

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Заглушка транспортного уровня (для тестов агента)")
buffers: Dict[str, Dict[int, str]] = defaultdict(dict)


class IncomingSegment(BaseModel):
    sendTime: str
    totalSegments: int
    segmentNumber: int
    payload: str


@app.post("/simplify/transfer")
async def receive_segment(seg: IncomingSegment):
    byte_size = len(seg.payload.encode("utf-8"))
    print(
        f"[TRANSPORT STUB] Сегмент {seg.segmentNumber}/{seg.totalSegments} "
        f"| sendTime={seg.sendTime} | {byte_size} Б"
    )
    buffers[seg.sendTime][seg.segmentNumber] = seg.payload

    if len(buffers[seg.sendTime]) == seg.totalSegments:
        assembled = "".join(
            buffers[seg.sendTime][i] for i in range(1, seg.totalSegments + 1)
        )
        parsed = json.loads(assembled)
        print(f"[TRANSPORT STUB] Собрана часть #{parsed['partNumber']} (isFinal={parsed['isFinal']}): {parsed['simplifiedText']}")
        del buffers[seg.sendTime]

    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)