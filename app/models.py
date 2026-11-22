from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import settings


class TransferRequest(BaseModel):
    sender: str = Field(..., min_length=1, description="Имя пользователя (отправителя запроса)")
    sendTime: str = Field(
        ...,
        min_length=1,
        description="Время отправки исходного запроса (используется как идентификатор запроса)",
    )
    complexText: str = Field(
        ...,
        min_length=1,
        description="Исходный сложный научный или технический текст для упрощения",
    )
    requestId: Optional[str] = Field(
        default=None,
        description="Опциональный идентификатор запроса (если не передан, используется sendTime)",
    )

    @field_validator("sender", "sendTime", "complexText")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Поле не должно быть пустым или состоять только из пробелов")
        return stripped

    @property
    def effective_request_id(self) -> str:
        if self.requestId and self.requestId.strip():
            return self.requestId.strip()
        return self.sendTime


class SimplifiedPartMessage(BaseModel):
    sender: str = Field(..., min_length=1)
    requestId: str = Field(..., min_length=1)
    partNumber: int = Field(..., ge=1)
    simplifiedText: str
    isFinal: bool


class SegmentRequest(BaseModel):
    sendTime: str = Field(..., min_length=1)
    totalSegments: int = Field(..., ge=1)
    segmentNumber: int = Field(..., ge=1)
    payload: str

    @field_validator("payload")
    @classmethod
    def validate_payload_byte_size(cls, value: str) -> str:
        byte_len = len(value.encode("utf-8"))
        if byte_len > settings.max_segment_bytes:
            raise ValueError(
                f"Размер payload ({byte_len} байт) превышает лимит {settings.max_segment_bytes} байт"
            )
        return value

    @model_validator(mode="after")
    def validate_segment_order(self) -> "SegmentRequest":
        if self.segmentNumber > self.totalSegments:
            raise ValueError("segmentNumber не может превышать totalSegments")
        return self


class StatusResponse(BaseModel):
    status: str = "ok"


class ErrorResponse(BaseModel):
    error: bool = True
    message: str