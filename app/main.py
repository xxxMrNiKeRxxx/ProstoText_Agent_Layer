import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import settings
from app.llm_client import OllamaClient, OllamaError
from app.models import (
    ErrorResponse,
    SegmentRequest,
    SimplifiedPartMessage,
    StatusResponse,
    TransferRequest,
)
from app.segmenter import (
    SentenceAccumulator,
    UniqueSendTimeGenerator,
    build_segments_for_part,
)
from app.transport_client import TransportClient, TransportError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent_service")

ollama_client = OllamaClient(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    timeout_sec=settings.ollama_timeout_sec,
)
transport_client = TransportClient(
    transfer_url=settings.transport_transfer_url,
    timeout_sec=settings.transport_timeout_sec,
)
time_generator = UniqueSendTimeGenerator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient()
    logger.info(
        "Агентный сервис запущен (модель=%s, транспорт=%s, лимит сегмента=%d байт)",
        settings.ollama_model,
        settings.transport_transfer_url,
        settings.max_segment_bytes,
    )
    try:
        yield
    finally:
        await app.state.http_client.aclose()


app = FastAPI(
    title="Веб-сервис агентного уровня (Вариант 9 — Упрощение сложных текстов с помощью LLM)",
    description=(
        "API сервиса агентного уровня распределённой системы «ПростоТекст» "
        "на базе локальной языковой модели Qwen2.5-3B-Instruct (Ollama)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    """Возвращает HTTP 400 в формате ErrorResponse согласно swagger_agent.yaml."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=True,
            message=f"Ошибка валидации входного JSON: {exc.errors()}",
        ).model_dump(),
    )


async def _process_and_stream_text(
    http_client: httpx.AsyncClient,
    request: TransferRequest,
) -> None:
    """Фоновый конвейер: генерация LLM -> группировка по 2–3 предложения -> сегментация -> отправка."""
    request_id = request.effective_request_id
    accumulator = SentenceAccumulator(sentences_per_part=settings.sentences_per_part)
    part_number = 0

    try:
        async for token in ollama_client.stream_simplification(http_client, request.complexText):
            for text_part in accumulator.feed(token):
                part_number += 1
                part_msg = SimplifiedPartMessage(
                    sender=request.sender,
                    requestId=request_id,
                    partNumber=part_number,
                    simplifiedText=text_part,
                    isFinal=False,
                )
                segments = build_segments_for_part(
                    part=part_msg,
                    send_time=time_generator.next_time(),
                    max_bytes=settings.max_segment_bytes,
                )
                await transport_client.send_segments(http_client, segments)

        for tail_part in accumulator.flush():
            part_number += 1
            part_msg = SimplifiedPartMessage(
                sender=request.sender,
                requestId=request_id,
                partNumber=part_number,
                simplifiedText=tail_part,
                isFinal=False,
            )
            segments = build_segments_for_part(
                part=part_msg,
                send_time=time_generator.next_time(),
                max_bytes=settings.max_segment_bytes,
            )
            await transport_client.send_segments(http_client, segments)

        final_part = SimplifiedPartMessage(
            sender=request.sender,
            requestId=request_id,
            partNumber=part_number + 1,
            simplifiedText="",
            isFinal=True,
        )
        final_segments = build_segments_for_part(
            part=final_part,
            send_time=time_generator.next_time(),
            max_bytes=settings.max_segment_bytes,
        )
        await transport_client.send_segments(http_client, final_segments)
        logger.info(
            "Генерация завершена для requestId=%s (частей с текстом: %d)",
            request_id,
            part_number,
        )
    except (OllamaError, TransportError) as exc:
        logger.error("Сбой обработки запроса requestId=%s: %s", request_id, exc)
    except Exception:
        logger.exception("Непредвиденная ошибка при обработке запроса requestId=%s", request_id)


@app.post(
    "/transfer",
    tags=["AgentLayer"],
    response_model=StatusResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Ошибка валидации входного JSON"},
        500: {"model": ErrorResponse, "description": "Ошибка при обращении к локальной языковой модели Ollama"},
    },
    summary="Метод Transfer — приём исходного сложного текста от транспортного уровня",
)
async def transfer_complex_text(
    payload: TransferRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    http_client: httpx.AsyncClient = request.app.state.http_client

    if not await ollama_client.check_availability(http_client):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=True,
                message=f"Локальная модель Ollama ({settings.ollama_model}) недоступна",
            ).model_dump(),
        )

    background_tasks.add_task(_process_and_stream_text, http_client, payload)
    return StatusResponse(status="ok")


@app.post(
    "/send",
    tags=["AgentLayer"],
    response_model=StatusResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Некорректный формат сегмента"},
        502: {"model": ErrorResponse, "description": "Транспортный уровень недоступен"},
    },
    summary="Метод Send — последовательная передача сегментов результата на транспортный уровень",
)
async def send_segment(segment: SegmentRequest, request: Request):
    http_client: httpx.AsyncClient = request.app.state.http_client
    try:
        await transport_client.send_segment(http_client, segment)
        return StatusResponse(status="ok")
    except TransportError as exc:
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(error=True, message=str(exc)).model_dump(),
        )


@app.get("/health", tags=["Monitoring"], summary="Проверка работоспособности агентного сервиса")
async def health_check(request: Request):
    http_client: httpx.AsyncClient = request.app.state.http_client
    ollama_ready = await ollama_client.check_availability(http_client)
    return {
        "status": "ok",
        "service": "prostotext-agent",
        "model": settings.ollama_model,
        "ollamaReady": ollama_ready,
    }