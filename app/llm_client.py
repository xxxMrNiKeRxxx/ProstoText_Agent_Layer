import json
from typing import AsyncIterator

import httpx

SYSTEM_PROMPT = (
    "Ты помощник системы «ПростоТекст». Твоя задача — переписать сложный научный или "
    "технический текст на простой, понятный русский язык без потери смысла. "
    "Пиши короткими и ясными предложениями. Не используй Markdown-разметку, списки, "
    "заголовки и вводные фразы вроде «Вот упрощённый текст». Сразу выдавай упрощённое объяснение."
)


class OllamaError(RuntimeError):
    """Ошибка взаимодействия с локальным сервером Ollama."""


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_sec: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    async def check_availability(self, client: httpx.AsyncClient) -> bool:
        """Проверяет доступность Ollama и наличие требуемой модели."""
        try:
            response = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return any(
                name == self.model or name.startswith(f"{self.model}:")
                for name in models
            )
        except (httpx.HTTPError, ValueError):
            return False

    async def stream_simplification(
        self,
        client: httpx.AsyncClient,
        complex_text: str,
    ) -> AsyncIterator[str]:
        """Выполняет потоковый запрос к Ollama (/api/chat) и отдаёт текстовые фрагменты."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": complex_text},
            ],
            "options": {
                "temperature": 0.3,
            },
        }

        try:
            async with client.stream(
                "POST",
                url,
                json=payload,
                timeout=self.timeout_sec,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise OllamaError(
                        f"Ollama вернула статус {response.status_code}: {body.decode('utf-8', errors='replace')}"
                    )

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise OllamaError(str(chunk["error"]))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise OllamaError(f"Не удалось связаться с Ollama ({url}): {exc}") from exc