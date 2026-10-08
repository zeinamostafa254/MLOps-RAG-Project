
import httpx
from .config import get_settings


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def generate(self, question: str, context: list[dict]) -> str:
        sources = "\n\n".join(
            f"[{h['metadata']['citation']}]\n{h['metadata']['text_ar']}"
            for h in context
        )
        prompt = f"""
You are an Egyptian legal-document QA assistant.
Answer ONLY from the supplied context.
If the context does not contain enough evidence, say that the available corpus does not
contain enough information. Do not invent legal rules, article numbers, dates, or citations.
Answer in Arabic unless the user asks in English.
Cite every substantive legal claim as [Article N].

Question:
{question}

Context:
{sources}
""".strip()

        payload = {
            "model": self.settings.vllm_model,
            "messages": [
                {"role": "system", "content": "You are a precise legal RAG assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 700,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.settings.vllm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.vllm_api_key}"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
