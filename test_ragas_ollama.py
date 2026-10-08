
import asyncio
import os

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434/v1",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# RAGAS needs an asynchronous OpenAI-compatible client.
client = AsyncOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

ragas_llm = llm_factory(
    OLLAMA_MODEL,
    client=client,
)

metric = Faithfulness(llm=ragas_llm)

async def main():
    result = await metric.ascore(
        user_input="What are the conditions for a valid contract?",
        response=(
            "A valid contract requires the conditions "
            "specified in the applicable legal context."
        ),
        retrieved_contexts=[
            (
                "The validity of a contract depends on the "
                "applicable legal requirements."
            )
        ],
    )

    print("Faithfulness score:", result.value)

if __name__ == "__main__":
    asyncio.run(main())