import asyncio
from src.config import get_settings
from src.llm import LLMClient

async def main():
    settings = get_settings()

    print("=== LLM CONFIGURATION ===")
    print("Model:", settings.vllm_model)
    print("Base URL:", settings.vllm_base_url)

    context = [
        {
            "metadata": {
                "citation": "Diagnostic test context",
                "text_ar": (
                    "هذا نص قانوني تجريبي لاختبار توليد الإجابة. "
                    "إذا سُئلت عن النص، اذكر أنه نص تجريبي."
                ),
            }
        }
    ]

    print("\n=== GENERATION TEST ===")
    try:
        answer = await LLMClient().generate(
            "ما طبيعة النص المقدم؟",
            context,
        )
        print("Answer:", repr(answer))
        print("Answer length:", len(answer or ""))

        if not answer or not answer.strip():
            print("FAIL: The model returned an empty answer.")
        else:
            print("PASS: The model returned non-empty content.")

    except Exception as exc:
        print("ERROR:", type(exc).__name__)
        print("Details:", str(exc))

asyncio.run(main())
