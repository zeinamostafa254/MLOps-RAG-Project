import asyncio
from src.rag import LegalRAG

async def main():
    rag = LegalRAG()
    question = "ما القواعد التي ينص عليها القانون المدني بشأن تنفيذ العقد؟"

    print("Question:", question)
    print("\nRunning complete RAG pipeline...")

    try:
        result = await rag.ask(question)

        print("\n=== ANSWER ===")
        print(result["answer"])

        print("\n=== SOURCES ===")
        for source in result["sources"]:
            print("-", source)

        print("\n=== SOURCE DETAILS ===")
        print("Retrieved source details:", len(result["source_details"]))
        print("Model:", result["model_version"])

        if result["answer"] and result["answer"].strip():
            print("\nPASS: Complete RAG pipeline returned an answer.")
        else:
            print("\nFAIL: The RAG pipeline returned an empty answer.")

    except Exception as exc:
        print("\nFAIL:", type(exc).__name__)
        print("Details:", str(exc))

asyncio.run(main())
