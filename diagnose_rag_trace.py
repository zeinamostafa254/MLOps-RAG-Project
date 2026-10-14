import asyncio
import traceback
from src.rag import LegalRAG

async def main():
    try:
        rag = LegalRAG()
        result = await rag.ask(
            "ما القواعد التي ينص عليها القانون المدني بشأن تنفيذ العقد؟"
        )
        print("ANSWER:", result["answer"])
        print("SOURCES:", result["sources"])
    except Exception:
        traceback.print_exc()

asyncio.run(main())
