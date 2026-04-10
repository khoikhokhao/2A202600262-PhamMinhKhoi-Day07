from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve top-k relevant chunks from the store
        results = self.store.search(question, top_k=top_k)
        
        # 2. Build context string
        context_blocks = [r["content"] for r in results]
        context_str = "\n".join(context_blocks)
        
        # 3. Build a prompt
        prompt = (
            f"Please answer the following question using only the provided context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        
        # 4. Call the LLM to generate an answer
        return self.llm_fn(prompt)
