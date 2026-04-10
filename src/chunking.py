from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.
    Sentence detection: split on ". ", "! ", "? " or ".\n".
    """
    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Tách câu dựa trên dấu chấm, chấm than, hỏi chấm theo sau bởi khoảng trắng hoặc xuống dòng
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\.\n', text) if s.strip()]
        chunks = []
        
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_text = " ".join(sentences[i:i + self.max_sentences_per_chunk])
            chunks.append(chunk_text)
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.
    """
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Trả về nếu text đã đủ nhỏ hoặc hết separator để chia
        if len(current_text) <= self.chunk_size or not remaining_separators:
            return [current_text] if current_text else []

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        splits = current_text.split(sep)
        chunks = []
        current_chunk = ""

        for s in splits:
            temp = current_chunk + sep + s if current_chunk else s
            if len(temp) <= self.chunk_size:
                current_chunk = temp
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(s) > self.chunk_size:
                    # Nếu đoạn con s vẫn vượt chunk_size, đệ quy dùng separator tiếp theo
                    if next_seps:
                        chunks.extend(self._split(s, next_seps))
                    else:
                        # Fallback chia theo ký tự nếu đã hết separator
                        chunks.extend([s[i:i+self.chunk_size] for i in range(0, len(s), self.chunk_size)])
                    current_chunk = ""
                else:
                    current_chunk = s

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fs_chunks = FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text)
        bs_chunks = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        rc_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def get_stats(chunks):
            count = len(chunks)
            avg = sum(len(c) for c in chunks) / count if count > 0 else 0
            return {'count': count, 'avg_length': avg, 'chunks': chunks}

        return {
            'fixed_size': get_stats(fs_chunks),
            'by_sentences': get_stats(bs_chunks),
            'recursive': get_stats(rc_chunks)
        }
