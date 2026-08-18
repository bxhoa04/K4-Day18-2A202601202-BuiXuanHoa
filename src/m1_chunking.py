from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


# Cache embedding model for semantic chunking
_semantic_model = None


def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    import numpy as np
    metadata = metadata or {}
    
    # Split text thành sentences / units
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    model = _get_semantic_model()
    embeddings = model.encode(sentences)

    def _cosine_sim(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b + 1e-9))

    chunks_text = []
    current_sentences = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunks_text.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
        else:
            current_sentences.append(sentences[i])
    if current_sentences:
        chunks_text.append(" ".join(current_sentences))

    return [
        Chunk(text=t, metadata={**metadata, "strategy": "semantic", "chunk_index": idx})
        for idx, t in enumerate(chunks_text)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    # Group paragraphs into parent texts
    parent_texts = []
    current_parent = ""
    for para in paragraphs:
        if len(current_parent) + len(para) > parent_size and current_parent:
            parent_texts.append(current_parent.strip())
            current_parent = ""
        current_parent += para + "\n\n"
    if current_parent.strip():
        parent_texts.append(current_parent.strip())

    parents = []
    children = []

    for p_idx, p_text in enumerate(parent_texts):
        pid = f"parent_{p_idx}"
        p_meta = {**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": p_idx}
        parents.append(Chunk(text=p_text, metadata=p_meta))

        # Split parent into smaller child chunks
        p_paras = [x.strip() for x in p_text.split("\n\n") if x.strip()]
        child_units = []
        for para in p_paras:
            if len(para) > child_size:
                # Split large paragraph into sentences
                s_list = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
                child_units.extend(s_list if s_list else [para])
            else:
                child_units.append(para)

        current_child = ""
        for unit in child_units:
            if len(current_child) + len(unit) > child_size and current_child:
                c_meta = {**metadata, "chunk_type": "child", "chunk_index": len(children)}
                children.append(Chunk(text=current_child.strip(), metadata=c_meta, parent_id=pid))
                current_child = ""
            current_child += (unit + "\n\n" if "\n" not in unit else unit + " ")
        if current_child.strip():
            c_meta = {**metadata, "chunk_type": "child", "chunk_index": len(children)}
            children.append(Chunk(text=current_child.strip(), metadata=c_meta, parent_id=pid))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    cur_head = ""
    cur_lines: list[str] = []

    for line in lines:
        m = re.match(r'^(#{1,3})\s+(.+)$', line.strip())
        if m:
            if cur_head or cur_lines:
                body = "\n".join(cur_lines).strip()
                full_text = f"{cur_head}\n\n{body}".strip() if cur_head else body
                if full_text:
                    sections.append((cur_head, full_text))
            cur_head = line.strip()
            cur_lines = []
        else:
            cur_lines.append(line)

    if cur_head or cur_lines:
        body = "\n".join(cur_lines).strip()
        full_text = f"{cur_head}\n\n{body}".strip() if cur_head else body
        if full_text:
            sections.append((cur_head, full_text))

    if not sections and text.strip():
        sections.append(("", text.strip()))

    chunks = []
    for idx, (head, full_text) in enumerate(sections):
        c_meta = {**metadata, "strategy": "structure", "chunk_index": idx}
        if head:
            section_name = re.sub(r'^#+\s*', '', head).strip()
            c_meta["section"] = section_name
        chunks.append(Chunk(text=full_text, metadata=c_meta))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
