"""
build_embeddings.py — Precompute semantic embeddings for the PubMed KB

Run ONCE after fetching new papers (or whenever you change the embedding model).
Output: agents/pubmed_embeddings.npy + agents/pubmed_embeddings_meta.json

Usage:
    python build_embeddings.py

Time: ~5-15 min for 2527 papers on CPU (one-time cost).
After that, RAGAgent loads the precomputed embeddings in ~3 seconds.

Model: paraphrase-multilingual-MiniLM-L12-v2
- Multilingual (50+ languages including es/en/fr/ar)
- Small (~470MB), fast on CPU
- 384-dim embeddings
- Trained on paraphrase pairs → great for semantic similarity
"""

import json
import os
import sys
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
KB_FILE = os.path.join("agents", "pubmed_knowledge_base.json")
EMBEDDINGS_FILE = os.path.join("agents", "pubmed_embeddings.npy")
META_FILE = os.path.join("agents", "pubmed_embeddings_meta.json")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32  # papers per batch — increase if you have lots of RAM
MAX_ABSTRACT_CHARS = 2000  # truncate very long abstracts


def build_paper_text(paper: dict) -> str:
    """
    Combine title + abstract into a single text for embedding.
    Title is repeated to give it more weight (titles are usually
    more concise and topical than abstracts).
    """
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()

    # Truncate very long abstracts (keeps embedding quality + speed)
    if len(abstract) > MAX_ABSTRACT_CHARS:
        abstract = abstract[:MAX_ABSTRACT_CHARS]

    # Title weighted x2 by repetition — works well in practice
    return f"{title}. {title}. {abstract}".strip()


def main():
    print("=" * 70)
    print("BUILD EMBEDDINGS — PubMed Knowledge Base")
    print("=" * 70)

    # ── Load papers ──────────────────────────────────────────────────────
    if not os.path.exists(KB_FILE):
        print(f"❌ Knowledge base not found at: {KB_FILE}")
        print("   Run pubmed_fetcher.py first.")
        sys.exit(1)

    print(f"\n📂 Loading papers from {KB_FILE}...")
    with open(KB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data.get("papers", [])
    if not papers:
        print("❌ No papers found in knowledge base.")
        sys.exit(1)

    print(f"   {len(papers)} papers loaded.")

    # ── Build texts for embedding ────────────────────────────────────────
    print(f"\n📝 Preparing texts (title + abstract)...")
    texts = [build_paper_text(p) for p in papers]

    # Quick sanity check
    empty_count = sum(1 for t in texts if len(t) < 20)
    if empty_count > 0:
        print(f"   ⚠️  {empty_count} papers have very short text (will still be embedded)")

    avg_len = sum(len(t) for t in texts) / len(texts)
    print(f"   Avg text length: {avg_len:.0f} chars")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n🤖 Loading embedding model: {MODEL_NAME}")
    print("   (First run downloads ~470MB — please be patient)")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"   Model loaded in {time.time() - t0:.1f}s")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # ── Encode in batches ────────────────────────────────────────────────
    print(f"\n⚙️  Encoding {len(texts)} papers in batches of {BATCH_SIZE}...")
    print("   (this is the slow part — ~5-15 min on CPU)")
    t0 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # ← KEY: enables cosine similarity via dot product
    )

    elapsed = time.time() - t0
    print(f"\n   ✅ Encoded in {elapsed:.1f}s ({elapsed / len(texts) * 1000:.1f}ms per paper)")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Memory size: {embeddings.nbytes / 1024 / 1024:.1f} MB")

    # ── Save embeddings ──────────────────────────────────────────────────
    print(f"\n💾 Saving to {EMBEDDINGS_FILE}...")
    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)

    # ── Save metadata (for verification & rebuild detection) ─────────────
    meta = {
        "model_name": MODEL_NAME,
        "embedding_dim": int(embeddings.shape[1]),
        "num_papers": int(embeddings.shape[0]),
        "normalized": True,
        "kb_file": KB_FILE,
        "kb_total_papers": len(papers),
        "build_time_seconds": round(elapsed, 1),
        "max_abstract_chars": MAX_ABSTRACT_CHARS,
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"💾 Metadata saved to {META_FILE}")

    # ── Done ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print(f"   Embeddings file: {EMBEDDINGS_FILE} ({os.path.getsize(EMBEDDINGS_FILE) / 1024 / 1024:.1f} MB)")
    print(f"   Metadata file:   {META_FILE}")
    print(f"\n   Next step: replace agents/rag_agent.py with v3.0")
    print(f"   Then run: python api.py")
    print()


if __name__ == "__main__":
    main()