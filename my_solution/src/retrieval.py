import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
 
REQUIRED_FIELDS = ["id", "context", "question", "answer", "reference_ids"]
 
EVAL_FIELDS = {
    "id": str,
    "question": str,
    "expected_sources": list,
    "required_terms": list,
    "forbidden_terms": list,
    "should_abstain": bool,
}
 
 
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
 
 
def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
 
 
# ---------------------------------------------------------------- documents
 
@dataclass(frozen=True)
class Document:
    document_id: str   # file name without .md, matches reference_ids / expected_sources
    title: str
    text: str
 
 
def load_documents(directory) -> list[Document]:
    """Read every .md file; the first '# ' heading becomes the title."""
    docs = []
    for path in sorted(Path(directory).glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        first, _, rest = raw.partition("\n")
        if first.startswith("# "):
            title, body = first[2:].strip(), rest.strip()
        else:
            title, body = path.stem.replace("_", " ").title(), raw
        docs.append(Document(path.stem, title, body))
    return docs
 
 
# ---------------------------------------------------------------- chunking
 
@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # "<document_id>#<n>", used for citations
    document_id: str
    title: str
    text: str
 
    @property
    def position(self) -> int:
        return int(self.chunk_id.rsplit("#", 1)[1])
 
 
def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split a paragraph longer than max_chars on sentence boundaries."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    parts: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts
 
 
MAX_CHARS = 1000      # whole KB articles fit; stays under MiniLM's 256-token limit
OVERLAP_CHARS = 300   # carry the previous paragraph over only if it is this short
 
 
def chunk_document(
    doc: Document, max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[Chunk]:
    """Pack paragraphs into chunks of at most ~max_chars characters.
 
    When a document splits, each new chunk starts with the previous chunk's
    last paragraph if that paragraph is <= overlap_chars and still fits.
    """
    paragraphs = [
        piece
        for block in doc.text.split("\n\n")
        if block.strip()
        for piece in split_long_paragraph(block.strip(), max_chars)
    ]
    groups: list[list[str]] = [[]]
    size = 0
    for para in paragraphs:
        if groups[-1] and size + len(para) > max_chars:
            carry = groups[-1][-1]
            if len(carry) <= overlap_chars and len(carry) + 2 + len(para) <= max_chars:
                groups.append([carry])
                size = len(carry) + 2
            else:
                groups.append([])
                size = 0
        groups[-1].append(para)
        size += len(para) + 2  # account for the "\n\n" joiner
    return [
        Chunk(f"{doc.document_id}#{i}", doc.document_id, doc.title, "\n\n".join(g))
        for i, g in enumerate(groups, start=1)
        if g  # a document with no body text yields no chunks
    ]
 
 
def chunk_documents(
    docs: list[Document], max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[Chunk]:
    chunks = [c for doc in docs for c in chunk_document(doc, max_chars, overlap_chars)]
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs; check for duplicate document IDs")
    return chunks
 
 
# ---------------------------------------------------------------- vector index
 
COLLECTION_NAME = "knowledge_base"
 
 
def load_embedder(model_name: str, revision: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, revision=revision)


def embed(embedder, texts: list[str]):
    return embedder.encode(texts, normalize_embeddings=True).astype("float32")


@dataclass
class VectorIndex:
    """FAISS inner-product index; on normalized vectors that is cosine similarity."""
    index: object          # faiss.IndexFlatIP
    chunks: list[Chunk]    # row i in the index belongs to chunks[i]

    def count(self) -> int:
        return self.index.ntotal


def build_index(chunks: list[Chunk], embedder, persist_dir=None) -> VectorIndex:
    """Embed chunks into a FAISS index; save it to persist_dir if given."""
    import faiss

    # Title + text is embedded so the title also helps matching.
    vectors = embed(embedder, [f"{c.title}\n\n{c.text}" for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    if persist_dir is not None:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(persist_dir / "index.faiss"))
        with open(persist_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    return VectorIndex(index, chunks)


def open_index(persist_dir) -> VectorIndex:
    import faiss

    persist_dir = Path(persist_dir)
    index = faiss.read_index(str(persist_dir / "index.faiss"))
    chunks = [Chunk(**row) for row in read_jsonl(persist_dir / "chunks.jsonl")]
    return VectorIndex(index, chunks)


def retrieve(
    vector_index: VectorIndex, embedder, question: str, k: int = 3, min_score: float = 0.0
) -> list[dict]:
    """Return up to k chunks with cosine similarity >= min_score, best first."""
    scores, rows = vector_index.index.search(embed(embedder, [question]), k)
    hits = []
    for score, row in zip(scores[0], rows[0]):
        if row == -1 or score < min_score:   # -1 = fewer than k chunks in the index
            continue
        c = vector_index.chunks[row]
        hits.append({
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "title": c.title,
            "position": c.position,
            "text": c.text,
            "score": float(score),
        })
    return hits
 

ABSTAIN_MESSAGE = (
    "I can't answer that from the knowledge base. "
    "Please contact support or rephrase your question."
)


@dataclass(frozen=True)
class Evidence:
    question: str
    hits: list[dict]       # chunks with score >= min_score, best first
    top_score: float       # best raw score, even if below the threshold

    @property
    def abstain(self) -> bool:
        return not self.hits

    @property
    def sources(self) -> list[str]:
        return sorted({h["document_id"] for h in self.hits})


def gather_evidence(
    vector_index: VectorIndex, embedder, question: str, top_k: int, min_score: float
) -> Evidence:
    """Retrieve top_k chunks and keep only those above the evidence threshold.

    If nothing clears min_score, the result abstains: the nearest weak match
    is never passed on as evidence.
    """
    candidates = retrieve(vector_index, embedder, question, k=top_k, min_score=-1.0)
    top_score = candidates[0]["score"] if candidates else 0.0
    hits = [h for h in candidates if h["score"] >= min_score]
    return Evidence(question, hits, top_score)


def evaluate_retrieval(cases, vector_index, embedder, top_k: int, min_score: float) -> dict:
    """Check abstention and expected sources for every evaluation case."""
    rows = []
    for case in cases:
        ev = gather_evidence(vector_index, embedder, case["question"], top_k, min_score)
        abstain_ok = ev.abstain == case["should_abstain"]
        sources_ok = case["should_abstain"] or set(case["expected_sources"]) <= set(ev.sources)
        rows.append({
            "id": case["id"],
            "should_abstain": case["should_abstain"],
            "abstained": ev.abstain,
            "top_score": round(ev.top_score, 3),
            "sources": ev.sources,
            "passed": abstain_ok and sources_ok,
        })
    passed = sum(r["passed"] for r in rows)
    return {"passed": passed, "total": len(rows), "cases": rows}

# ---------------------------------------------------------------- validation
 
def find_duplicate_ids(items):
    counts = Counter(item.get("id") for item in items)
    return [i for i, n in counts.items() if n > 1]
 
 
def validate_examples(examples):
    errors = []
    for i, ex in enumerate(examples):
        missing = [k for k in REQUIRED_FIELDS if ex.get(k) in (None, "", [])]
        if missing:
            errors.append(f"Example {i}: missing {missing}")
    for dup in find_duplicate_ids(examples):
        errors.append(f"Duplicate id {dup!r}")
    return errors
 
 
def _normalize(text):
    return " ".join(str(text).lower().split())
 
 
def validate_evaluation(evaluation, training):
    errors = [f"Duplicate eval id {d!r}" for d in find_duplicate_ids(evaluation)]
 
    for i, case in enumerate(evaluation):
        for field, ftype in EVAL_FIELDS.items():
            if field not in case:
                errors.append(f"Eval case {i}: missing {field!r}")
            elif not isinstance(case[field], ftype):
                errors.append(f"Eval case {i}: {field!r} should be {ftype.__name__}")
        if case.get("id") == "" or case.get("question") == "":
            errors.append(f"Eval case {i}: empty id or question")
        if case.get("should_abstain") is False and not case.get("expected_sources"):
            errors.append(f"Eval case {i}: expected_sources empty but should_abstain is false")
 
    train_ids = {ex.get("id") for ex in training}
    train_questions = {_normalize(ex["question"]) for ex in training if ex.get("question")}
 
    for case in evaluation:
        cid = case.get("id")
        if cid in train_ids:
            errors.append(f"Eval case {cid!r}: id also in training")
        if case.get("question") and _normalize(case["question"]) in train_questions:
            errors.append(f"Eval case {cid!r}: question also in training")
    return errors
 
 
def validate_sources(training, evaluation, source_ids):
    errors = []
    for ex in training:
        for ref in ex.get("reference_ids") or []:
            if ref not in source_ids:
                errors.append(f"Training {ex.get('id')!r}: unknown source {ref!r}")
    for case in evaluation:
        for ref in case.get("expected_sources") or []:
            if ref not in source_ids:
                errors.append(f"Eval case {case.get('id')!r}: unknown source {ref!r}")
    return errors
 