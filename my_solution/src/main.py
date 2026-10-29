
from pathlib import Path
from retrieval import *
from config import SETTINGS
from training import *

PROJECT_ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    training = read_jsonl(Path(PROJECT_ROOT) / "data" / "training" / "instruction_examples.jsonl")
    evaluation = read_json(Path(PROJECT_ROOT) / "data" / "evaluation" / "cases.json")
    documents = load_documents(Path(PROJECT_ROOT) / "data" / "knowledge_base")
 
    # 2. Validate data
    errors = (
        validate_examples(training)
        + validate_evaluation(evaluation, training)
        + validate_sources(training, evaluation, {d.document_id for d in documents})
    )
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"Data OK: {len(training)} training, {len(evaluation)} eval, {len(documents)} docs")

    # 3. Chunk, embed, index
    chunks = chunk_documents(documents)
    embedder = load_embedder(SETTINGS.embedding_model, SETTINGS.embedding_revision)
    index = build_index(chunks, embedder, SETTINGS.vector_storage_path)
    print(f"Indexed {index.count()} chunks (top_k={SETTINGS.top_k}, min_score={SETTINGS.min_score})")

    # 4. Example query with the evidence gate
    question = "Can I get a refund 12 hours before a workshop?"
    evidence = gather_evidence(index, embedder, question, SETTINGS.top_k, SETTINGS.min_score)
    print(f"\nQ: {question}")
    if evidence.abstain:
        print(f"Abstained (top score {evidence.top_score:.3f}): {ABSTAIN_MESSAGE}")
    else:
        for hit in evidence.hits:
            print(f"  {hit['score']:.3f}  {hit['chunk_id']}  {hit['title']}")

    # 5. Retrieval evaluation
    report = evaluate_retrieval(evaluation, index, embedder, SETTINGS.top_k, SETTINGS.min_score)
    print(f"\nRetrieval eval: {report['passed']}/{report['total']} passed")
    for row in report["cases"]:
        mark = "PASS" if row["passed"] else "FAIL"
        print(
            f"  {mark}  {row['id']:<30} score={row['top_score']:.3f}  "
            f"abstain={row['abstained']} (expected {row['should_abstain']})  sources={row['sources']}"
        )

    # 6.LorA

    train();