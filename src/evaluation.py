from src.vector_store import load_vector_store
from src.chain import answer_question

# edit this with real questions/keywords from your actual regulation PDF
TEST_SET = [
    {
        "question": "كام ساعة معتمدة مطلوبة للتخرج؟",
        "expected_keywords": ["ساعة", "معتمدة"],
    },
    {
        "question": "What is the minimum GPA required to avoid academic probation?",
        "expected_keywords": ["GPA", "probation"],
    },
]


def keyword_hit(answer, keywords):
    answer_lower = answer.lower()
    return any(k.lower() in answer_lower for k in keywords)


def run_evaluation(test_set=TEST_SET):
    vector_store = load_vector_store()
    results = []

    for case in test_set:
        result = answer_question(vector_store, case["question"])
        hit = keyword_hit(result["answer"], case["expected_keywords"])

        results.append({
            "question": case["question"],
            "answer": result["answer"],
            "model_confidence": result["confidence"],
            "avg_confidence": result["avg_retrieval_confidence"],
            "keyword_hit": hit,
        })

    # Summary stats
    total = len(results)
    hits = sum(1 for r in results if r["keyword_hit"])
    avg_conf = sum(r["avg_confidence"] for r in results) / total if total else 0

    print("=" * 50)
    print(f"Keyword grounding rate: {hits}/{total} ({100*hits/total:.1f}%)")
    print(f"Average retrieval confidence: {avg_conf:.3f}")
    print("=" * 50)

    for r in results:
        status = "PASS" if r["keyword_hit"] else "FAIL"
        print(f"\n[{status}] Q: {r['question']}")
        print(f"Confidence: {r['avg_confidence']}")
        print(f"A: {r['answer'][:200]}...")

    return results


if __name__ == "__main__":
    # python -m src.evaluation
    run_evaluation()