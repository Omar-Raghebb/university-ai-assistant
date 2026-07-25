from src.vector_store import load_vector_store
from src.chain import answer_question

TEST_SET = [
    {
        "question": "What are the prerequisites for CS 3301, and what minimum grade is required in each?",
        "expected_keywords": ["CS 2201", "CS 2401", "MATH 2301", "C"],
        "require_all": True,
    },
    {
        "question": "A student completed CS 1101 with a D+. Can they enroll in CS 1102? Why or why not?",
        "expected_keywords": ["no", "C", "grade", "prerequisite"],
        "require_all": False,
    },
    {
        "question": "How many total credit hours are required under the BSCS graduation rules?",
        "expected_keywords": ["120"],
        "require_all": True,
    },
    {
        "question": "What is the minimum major GPA required to qualify for graduation?",
        "expected_keywords": ["2.5", "major"],
        "require_all": True,
    },
    {
        "question": "Which two courses form the required senior capstone sequence?",
        "expected_keywords": ["CS 4901", "CS 4902"],
        "require_all": True,
    },
    {
        "question": "What happens to a student whose cumulative GPA falls below 2.0 for two consecutive semesters?",
        "expected_keywords": ["suspension", "academic"],
        "require_all": False,
    },
    {
        "question": "What is the minimum grade required in CS 2201 to proceed to CS 3301?",
        "expected_keywords": ["C", "CS 2201", "CS 3301"],
        "require_all": True,
    },
    {
        "question": "List the three specialization tracks mentioned in the handbook.",
        "expected_keywords": ["Artificial Intelligence", "Cybersecurity", "Software Engineering"],
        "require_all": True,
    },
    {
        "question": "What is the maximum number of transfer credit hours allowed toward the BSCS degree?",
        "expected_keywords": ["60", "transfer"],
        "require_all": True,
    },
    {
        "question": "Can a student on academic probation enroll in CS 4901? Cite the relevant policy.",
        "expected_keywords": ["no", "probation", "capstone"],
        "require_all": False,
    },
]


def _ensure_string(value):
    """Ensure value is a string for comparison."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def keyword_hit(answer, keywords, require_all=True):
    answer_lower = _ensure_string(answer).lower()
    if require_all:
        return all(k.lower() in answer_lower for k in keywords)
    else:
        return any(k.lower() in answer_lower for k in keywords)


def run_evaluation(test_set=TEST_SET):
    vector_store = load_vector_store()
    results = []

    for case in test_set:
        result = answer_question(vector_store, case["question"])
        require_all = case.get("require_all", True)
        hit = keyword_hit(result["answer"], case["expected_keywords"], require_all)

        results.append({
            "question": case["question"],
            "answer": result["answer"],
            "model_confidence": result["confidence"],
            "avg_confidence": result["avg_retrieval_confidence"],
            "keyword_hit": hit,
            "sources": result["sources"],
        })

    total = len(results)
    hits = sum(1 for r in results if r["keyword_hit"])
    avg_conf = sum(r["avg_confidence"] for r in results) / total if total else 0

    print("=" * 70)
    print(f"Keyword grounding rate: {hits}/{total} ({100*hits/total:.1f}%)")
    print(f"Average retrieval confidence: {avg_conf:.3f}")
    print("=" * 70)

    for r in results:
        status = "PASS" if r["keyword_hit"] else "FAIL"
        print(f"\n[{status}] Q: {r['question']}")
        print(f"Confidence: {r['avg_confidence']}")
        print(f"A: {r['answer'][:300]}...")

        if not r["keyword_hit"]:
            print("\n--- RETRIEVED CHUNKS ---")
            for i, src in enumerate(r["sources"][:3], 1):
                print(f"[{i}] Type: {src.get('chunk_type', 'unknown')} | "
                      f"Codes: {src.get('course_codes', [])} | "
                      f"Conf: {src['confidence']}")
                print(f"    Content: {src['content'][:200]}...")
            print("--- END CHUNKS ---\n")

    return results


if __name__ == "__main__":
    run_evaluation()