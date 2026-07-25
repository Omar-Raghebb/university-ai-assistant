from src import config
from src.retriever import retrieve_with_scores, format_context_with_citations
from src.llm import generate_text
from src.output_parser import parse_structured_answer

try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    HAS_RERANKER = True
except Exception:
    HAS_RERANKER = False
    reranker = None


def rerank_chunks(question, chunks, top_n=config.RERANK_TOP_N):
    """Re-rank chunks using cross-encoder for better relevance."""
    if not HAS_RERANKER or not chunks or len(chunks) <= top_n:
        return chunks[:top_n]

    pairs = [(question, c["content"]) for c in chunks]
    scores = reranker.predict(pairs)

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in scored[:top_n]]


prompt_template = """You are an academic advisor for the BSCS program.

Answer using ONLY the provided context.
NEVER guess or use outside knowledge.
If the information is not present in the context, say so.

CRITICAL VALIDATION RULES:

1. Before answering ANY question about a specific course, verify that the context contains:
   - The exact course code (e.g., CS 3301)
   - The prerequisites listed for that course

   If the context only mentions the course name but NOT its prerequisites, you MUST answer:
   "I could not find the prerequisite information for this course in the provided documents."

2. For prerequisite questions, look for:
   - [COURSE_BLOCK] sections
   - [PREREQ_MATRIX] or [TABLE_ROW] sections
   - Text that says "Prerequisites:" followed by course codes

3. Copy all course codes EXACTLY as they appear in the context.
   - CS 3301 is NOT the same as CS304
   - MATH 2301 is NOT the same as MATH 2302
   - Never abbreviate or change course codes

4. If a table row says:
   "Course: CS 3301 | Prerequisites: CS 2201, CS 2401, MATH 2301 | MinGrade: C"
   Then the answer is: CS 3301 requires CS 2201, CS 2401, and MATH 2301 with a minimum grade of C.

NUMERICAL RULES:
- Copy all numbers EXACTLY from the context.
- Never estimate, calculate, or round.

If the answer does NOT exist in the context, return exactly:
{{
    "answer": "I could not find the answer to your question in the available documents.",
    "source_page": "unknown",
    "confidence": "none"
}}

OUTPUT FORMAT

Return a single JSON object with exactly these three fields:

{{
    "answer": "the answer to the student's question, based only on the given context",
    "source_page": "the page number(s) the answer came from, e.g. '2' or '1, 3'",
    "confidence": "one of: high, medium, low"
}}

IMPORTANT RULES:

Return ONLY valid JSON.
Do NOT write explanations.
Do NOT write Markdown.
Do NOT use ```json.
Do NOT write any text before or after the JSON.

CONTEXT

{context}

STUDENT QUESTION ({language})

{question}

Follow these steps:
1. Find the relevant information in the context.
2. Verify the exact course codes and prerequisites.
3. Extract the exact values.
4. Answer in the same language as the question.
5. Return ONLY valid JSON.
"""


def detect_language(text):
    arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    return "Arabic" if arabic_chars > 0 else "English"


def _ensure_string(value):
    """Ensure the answer value is a string, not a list or other type."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def answer_question(vector_store, question, k=config.TOP_K_RESULTS):
    scored_chunks = retrieve_with_scores(vector_store, question, k=k)

    if not scored_chunks:
        return {
            "answer": "I could not find the answer to your question in the available documents.",
            "source_page": "unknown",
            "confidence": "low",
            "sources": [],
            "avg_retrieval_confidence": 0.0,
        }

    best_chunks = rerank_chunks(question, scored_chunks)

    context = format_context_with_citations(best_chunks)
    language = detect_language(question)

    prompt = prompt_template.format(
        context=context,
        question=question,
        language=language,
    )

    raw_output = generate_text(prompt)
    parsed = parse_structured_answer(raw_output)

    answer = _ensure_string(parsed.get("answer", raw_output))

    avg_confidence = sum(c["confidence"] for c in best_chunks) / len(best_chunks)

    return {
        "answer": answer,
        "source_page": parsed.get("source_page", "unknown"),
        "confidence": parsed.get("confidence", "unknown"),
        "sources": best_chunks,
        "avg_retrieval_confidence": round(avg_confidence, 3),
    }