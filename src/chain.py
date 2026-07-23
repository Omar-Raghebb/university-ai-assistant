from src import config
from src.retriever import retrieve_with_scores, format_context_with_citations
from src.llm import generate_text
from src.output_parser import format_instructions, parse_structured_answer

# Re-ranking with cross-encoder
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False
    reranker = None


def rerank_chunks(question, chunks, top_n=4):
    """Re-rank chunks using cross-encoder for better relevance."""
    if not HAS_RERANKER or not chunks or len(chunks) <= top_n:
        return chunks[:top_n]
    
    pairs = [(question, c["content"]) for c in chunks]
    scores = reranker.predict(pairs)
    
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in scored[:top_n]]


# Improved prompt with explicit table handling
prompt_template = """You are an academic advisor for Sadat Academy CS program.

Answer using ONLY the provided context.
NEVER guess or use outside knowledge.
If the information is not present in the context, say so.

CRITICAL TABLE READING RULES:

Each table row is labeled like:

"[TABLE_ROW] الساعات_المعتمدة: 3 | اسم_المقرر: أساسيات البرمجة | كود_المقرر: CS101 | المتطلب_السابق: لا يوجد"

Interpret the fields as follows:

- "كود_المقرر" = Course Code (the course itself)
- "المتطلب_السابق" = Prerequisite(s) required BEFORE taking that course

Example:

If a row contains:

كود_المقرر: CS305
المتطلب_السابق: CS204, CS304

This means:

CS305 requires CS204 and CS304.

It DOES NOT mean CS204 requires CS305.

Never confuse a course with its prerequisites.

NUMERICAL RULES:

- Copy all numbers EXACTLY from the context.
- Never estimate.
- Never calculate.
- Never round.

Example:

"141 ساعة معتمدة"

must remain exactly

"141 ساعة معتمدة"

If the answer does NOT exist in the context, return exactly:

{{
    "answer": "لم أجد معلومات متعلقة بسؤالك في المستندات المتاحة.",
    "source_page": "unknown",
    "confidence": "none"
}}

OUTPUT FORMAT

{format_instructions}

IMPORTANT RULES:

Return ONLY valid JSON.

Example:

{{
    "answer": "...",
    "source_page": "3",
    "confidence": "high"
}}

Do NOT write explanations.

Do NOT write Markdown.

Do NOT use ```json.

Do NOT write any text before or after the JSON.

CONTEXT

{context}

STUDENT QUESTION ({language})

{question}

Follow these steps:

1. Find the relevant information.
2. Extract the exact values.
3. Answer in the same language as the question.
4. Return ONLY valid JSON.
"""


def detect_language(text):
    arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    return "Arabic" if arabic_chars > 0 else "English"


def answer_question(vector_store, question, k=config.TOP_K_RESULTS):
    # 1. Retrieve chunks
    scored_chunks = retrieve_with_scores(vector_store, question, k=k)
    
    if not scored_chunks:
        return {
            "answer": "لم أجد معلومات متعلقة بسؤالك في المستندات المتاحة.",
            "source_page": "unknown",
            "confidence": "low",
            "sources": [],
            "avg_retrieval_confidence": 0.0,
        }

    # 2. Re-rank to get most relevant chunks
    best_chunks = rerank_chunks(question, scored_chunks, top_n=4)
    
    # 3. Build context
    context = format_context_with_citations(best_chunks)
    language = detect_language(question)


    prompt = prompt_template.format(
        context=context,
        question=question,
        language=language,
        format_instructions=format_instructions,
    )

    # 4. Generate and parse
    raw_output = generate_text(prompt)
    parsed = parse_structured_answer(raw_output)

    # 5. Calculate confidence
    avg_confidence = sum(c["confidence"] for c in best_chunks) / len(best_chunks)

    return {
        "answer": parsed.get("answer", raw_output),
        "source_page": parsed.get("source_page", "unknown"),
        "confidence": parsed.get("confidence", "unknown"),
        "sources": best_chunks,
        "avg_retrieval_confidence": round(avg_confidence, 3),
    }