from src import config
import re


def retrieve_with_scores(vector_store, query, k=config.TOP_K_RESULTS):
    """Retrieve chunks with course-code boosting for CS/MATH/PHYS queries."""
    query_codes = re.findall(r'\b(CS|MATH|PHYS|ENG|COMM)\s*(\d{4})\b', query, re.IGNORECASE)
    query_codes = [f"{c[0].upper()} {c[1]}" for c in query_codes]

    results = vector_store.similarity_search_with_score(query, k=k)

    scored = []
    for doc, distance in results:
        confidence = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

        if query_codes and doc.metadata.get("course_codes"):
            chunk_codes = [c.upper() for c in doc.metadata["course_codes"]]
            if any(qc.upper() in chunk_codes for qc in query_codes):
                confidence = min(1.0, confidence + 0.25)

        if "prerequisite" in query.lower() or "prereq" in query.lower():
            if doc.metadata.get("chunk_type") == "prerequisite_table":
                confidence = min(1.0, confidence + 0.15)

        scored.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "confidence": round(confidence, 3),
            "chunk_type": doc.metadata.get("chunk_type", "unknown"),
            "course_codes": doc.metadata.get("course_codes", []),
        })

    scored.sort(key=lambda x: x["confidence"], reverse=True)
    return scored


def format_context_with_citations(scored_chunks):
    """
    Format chunks with clear citations and structure markers.
    Each chunk is clearly separated and labeled.
    """
    blocks = []
    for i, chunk in enumerate(scored_chunks, start=1):
        source_name = chunk["source"].split("/")[-1].split("\\")[-1]
        page = chunk["page"]
        content = chunk["content"]
        chunk_type = chunk.get("chunk_type", "")
        codes = chunk.get("course_codes", [])

        type_hint = f"[Type: {chunk_type}]" if chunk_type else ""
        codes_hint = f"[Courses: {', '.join(codes)}]" if codes else ""
        tag = f"[Source {i}: {source_name}, page {page}] {type_hint} {codes_hint}"
        blocks.append(f"{tag}\n{content}")

    return "\n\n---\n\n".join(blocks)