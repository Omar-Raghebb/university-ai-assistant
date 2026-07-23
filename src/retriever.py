from src import config


def retrieve_with_scores(vector_store, query, k=config.TOP_K_RESULTS):
    results = vector_store.similarity_search_with_score(query, k=k)

    scored = []
    for doc, distance in results:
        # FAISS uses L2 distance, convert to similarity score
        # Lower distance = higher similarity
        confidence = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
        scored.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "confidence": round(confidence, 3),
        })

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

        # Add structure hints
        tag = f"[Source {i}: {source_name}, page {page}]"
        blocks.append(f"{tag}\n{content}")

    return "\n\n---\n\n".join(blocks)