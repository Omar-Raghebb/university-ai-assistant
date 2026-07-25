import re
import json
from langchain_classic.output_parsers.structured import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="answer", description="the answer to the student's question, based only on the given context"),
    ResponseSchema(name="source_page", description="the page number(s) the answer came from, e.g. '2' or '1, 3'"),
    ResponseSchema(name="confidence", description="one of: high, medium, low"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# NOTE: format_instructions is kept available for reuse elsewhere, but
# chain.py does NOT inject this into the prompt. langchain's default
# format instructions tell the model to wrap output in a ```json fenced
# block, which directly contradicts chain.py's "no markdown, no ```json"
# instruction. Feeding both to a small quantized model produces
# inconsistent formatting. chain.py instead describes the JSON schema
# inline in its own prompt, and this module's job is purely robust
# parsing of whatever comes back (fenced, bare, or malformed).
format_instructions = output_parser.get_format_instructions()


def extract_json_block(text):
    """Extract JSON from various formats the model might output."""
    matches = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    matches = re.findall(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    matches = re.findall(r"(\{.*?\})", text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    return text.strip()


def clean_json_text(text):
    """Clean common JSON formatting issues from model output."""
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*', '', text)

    text = text.replace('،', ',')

    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")

    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    return text


def parse_structured_answer(raw_text):
    """
    Parse model output into structured JSON.
    Falls back gracefully if parsing fails.
    """
    json_block = extract_json_block(raw_text)
    json_block = clean_json_text(json_block)

    try:
        parsed = json.loads(json_block)
        return {
            "answer": parsed.get("answer", json_block),
            "source_page": str(parsed.get("source_page", "unknown")),
            "confidence": parsed.get("confidence", "unknown"),
        }
    except json.JSONDecodeError:
        pass

    try:
        wrapped = f"```json\n{json_block}\n```"
        return output_parser.parse(wrapped)
    except Exception:
        pass

    return {
        "answer": raw_text.strip(),
        "source_page": "unknown",
        "confidence": "unknown",
    }