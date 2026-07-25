import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from src import config


def preprocess_pdf_text(text):
    """
    Pre-process PDF text to fix common extraction issues:
    1. Mark course blocks so they stay together
    2. Mark prerequisite matrix tables
    3. Mark graduation rules and GPA requirements sections
    4. Clean up broken lines
    """
    text = re.sub(
        r'(?m)^((?:CS|MATH|PHYS|ENG|COMM)\s+\d{4})\s*[-—–-]\s*([^\n]+)',
        r'\n[COURSE_BLOCK] \1 — \2',
        text
    )

    text = re.sub(
        r'(?i)(prerequisite relationship matrix|prerequisite relationships)',
        r'[PREREQ_MATRIX] \1',
        text
    )

    text = re.sub(
        r'(?i)(graduation rules|degree requirements)',
        r'[GRAD_RULES] \1',
        text
    )

    text = re.sub(
        r'(?i)(gpa requirements|minimum gpa)',
        r'[GPA_REQ] \1',
        text
    )

    text = re.sub(
        r'(?m)^((?:CS|MATH|PHYS|ENG|COMM)\s+\d{4})\s*\|\s*([^\n|]+)\s*\|\s*([A-C])\s*$',
        r'[TABLE_ROW] Course: \1 | Prerequisites: \2 | MinGrade: \3',
        text
    )

    text = re.sub(r'(?m)^(#{1,3}\s+.+)$', r'\n[SECTION] \1\n', text)

    return text


def extract_course_codes(text):
    """Extract all course codes from text for metadata tagging."""
    codes = re.findall(r'\b(CS|MATH|PHYS|ENG|COMM)\s+(\d{4})\b', text)
    return [f"{c[0]} {c[1]}" for c in codes]


def load_pdfs(data_dir=config.DATA_DIR):
    """Load all PDFs from data directory with preprocessing."""
    docs = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith('.pdf'):
            path = os.path.join(data_dir, filename)
            loader = PyPDFLoader(path)
            raw_docs = loader.load()

            for doc in raw_docs:
                doc.page_content = preprocess_pdf_text(doc.page_content)
                docs.append(doc)

    return docs


def split_documents(docs):
    """Split documents into chunks while preserving table and course structure."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " | ", " ", ""],
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        content = chunk.page_content

        if '[COURSE_BLOCK]' in content:
            chunk.metadata["chunk_type"] = "course_block"
        elif '[PREREQ_MATRIX]' in content or '[TABLE_ROW]' in content:
            chunk.metadata["chunk_type"] = "prerequisite_table"
        elif '[GRAD_RULES]' in content:
            chunk.metadata["chunk_type"] = "graduation_rules"
        elif '[GPA_REQ]' in content:
            chunk.metadata["chunk_type"] = "gpa_requirements"
        else:
            chunk.metadata["chunk_type"] = "general"

        codes = extract_course_codes(content)
        if codes:
            chunk.metadata["course_codes"] = list(set(codes))

    return chunks