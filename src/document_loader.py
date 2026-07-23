# src/document_loader.py
import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from src import config


def extract_table_rows(text):
    """
    Find table rows in the PDF text and label each cell with its column.
    This is critical because PDF text extraction loses table structure.
    """

    row_pattern = r'(\d+)\s*\(\s*([^)]+)\s*\)\s*([A-Z]+\d+)\s*([^\n$]+)'

    # Check if this text contains a table header
    has_table = 'الساعات المعتمدة' in text and 'كود المقرر' in text

    if not has_table:
        return text

    def label_row(match):
        hours = match.group(1).strip()
        name = match.group(2).strip()
        code = match.group(3).strip()
        prereq = match.group(4).strip()

        # Add explicit labels so each row is self-contained
        return (
            f"[TABLE_ROW] "
            f"الساعات_المعتمدة: {hours} | "
            f"اسم_المقرر: {name} | "
            f"كود_المقرر: {code} | "
            f"المتطلب_السابق: {prereq}"
        )

    # Replace raw rows with labeled rows
    labeled = re.sub(row_pattern, label_row, text)
    return labeled


def preprocess_pdf_text(text):
    """
    Pre-process PDF text to fix common extraction issues:
    1. Label table rows
    2. Fix concatenated Arabic text
    3. Add section markers
    """
    # Label tables
    text = extract_table_rows(text)

    # Add section markers for important sections
    text = re.sub(r'(#\s*الفصل\s+[\w\s]+)', r'\n\n[SECTION] \1\n', text)

    text = re.sub(r'(?<!\d)(\d)\s(\d{2})(?!\d)', r'\1\2', text)

    return text


def load_pdfs(data_dir=config.DATA_DIR):
    """Load all PDFs from data directory with preprocessing."""
    docs = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith('.pdf'):
            path = os.path.join(data_dir, filename)
            loader = PyPDFLoader(path)
            raw_docs = loader.load()

            for doc in raw_docs:
                # Pre-process each page's text
                doc.page_content = preprocess_pdf_text(doc.page_content)
                docs.append(doc)

    return docs


def split_documents(docs):
    """Split documents into chunks while preserving table structure."""
    # Use smaller chunks with more overlap for tables
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " | ", " ", ""],
        # Keep table rows together
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(docs)

    # Post-process: ensure each chunk with table rows has context
    for chunk in chunks:
        if '[TABLE_ROW]' in chunk.page_content and '[SECTION]' not in chunk.page_content:
            # Add a note that this is from the course table
            chunk.page_content = (
                "[FROM_COURSE_TABLE] " + chunk.page_content
            )

    return chunks