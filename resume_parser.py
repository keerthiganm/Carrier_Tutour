# ============================================================
# RESUME PARSER
#
# Responsibility:
# Read uploaded resume files and convert them into plain text.
#
# Currently supported:
# - PDF
# - DOCX
#
# This file should NOT contain Streamlit UI code.
# ============================================================


import pymupdf
from docx import Document


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF resume.

    Parameters:
        uploaded_file:
            PDF uploaded through Streamlit.

    Returns:
        str:
            Plain text extracted from all PDF pages.
    """

    # Read uploaded PDF as binary data
    pdf_bytes = uploaded_file.read()

    # Open the PDF directly from memory
    pdf_document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    extracted_text = ""

    # Extract text page by page
    for page in pdf_document:

        page_text = page.get_text()

        extracted_text += page_text + "\n"

    pdf_document.close()

    return extracted_text.strip()


def extract_text_from_docx(uploaded_file):
    """
    Extract text from an uploaded DOCX resume.

    Both normal paragraphs and table cells are processed
    because resumes frequently use tables for formatting.

    Parameters:
        uploaded_file:
            DOCX file uploaded through Streamlit.

    Returns:
        str:
            Plain text extracted from the Word document.
    """

    document = Document(uploaded_file)

    extracted_parts = []

    # -------------------------
    # Extract normal paragraphs
    # -------------------------

    for paragraph in document.paragraphs:

        paragraph_text = paragraph.text.strip()

        if paragraph_text:

            extracted_parts.append(
                paragraph_text
            )

    # -------------------------
    # Extract table contents
    # -------------------------

    for table in document.tables:

        for row in table.rows:

            for cell in row.cells:

                cell_text = cell.text.strip()

                if cell_text:

                    extracted_parts.append(
                        cell_text
                    )

    # Join everything together
    return "\n".join(extracted_parts)


def extract_resume_text(uploaded_file):
    """
    Main resume extraction function.

    Detect the file type and call the correct parser.

    This gives app.py one simple function to call,
    regardless of whether the resume is PDF or DOCX.
    """

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):

        return extract_text_from_pdf(
            uploaded_file
        )

    elif file_name.endswith(".docx"):

        return extract_text_from_docx(
            uploaded_file
        )

    else:

        raise ValueError(
            "Unsupported resume format."
        )