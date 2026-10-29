def extract_text(pdf_path):
    """
    Extracts text from all pages of a PDF document using pypdf,
    with an automatic fallback to PyMuPDF (fitz) if needed.
    """
    text = ""
    # Try pypdf first
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return text
    except Exception as e:
        print(f"pypdf extraction notice: {e}, falling back to PyMuPDF")

    # Fallback to PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
        doc.close()
    except Exception as e:
        print(f"PyMuPDF extraction notice: {e}")

    return text
