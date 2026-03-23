import PyPDF2
import io


def extract_text_from_pdf(file):
    text = ""
    try:
        if hasattr(file, "read"):
            reader = PyPDF2.PdfReader(file)
        else:
            reader = PyPDF2.PdfReader(io.BytesIO(file))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    except Exception:
        text = ""
    return text.strip()
