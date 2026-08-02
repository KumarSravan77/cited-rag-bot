from io import BytesIO

from reportlab.pdfgen.canvas import Canvas

from app.pdf_ingest import extract_pdf


def sample_pdf():
    buffer = BytesIO()
    canvas = Canvas(buffer)
    canvas.drawString(72, 720, "First page refund policy.")
    canvas.showPage()
    canvas.drawString(72, 720, "Second page backup schedule.")
    canvas.save()
    return buffer.getvalue()


def test_pdf_extraction_preserves_page_numbers():
    rows = extract_pdf(sample_pdf(), "../../handbook.pdf")
    assert rows[0][0] == "handbook.pdf"
    assert rows[0][1] == 1
    assert rows[-1][1] == 2
