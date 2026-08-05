import pytest


def build_pdf(page_texts):
    """Build a minimal, uncompressed PDF (one Helvetica line per page) whose
    text PyPDF2 can extract. Returns the PDF file bytes."""
    n = len(page_texts)
    objects = []
    kids = " ".join(f"{3 + i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_obj = 3 + 2 * n
    for i in range(n):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {3 + n + i} 0 R >>".encode()
        )
    for text in page_texts:
        safe = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref_pos))
    return bytes(out)


@pytest.fixture
def sample_pdf(tmp_path):
    """A 2-page PDF with known, extractable text."""
    path = tmp_path / "sample.pdf"
    path.write_bytes(build_pdf(["Hello page one about apples",
                                "Second page about bananas"]))
    return str(path)
