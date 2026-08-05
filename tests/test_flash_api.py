from io import BytesIO

import pymupdf

from pageindex.flash import page_index_flash


def _sparse_bookmarked_pdf():
    doc = pymupdf.open()
    for marker in ("x", "y", "z"):
        page = doc.new_page()
        page.insert_text((72, 72), marker)
    doc.set_toc([
        [1, "Introduction", 1],
        [1, "Methods", 2],
        [1, "Results", 3],
    ])
    data = doc.tobytes()
    doc.close()
    return BytesIO(data)


def test_sparse_bookmarked_pdf_can_use_default_summaries():
    result = page_index_flash(_sparse_bookmarked_pdf(), summary=True)

    assert result["toc_source"] == "hybrid"
    assert [node["title"] for node in result["structure"]] == [
        "Introduction",
        "Methods",
        "Results",
    ]
    assert all(node.get("summary") for node in result["structure"])
    assert "page_texts" not in result
