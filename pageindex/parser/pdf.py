import pymupdf
from pathlib import Path
from .protocol import ContentNode, ParsedDocument
from ..tokens import count_tokens

# Minimum image dimension to keep (skip icons/artifacts)
_MIN_IMAGE_SIZE = 32


class PdfParser:
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        path = Path(file_path)
        model = kwargs.get("model")
        images_dir = kwargs.get("images_dir")
        nodes = []

        with pymupdf.open(str(path)) as doc:
            for i, page in enumerate(doc):
                page_num = i + 1
                if images_dir:
                    content, images = self._extract_page_with_images(
                        doc, page, page_num, images_dir)
                else:
                    content = page.get_text()
                    images = None

                tokens = count_tokens(content, model=model)
                nodes.append(ContentNode(
                    content=content or "",
                    tokens=tokens,
                    index=page_num,
                    images=images if images else None,
                ))

        return ParsedDocument(doc_name=path.stem, nodes=nodes)

    @staticmethod
    def _extract_page_with_images(doc, page, page_num: int,
                                  images_dir: str) -> tuple[str, list[dict]]:
        """Extract text and images from a page, preserving their relative order.

        Uses get_text("dict") to iterate blocks in reading order.
        Text blocks become text; image blocks are saved to disk and replaced
        with an inline placeholder: ![image](path)
        """
        images_path = Path(images_dir)
        images_path.mkdir(parents=True, exist_ok=True)
        # Store an absolute path so the ![image](...) reference resolves
        # regardless of the process's cwd at query time. (cwd-relative paths
        # break as soon as the query runs from a different directory.)
        abs_images_path = images_path.resolve()

        parts: list[str] = []
        images: list[dict] = []
        img_idx = 0

        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:  # text block
                lines = []
                for line in block["lines"]:
                    spans_text = "".join(span["text"] for span in line["spans"])
                    lines.append(spans_text)
                parts.append("\n".join(lines))

            elif block["type"] == 1:  # image block
                width = block.get("width", 0)
                height = block.get("height", 0)
                if width < _MIN_IMAGE_SIZE or height < _MIN_IMAGE_SIZE:
                    continue

                image_bytes = block.get("image")
                if not image_bytes:
                    continue

                try:
                    pix = pymupdf.Pixmap(image_bytes)
                    # n includes the alpha channel, so a plain RGBA pixmap also
                    # has n==4 — subtract alpha before comparing. Without this,
                    # a CMYK image with no alpha (n==4, same as RGBA) skips the
                    # RGB conversion, and pix.save() as .png then raises
                    # "unsupported colorspace for 'png'", silently dropping the
                    # image via the bare except below.
                    if pix.n - pix.alpha >= 4:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    filename = f"p{page_num}_img{img_idx}.png"
                    save_path = images_path / filename
                    pix.save(str(save_path))
                    pix = None
                except Exception:
                    continue

                img_path = str(abs_images_path / filename)
                images.append({
                    "path": img_path,
                    "width": width,
                    "height": height,
                })
                parts.append(f"![image]({img_path})")
                img_idx += 1

        content = "\n".join(parts)
        return content, images
