import logging
import PyPDF2
from pathlib import Path
from .protocol import ContentNode, ParsedDocument
from ..tokens import count_tokens

# Minimum image dimension to keep (skip icons/artifacts)
_MIN_IMAGE_SIZE = 32

_warned_no_pymupdf = False


class PdfParser:
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        path = Path(file_path)
        model = kwargs.get("model")
        images_dir = kwargs.get("images_dir")

        # Images are extracted with PyMuPDF (optional); text stays with PyPDF2
        # below so the extracted text — and therefore the tree — matches the
        # CLI / pre-SDK default.
        page_images = self._extract_images(path, images_dir) if images_dir else {}

        reader = PyPDF2.PdfReader(str(path))
        nodes = []
        for i, page in enumerate(reader.pages):
            page_num = i + 1
            content = page.extract_text() or ""
            images = page_images.get(page_num)
            if images:
                refs = "\n".join(f"![image]({img['path']})" for img in images)
                content = f"{content}\n{refs}" if content else refs

            tokens = count_tokens(content, model=model)
            nodes.append(ContentNode(
                content=content,
                tokens=tokens,
                index=page_num,
                images=images,
            ))

        return ParsedDocument(doc_name=path.name, nodes=nodes,
                              metadata={"page_count": len(reader.pages)})

    @staticmethod
    def _extract_images(path: Path, images_dir: str) -> dict[int, list[dict]]:
        """Extract images per page. Requires the optional PyMuPDF dependency;
        without it, indexing proceeds text-only."""
        global _warned_no_pymupdf
        try:
            import pymupdf
        except ImportError:
            if not _warned_no_pymupdf:
                logging.getLogger(__name__).warning(
                    "PyMuPDF is not installed; skipping image extraction. "
                    "Install with: pip install pymupdf")
                _warned_no_pymupdf = True
            return {}

        page_images: dict[int, list[dict]] = {}
        with pymupdf.open(str(path)) as doc:
            for i, page in enumerate(doc):
                images = PdfParser._extract_page_images(page, i + 1, images_dir)
                if images:
                    page_images[i + 1] = images
        return page_images

    @staticmethod
    def _extract_page_images(page, page_num: int, images_dir: str) -> list[dict]:
        """Save a page's images to disk and return their metadata."""
        import pymupdf
        images_path = Path(images_dir)
        images_path.mkdir(parents=True, exist_ok=True)
        # Store an absolute path so the ![image](...) reference resolves
        # regardless of the process's cwd at query time. (cwd-relative paths
        # break as soon as the query runs from a different directory.)
        abs_images_path = images_path.resolve()

        images: list[dict] = []
        img_idx = 0

        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 1:  # image blocks only
                continue
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
                pix.save(str(images_path / filename))
                pix = None
            except Exception:
                continue

            images.append({
                "path": str(abs_images_path / filename),
                "width": width,
                "height": height,
            })
            img_idx += 1

        return images
