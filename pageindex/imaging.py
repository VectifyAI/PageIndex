"""Image helpers for citation highlighting."""
from __future__ import annotations

from io import BytesIO
from typing import Union

from PIL import Image as PILImage, ImageDraw


def highlight_region(
    image: Union[PILImage.Image, bytes],
    bbox: list,
    scale: int = 1000,
) -> PILImage.Image:
    """Draw a translucent highlight over a bounding-box region.

    Args:
        image: A PIL Image or raw image bytes (JPEG/PNG).
        bbox: ``[x0, y0, x1, y1]`` in units of ``scale`` from the
            top-left corner, as ``get_block()`` returns it.
        scale: The coordinate space ``bbox`` lives in (default 1000,
            matching the PageIndex block coordinate system).

    Returns:
        A new PIL Image with a yellow highlight and orange outline
        over the region.
    """
    if isinstance(image, bytes):
        image = PILImage.open(BytesIO(image))
    x0, y0, x1, y1 = map(float, bbox)
    if scale <= 0 or not (0 <= x0 < x1 <= scale and 0 <= y0 < y1 <= scale):
        raise ValueError(f"Invalid bbox for scale {scale}: {bbox}")
    page = image.convert("RGBA")
    width, height = page.size
    rect = (x0 / scale * width, y0 / scale * height,
            x1 / scale * width, y1 / scale * height)
    overlay = PILImage.new("RGBA", page.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(
        rect, fill=(255, 210, 0, 65), outline=(255, 140, 0, 255),
        width=max(2, round(width / 400)),
    )
    return PILImage.alpha_composite(page, overlay).convert("RGB")
