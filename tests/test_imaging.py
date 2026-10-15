"""highlight_region self-check."""
import pytest
from PIL import Image
from pageindex.imaging import highlight_region


def test_highlight_region_draws_on_image():
    img = Image.new("RGB", (500, 2000), "white")
    result = highlight_region(img, [100, 200, 300, 400])
    assert isinstance(result, Image.Image)
    assert result.size == (500, 2000)
    assert result.getpixel((75, 700)) != (255, 255, 255)
    assert result.getpixel((300, 300)) == (255, 255, 255)


def test_highlight_region_accepts_bytes():
    from io import BytesIO
    img = Image.new("RGB", (500, 500), "white")
    buf = BytesIO()
    img.save(buf, "PNG")
    result = highlight_region(buf.getvalue(), [0, 0, 500, 500], scale=500)
    assert isinstance(result, Image.Image)


def test_highlight_region_invalid_bbox():
    img = Image.new("RGB", (100, 100), "white")
    with pytest.raises(ValueError, match="Invalid bbox"):
        highlight_region(img, [300, 200, 100, 400])
    with pytest.raises(ValueError, match="Invalid bbox"):
        highlight_region(img, [0, 0, 1001, 500])
    with pytest.raises(ValueError, match="Invalid bbox"):
        highlight_region(img, [0, 0, 500, 500], scale=0)
