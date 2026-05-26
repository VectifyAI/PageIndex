import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_metadata_generator_uses_provider_parameter():
    from pageindex.filesystem.metadata_generation import (
        MetadataGenerationError,
        MetadataGenerationInput,
        MetadataGenerator,
    )

    generator = MetadataGenerator(provider="unsupported", model="unused")
    request = MetadataGenerationInput(
        file_ref="file_a",
        external_id="doc_a",
        title="A",
        source_path="docs/a.txt",
        content_type="text/plain",
        source_type=None,
        text="hello",
    )

    with pytest.raises(MetadataGenerationError, match="unsupported metadata provider: unsupported"):
        generator.generate(request, fields=["summary"])
