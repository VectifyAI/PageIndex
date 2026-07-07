"""Layering / protocol-contract guards for the SDK."""
import ast
from pathlib import Path

import pytest

from pageindex.backend.protocol import Backend, SupportsParserRegistration
from pageindex.backend.cloud import CloudBackend


def test_parser_layer_does_not_import_index():
    """parser/* must not depend on the index package (reverse dependency)."""
    parser_dir = Path(__file__).parent.parent / "pageindex" / "parser"
    offenders = []
    for py in parser_dir.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "index" in node.module.split("."):
                offenders.append(f"{py.name}: from {node.module}")
    assert not offenders, f"parser imports index: {offenders}"


def test_count_tokens_lives_in_leaf_module():
    from pageindex.tokens import count_tokens
    from pageindex.index.utils import count_tokens as reexport
    from pageindex.parser.pdf import count_tokens as parser_ct
    # index re-exports the leaf function; parser imports the same leaf.
    assert reexport is count_tokens is parser_ct


def test_parser_registration_is_a_capability_protocol():
    from unittest.mock import MagicMock
    from pageindex.backend.local import LocalBackend
    lb = LocalBackend(storage=MagicMock(), files_dir="/tmp/x", model="m")
    assert isinstance(lb, SupportsParserRegistration)
    assert not isinstance(CloudBackend(api_key="pi-test"), SupportsParserRegistration)


def test_register_parser_rejected_in_cloud_mode():
    from pageindex import CloudClient
    from pageindex.errors import PageIndexError
    client = CloudClient(api_key="pi-test")
    with pytest.raises(PageIndexError, match="not supported in cloud mode"):
        client.register_parser(object())


def test_both_backends_satisfy_backend_protocol():
    from unittest.mock import MagicMock
    from pageindex.backend.local import LocalBackend
    assert isinstance(CloudBackend(api_key="pi-test"), Backend)
    assert isinstance(LocalBackend(storage=MagicMock(), files_dir="/tmp/x", model="m"), Backend)


def test_typed_dicts_are_exported():
    import pageindex.types as t
    assert {"DocumentInfo", "DocumentDetail", "PageContent"} <= set(dir(t))
