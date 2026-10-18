"""Path confinement for tree_optimize --out / --log (issue #584).

``--out`` and ``--log`` used to be passed straight to ``open(path, "w")``, so a
typo or an untrusted invocation could silently overwrite any user-writable file
(e.g. ``--out /etc/cron.daily/x.json``). ``_confine_output_path`` resolves the
path and rejects anything outside the allowed directories (the CWD and the
input structure's directory).
"""
import os

import pytest

from pageindex.tree_optimize import _confine_output_path


def _real(p):
    return os.path.realpath(str(p))


def test_relative_path_resolves_under_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = _confine_output_path("out.json", flag="--out",
                               allowed_dirs=[str(tmp_path)])
    assert got == _real(tmp_path / "out.json")


def test_absolute_path_inside_allowed_dir(tmp_path):
    target = tmp_path / "sub" / "out.json"
    got = _confine_output_path(str(target), flag="--out",
                               allowed_dirs=[str(tmp_path)])
    assert got == _real(target)


def test_absolute_path_outside_allowed_dir_rejected(tmp_path):
    with pytest.raises(ValueError, match="refusing to write outside"):
        _confine_output_path("/etc/cron.daily/x.json", flag="--out",
                             allowed_dirs=[str(tmp_path)])


def test_dotdot_escape_rejected(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    with pytest.raises(ValueError, match="refusing to write outside"):
        _confine_output_path(str(sub / ".." / "evil.json"), flag="--log",
                             allowed_dirs=[str(sub)])


def test_symlink_escape_rejected(tmp_path):
    # A symlink inside an allowed dir that points outside must be rejected.
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = allowed / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="refusing to write outside"):
        _confine_output_path(str(link / "evil.json"), flag="--out",
                             allowed_dirs=[str(allowed)])


def test_second_allowed_dir_accepted(tmp_path):
    # The allowed set is the union of the CWD and the structure's directory, so
    # a path under the structure dir is fine even when the CWD differs.
    cwd = tmp_path / "cwd"
    struct = tmp_path / "struct"
    cwd.mkdir()
    struct.mkdir()
    target = struct / "tree.optimized.json"
    got = _confine_output_path(str(target), flag="--out",
                               allowed_dirs=[str(cwd), str(struct)])
    assert got == _real(target)


def test_default_out_path_derived_from_structure_allowed(tmp_path, monkeypatch):
    # main()'s default: --structure <dir>/tree.json -> <dir>/tree.optimized.json
    monkeypatch.chdir(tmp_path)
    struct = tmp_path / "data"
    struct.mkdir()
    target = struct / "tree.optimized.json"
    got = _confine_output_path(str(target), flag="--out",
                               allowed_dirs=[os.getcwd(), str(struct)])
    assert got == _real(target)
