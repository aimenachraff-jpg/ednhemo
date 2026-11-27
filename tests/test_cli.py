"""CLI smoke tests."""

import pytest

from ednhemo import __version__
from ednhemo.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_list_models(capsys):
    assert main(["--list-models"]) == 0
    out = capsys.readouterr().out
    assert "socle" in out
    assert "variante" in out
    assert "44.1" in out


def test_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "usage: ednhemo" in capsys.readouterr().out


def test_missing_file_returns_error_code(capsys, tmp_path):
    assert main([str(tmp_path / "nope.nii.gz")]) == 2
    assert "error" in capsys.readouterr().err.lower()
