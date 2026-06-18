"""Tiny helper to assemble Jupyter notebooks from (kind, source) cell tuples.

Keeps the per-question builder scripts readable: they just list cells as
``md(...)`` / ``code(...)`` and call :func:`write_notebook`.
"""
from __future__ import annotations

import nbformat as nbf


def md(source: str) -> tuple[str, str]:
    """Markdown cell."""
    return ("md", source.strip("\n"))


def code(source: str) -> tuple[str, str]:
    """Code cell."""
    return ("code", source.strip("\n"))


def write_notebook(path: str, cells: list[tuple[str, str]]) -> None:
    """Write a notebook to ``path`` from a list of ``md``/``code`` cell tuples."""
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
