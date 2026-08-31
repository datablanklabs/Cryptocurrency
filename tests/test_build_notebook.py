"""Golden check on the notebook generator: the cell list must validate as
nbformat and every generated code cell must at least parse."""

from __future__ import annotations

import nbformat

import build_notebook


def test_cells_validate_as_nbformat():
    assert len(build_notebook.CELLS) > 20
    nb = nbformat.v4.new_notebook(cells=build_notebook.CELLS)
    nbformat.validate(nb)          # raises on a malformed cell


def test_every_code_cell_compiles():
    for i, cell in enumerate(build_notebook.CELLS):
        if cell.get("cell_type") == "code":
            compile(cell["source"], f"<generated cell {i}>", "exec")
