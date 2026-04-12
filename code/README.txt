PRISM v3 split layout
======================

- 00_header.txt: original shebang/docstring/header block.
- cell_01.py ... cell_15.py: original code split by CELL sections.
- cell_order.txt: execution order used by the runner.
- run_prism_split.py: executes all split cells in one shared namespace.

Output paths
------------
- All plot images are saved to ../images
- All non-image artifacts (tables/logs/csv/text) are saved to ../outputs
