# Build spatial.pdf from spatial.md. Run from paper/.
#
# Flag rationale (md-to-latex skill), so the knowledge is not tribal:
#  - Title/author/abstract live in the .md YAML front matter, so the body
#    headings start at `##`; `--shift-heading-level-by=-1` maps them to
#    \section (otherwise pandoc emits no \section and numbering degrades).
#  - `-F pandoc-crossref` resolves the {#sec:..}/{#eq:..}/{#fig:..} labels.
#  - `--citeproc` builds the reference list from spatial.bib using the vendored,
#    self-contained ieee.csl (no independent-parent): offline and reproducible.
#  - Two-column IEEEtran (`documentclass: IEEEtran`, `classoption: journal`).
#    pandoc emits `longtable` for every pipe table, which is illegal in
#    two-column mode, so the tables are authored as raw LaTeX `table*` floats
#    (the same convention the source MLE paper uses); their `\label`s are plain
#    LaTeX labels referenced with `Table~\ref{...}`.
#  - pdflatex cannot embed SVG; the *.svg sources live in figures/ and their
#    *.pdf twins are committed, so a normal build never needs the converter.
#    Run `make figures` only after editing an SVG.  Do not use ImageMagick
#    (it rasterizes); cairosvg keeps the figure vector.
#
# Requires: pandoc, pandoc-crossref, a LaTeX engine with pdflatex, and cairosvg
# (only to regenerate figures).

PANDOC := pandoc
FLAGS  := -F pandoc-crossref --citeproc -s -t latex -N --reference-links \
          --shift-heading-level-by=-1 --pdf-engine=pdflatex

FIG_SVG := $(wildcard figures/*.svg)
FIG_PDF := $(FIG_SVG:.svg=.pdf)

.PHONY: paper figures clean

paper: spatial.pdf

spatial.pdf: spatial.md spatial.bib ieee.csl $(FIG_PDF)
	$(PANDOC) $(FLAGS) spatial.md -o $@

figures: $(FIG_PDF)

figures/%.pdf: figures/%.svg
	python -c "import cairosvg,sys; cairosvg.svg2pdf(url=sys.argv[1], write_to=sys.argv[2])" "$<" "$@"

clean:
	-@rm -f spatial.tex spatial.aux spatial.log spatial.out spatial.toc ; true
