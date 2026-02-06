# Makefile for RRMC Results Slides
# Uses Metropolis (mtheme) beamer theme

MAIN = main
LATEX = pdflatex
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# Path to mtheme source
MTHEME_SRC = ../mtheme/source

.PHONY: all clean install-theme slides

all: install-theme slides

# Install mtheme .sty files locally
install-theme:
	@echo "Installing Metropolis theme..."
	@cd $(MTHEME_SRC) && latex beamerthememetropolis.ins > /dev/null 2>&1 || true
	@cp -f $(MTHEME_SRC)/*.sty . 2>/dev/null || true
	@echo "Theme installed."

# Compile slides
slides: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Quick compile (single pass)
quick:
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz

# Deep clean (including PDF)
distclean: clean
	rm -f $(MAIN).pdf *.sty

# View PDF
view: $(MAIN).pdf
	@xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || echo "Please open $(MAIN).pdf manually"

help:
	@echo "Available targets:"
	@echo "  all          - Install theme and compile slides (default)"
	@echo "  slides       - Compile slides only"
	@echo "  quick        - Single-pass compile (faster, may have wrong refs)"
	@echo "  install-theme - Extract and install mtheme .sty files"
	@echo "  clean        - Remove auxiliary files"
	@echo "  distclean    - Remove all generated files including PDF"
	@echo "  view         - Open the PDF"
