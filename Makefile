.PHONY: all analyze figures verify paper clean

all: analyze figures verify

analyze:
	python3 src/analyze.py

figures:
	python3 src/figures.py

verify:
	python3 src/verify.py

paper:
	cd paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc
