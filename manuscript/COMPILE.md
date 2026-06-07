# Compiling the manuscript

The manuscript uses the official IEEE Access class (`ieeeaccess.cls`, which depends on
`IEEEtran.cls`). Everything needed to compile is in this folder.

## Overleaf (recommended)
1. Create a new project and upload all files in this `manuscript/` folder
   (`frontier.tex`, `ieeeaccess.cls`, `IEEEtran.cls`, `logo.png`, `notaglinelogo.png`,
   and the four `fig_*.png` files).
2. Set the main document to `frontier.tex` and the compiler to **pdfLaTeX**.
3. Compile. The bibliography is embedded with `thebibliography`, so no separate BibTeX
   pass is needed.

## Local
```
pdflatex frontier.tex
pdflatex frontier.tex   # second pass resolves cross-references and citations
```

## Files
- `frontier.tex` : the manuscript source (no em dashes; 26 references; 4 figures; 3 tables).
- `ieeeaccess.cls`, `IEEEtran.cls` : IEEE Access document classes.
- `logo.png`, `notaglinelogo.png` : required by the class.
- `fig_accuracy.png`, `fig_frontier.png`, `fig_complexity.png`, `fig_memory.png` : figures.

## Before submission (author tasks)
- Rewrite the Abstract, Introduction, Discussion, and Conclusion in your own words for the
  originality check; consider trimming the abstract toward 200 words.
- Fill the journal-supplied fields (`\history`, `\doi`).
