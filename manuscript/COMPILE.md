# Compiling the manuscript

The manuscript uses the official IEEE Access class (`ieeeaccess.cls`, which depends on
`IEEEtran.cls`). Everything needed to compile is in this folder, and a pre-built,
verified `frontier.pdf` is included.

## Overleaf (recommended)
1. Upload all files in this `manuscript/` folder.
2. Set the main document to `frontier.tex`, compiler to **pdfLaTeX**.
3. Compile twice (bibliography is embedded with `thebibliography`; no BibTeX pass needed).

## Local
Standard TeX Live `pdflatex` (run twice), or Tectonic: `tectonic frontier.tex`.

## Files
- `frontier.tex` : manuscript source (no em dashes; 26 references; 4 figures; 3 tables).
- `frontier.pdf` : pre-built PDF (verified compile).
- `ieeeaccess.cls`, `IEEEtran.cls` : IEEE Access classes.
- `fig_accuracy.png`, `fig_frontier.png`, `fig_complexity.png`, `fig_memory.png` : figures.
- Class assets required by ieeeaccess.cls: `logo.png`, `Logo.png`, `notaglinelogo.png`,
  `notaglineLogo.png`, `bullet.png` (both cases of the logo names are provided because the
  class references `Logo.png`/`notaglineLogo.png` while the template ships lowercase names).

## Maintainer note
The preamble defines `\providecommand{\xfigwd}{0pt}`. The IEEE Access class only sets
`\xfigwd` inside its custom `\Figure` command; this default lets a standard `figure`
environment work with the class caption macro on any engine.

## Before submission (author tasks)
- Replace the placeholder corresponding-author email with A. W. Malik's real UCP address.
- Add both authors' ORCID iDs.
- Rewrite Abstract/Introduction/Discussion/Conclusion in your own words; consider trimming
  the abstract toward 200 words.
- Fill the journal-supplied fields (`\history`, `\doi`).
