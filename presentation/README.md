# QAI 2026 talk

Beamer slides for the ME-DQAS talk at QAI 2026 (Nottingham, December 2026).
The look comes from the inofficial Lamarr slide template (`lamarrtheme.sty`, `img/`, `fonts/` WorkSans, OFL).
With `lualatex` (needs `luaotfload`) the slides use WorkSans; with `pdflatex` they fall back to Helvetica.

```bash
cd presentation
pdflatex main.tex && pdflatex main.tex
```

Put figures in `figures/`; the paper figures under `../dataset/` are found too.
