# Historical coursework reports

This directory preserves the original TeX report variants, presentation notes and figure-generation scripts. They are historical project artifacts; the maintained technical report is the [root README](../README.md), with [current design](../docs/design.md) and [measured results](../docs/results.md).

The retained reports and Python chart scripts contain older performance values for RTX 4060, RTX 4070 and GH200 hardware. The chart scripts embed those values as constants and do not load committed raw measurements. Those values are unverified historical claims and must not be cited as results of the current branch. Claims about feature coverage and test counts can also be obsolete.

The TeX files are distinct variants, not byte-identical duplicates. They are retained to preserve the coursework context. Generated PDFs, animations and legacy charts are ignored and are not part of the current evidence. Regenerate the maintained figures only through `bench/plot.py`, which reads the committed raw CSV.
