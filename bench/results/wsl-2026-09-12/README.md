# Preliminary collection

This complete first collection is retained for provenance. A repository cleanup operation overlapped part of its measurement interval. Because wall time includes PNG export to the mounted Windows filesystem, that operation could influence complete-run latency. Its effect was not isolated.

No row or outlier has been deleted. The published comparison and figures use [the subsequent collection](../wsl-2026-09-12-isolated/), which ran without another benchmark or cleanup operation from the coordinating agents. The reason for repeating was identified before that second run, rather than selecting individual observations after comparing results.

The CSV, environment manifest, summary and individual logs below preserve the preliminary run. Regenerating its figures would replace the shared figure filenames, so restore the maintained figures with `python3 bench/plot.py bench/results/wsl-2026-09-12-isolated` afterwards.
