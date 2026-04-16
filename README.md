# Delta Meth — Diagnostic Shift Detection Prototype

Research prototype scaffold for a diagnostic shift detection pipeline inspired by the DiaShift paper.

Structure:

- `data/` — raw and processed data
- `src/` — pipeline modules (preprocessing, alignment, nli, classification, pipeline orchestration, utils)
- `configs/config.yaml` — model & threshold placeholders
- `requirements.txt` — minimal dependencies

This repository contains minimal, runnable placeholders and a simple example dataset so you can run the pipeline locally as a research prototype.

See `src/pipeline/run_pipeline.py` to run the example.

Dataset statistics (ICE notes corpus):

- total files: 287
- multi-segment files: 156 (54.36%)
- segments counted: 1,269
- sentences per segment — mean: 6.902, std: 5.195, min: 0, 25%: 4.0, 50%: 6.0, 75%: 9.0, max: 45

Raw and processed data are excluded from the repository by `.gitignore` to avoid uploading PHI or large files.
