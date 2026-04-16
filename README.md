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

---

Segmentation Evaluation Results

Found 574 raw notes and 287 processed JSON files.

--- Segmentation Evaluation Results ---
total_evaluated: 287
single_segment: 131
multi_segment: 156
multi_segment_correct_start: 137
missing_raw_file: 0
text_length_mismatch: 1
empty_segments: 5

Multi-segment notes correctly starting with pre-ICU: 87.8%

--- Found 5 empty segments ---
Note ID: 25363214 -> Segment Index: 1, Date: '2024-06-03'
Note ID: 21363910 -> Segment Index: 29, Date: '2024-05-13'
Note ID: 25426210 -> Segment Index: 4, Date: '2024-09-09'
Note ID: 25477377 -> Segment Index: 12, Date: '26/11'
Note ID: 25477377 -> Segment Index: 13, Date: '27/11'

--- Investigating 19 multi-segment notes without pre-ICU start ---
Note ID: 23458098 -> Starts with Date: '22/11'
Note ID: 23458098Γò¼ΓûÆ -> Starts with Date: '25/10-21'
Note ID: 25332587 -> Starts with Date: '2024-04-21'
Note ID: 24954092 -> Starts with Date: '02-17/07'
Note ID: 20097899 -> Starts with Date: '11/03'
Note ID: 25349980 -> Starts with Date: '2024-05-22'
Note ID: 20196769 -> Starts with Date: '2024-02-28'
Note ID: 24855915 -> Starts with Date: '2024-08-19'
Note ID: 25469074 -> Starts with Date: '2024-11-01'
Note ID: 25400962 -> Starts with Date: '2024-08-19'
Note ID: 25056040 -> Starts with Date: '2024-09-05'
Note ID: 25375514Γò¼ΓûÆ -> Starts with Date: '21/07-24/07'
Note ID: 20090861 -> Starts with Date: '2024-01-06'
Note ID: 25383759 -> Starts with Date: '21/07-24'
Note ID: 25426210 -> Starts with Date: '02-06/09'
... and 4 more.

