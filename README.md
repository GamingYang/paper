# Battery SOH Prediction Paper Project

This repository stores the manuscript, reproducible scripts, figures, and evidence packages for the current lithium-ion battery state-of-health (SOH) prediction paper.

Current manuscript focus:

> Degradation-Aware Physics-Guided Hybrid Transfer Learning for Cross-Batch State-of-Health Prediction of Lithium-Ion Batteries

The current paper line is SOH-only. Earlier RUL/EOL materials may remain in the repository history or older tracked folders, but they are not the active manuscript direction.

## Current Status

The latest uploaded SOH evidence package is organized around:

- A physics-guided residual modeling strategy.
- A degradation-state-guided curve attention mechanism.
- A lifecycle-adaptive MMD alignment term.
- A strict cross-batch protocol:
  - Batches 1-8: source training data.
  - Batch 9: target-domain adaptation and checkpoint/model selection.
  - Batch 10: final blind-test reporting only.

Batch 10 is not used for model selection.

## Main Manuscript Files

Primary manuscript:

- `codex/7.20初稿修改/zjc8.5 第四版 SOH预测小论文英文版.docx`

Markdown text draft:

- `codex/final-draft/zjc8.4_SOH_prediction_English_submission_draft.md`

Project status and writing notes:

- `codex/CURRENT_SOH_PAPER_STATUS_LOG_20260722.md`
- `codex/CURRENT_WINDOW_CONTEXT_AND_ATTENTION_ABLATION_GUARDRAIL_20260722.md`
- `codex/SOH_Q2_PAPER_OUTLINE_AND_TASKS_20260622.md`
- `codex/SOH_SUPPLEMENTARY_EXPERIMENTS_ACCEPTANCE_20260622.md`
- `codex/SOH_TARGET_ADAPTATION_CONTROLS_ACCEPTANCE_20260622.md`
- `codex/SOH_STATIC_MMD_AND_WEIGHT_FIGURE_ACCEPTANCE_20260623.md`
- `codex/SOH_MMD_STORYTELLING_LITERATURE_AUDIT_20260623.md`

## Core Evidence Packages

Main attention and final-control experiments:

- `codex/adaptive-attention-v1/`
- `codex/adaptive-attention-final-controls-v1/`
- `codex/advanced-attention-candidates-v1/`

SOH ablation and diagnostic experiments:

- `codex/soh-target-adaptation-controls-v1/`
- `codex/soh-stage-tail-bootstrap-v1/`
- `codex/soh-mmd-representation-diagnostic-v1/`
- `codex/soh-static-mmd-v1/`
- `codex/soh-mmd-exponential-weight-figure-v1/`
- `codex/soh-physical-degree-ablation-v1/`
- `codex/physical-base-coefficient-optimization-v1/`

Figure and table packages:

- `codex/batch10-soh-prediction-figure-v1/`
- `codex/dataset-overview-figure-v1/`
- `codex/dataset-processing-mechanism-figure-v1/`

## Representative Result Summary

Under the current three-seed protocol, the proposed degradation-state-guided curve attention improves the overall cross-batch SOH result compared with the retained ordinary-attention version.

| Model variant | Batch 10 RMSE mean | RMSE std | Macro RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| Ordinary attention retained model | 0.010381 | 0.000867 | 0.009171 | 0.006253 |
| Degradation-state-guided curve attention | 0.009906 | 0.000557 | 0.008764 | 0.006189 |

The wording in the paper should stay cautious: the adaptive attention improves the current cross-batch SOH protocol, but late-stage prediction remains an important limitation and should not be overstated.

## What Is Not Uploaded

The repository intentionally avoids committing large or locally generated materials such as:

- Raw battery datasets.
- Processed `.npy`, `.npz`, `.pkl`, or large intermediate data files.
- Model checkpoints such as `.pt`, `.pth`, and `.ckpt`.
- Full run logs, caches, and temporary files.
- Large rendered QA folders.
- External reproduced repositories.
- Bulk reference PDFs and literature-download folders.
- Word/PDF drafts that are not part of the current final manuscript package.

Examples of local-only folders that should usually not be committed:

- `runs/`
- `codex/translation_work/`
- `codex/_qa_zjc8_2_units/`
- `codex/external-advanced-method-reproduction-v1/`
- `paper_spine_materials/`
- `paper_rewriting_output/`
- `paper-plot-skills-main/`
- `Visio-copy-main/`
- `skills/`

Large files should be shared through cloud storage, institutional storage, or Git LFS if they must be versioned.

## Environment

The scripts mainly use:

- Python 3.x
- numpy
- pandas
- torch
- scikit-learn
- matplotlib
- scipy
- openpyxl
- python-docx

Most experiment scripts require the local processed battery data directory. The data are not included in this GitHub repository.

## Typical Commands

Check repository status:

```powershell
cd C:\Users\ROG\Desktop\paper
git status --short --branch
```

Run the project smoke test:

```powershell
python codex_battery_project_smoke_test.py
```

Run the main adaptive-attention experiment, after local data paths are prepared:

```powershell
python codex\adaptive-attention-v1\run_adaptive_attention_v1.py
```

## Git Workflow Notes

Because this project contains many local-only files, avoid broad staging commands:

```powershell
git add .
git add -A
```

Prefer explicit staging:

```powershell
git add README.md
git add "codex/adaptive-attention-v1"
```

Before committing, check what will actually be included:

```powershell
git status
git diff --cached --name-only
```

Then commit and push:

```powershell
git commit -m "Update README for final SOH paper"
git push
```

If `git status --short --branch` shows:

```text
## master...origin/master
```

and does not show `[ahead 1]`, the local branch and GitHub are synchronized.
