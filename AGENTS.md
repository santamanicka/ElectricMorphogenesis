# Repository Guidelines

## Project Structure & Module Organization
- Core simulation logic lives in `cellularFieldNetwork.py`, `geneRegulatoryNetwork.py`, and `simulateCellularFieldNetwork.py`; `simulateTrainedModel.py` toggles between Mosaic and Stigmergic models via the `Model` variable.
- Embryo-specific behavior is in `embryo.py` and `embryoNetwork.py`; visualization and utilities live in `visualize.py` and `utilities.py`.
- Parameter files (e.g., `data/MosaicModelParameters.dat`, `data/StigmergicModelParameters.dat`) seed runs and sweeps; retain new artifacts under `data/` to keep experiments reproducible.
- Batch workflows use the provided shell wrappers (`runSimulateEmbryoNetwork.sh`, `runLearnCellularFieldNetwork.sh`, `runPlotAnalysisData.sh`) and analysis scripts (`analyze*`, `compute*`, `summarizeSearchResults.py`).

## Build, Test, and Development Commands
- Environment: Python 3 with PyTorch, NumPy, SciPy, and Matplotlib. GPU usage is optional but supported by PyTorch if installed.
- Quick run for a trained model: `python simulateTrainedModel.py` (set `Model` to `"Stigmergic"` or `"Mosaic"` inside the file).
- Custom configuration: `python simulateCellularFieldNetwork.py` after editing parameters in the script.
- End-to-end embryo pipeline: `bash runSimulateEmbryoNetwork.sh`; learning pipeline: `bash runLearnCellularFieldNetwork.sh`.
- Plotting existing sweep outputs: `bash runPlotAnalysisData.sh` or call `python plotAnalysisData.py` with the relevant file paths.

## Coding Style & Naming Conventions
- Follow Pythonic/PEP8 defaults: 4-space indentation, snake_case for functions/variables, and clear docstrings for new public methods. Preserve existing class names even when lowerCamel (e.g., `cellularFieldNetwork`) to avoid API drift.
- Keep configuration near the top of scripts (e.g., `Model` in `simulateTrainedModel.py`; sweep parameters in `compute*` scripts). Prefer adding helper functions in `utilities.py` rather than duplicating logic.
- When adding new scripts, mirror the naming patterns here (`simulate*.py`, `analyze*.py`, `run*.sh`) for discoverability.

## Testing Guidelines
- Smoke tests: `python testFacialGRN.py` (generates `facial_grn_visualization.png`) and `python testFacialGRN_compatibility.py` (prints interface checks).
- For simulation changes, run the relevant driver (e.g., `python simulateTrainedModel.py`) and confirm saved plots/metrics match expectations; include notable outputs in your PR description.
- Keep random seeds explicit when you add stochastic steps; log parameter choices to ease reruns.

## Model Flow: Stigmergic ↔ FacialGRN
- Stigmergic run produces a voltage-based face pattern (`run_stigmergic_facial_integration.py`); `FacePatternCoordinator` converts that snapshot into a feature mask/set-point.
- FacialGRN can be pre-seeded with that set-point to align morphogens/genes to the electric mask.
- Optional bidirectional loop: GRN evolves features → feeds back to the electric model via `apply_gene_voltage_feedback` → new set-point derived → GRN re-seeded; repeat for a few cycles.
- Diagnostic output: `stigmergic_facial_integration.png` shows Vmem, derived mask, GRN features, and Pax6 expression.

## Commit & Pull Request Guidelines
- Commit messages follow the existing style: concise, capitalized summaries (e.g., `Add stability check for field gating`); keep subject lines under ~72 characters.
- In PRs, include: a short intent summary, commands run, and pointers to generated artifacts (plots, `.dat` outputs). Link issues or TODOs where applicable and call out any backward-incompatible parameter changes.
