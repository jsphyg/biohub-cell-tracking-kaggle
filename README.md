# Biohub Cell Tracking During Development

Kaggle project for the [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) competition.

The repository is intentionally Kaggle-first: the competition data is too large for our local machine, so notebooks are designed to run in Kaggle's hosted environment. Local files contain code, notes, and experiment metadata only.

## Current baseline

`notebooks/baseline_pipeline.ipynb` implements a first end-to-end submission:

1. Open each test OME-Zarr store lazily.
2. Read one 3D frame at a time.
3. Detect bright cell-like blobs with 3D Difference-of-Gaussians.
4. Link detections between adjacent frames using greedy one-to-one nearest-neighbor matching in physical coordinates.
5. Write the required combined node/edge `submission.csv`.

This is a feasibility baseline, not a competitive model. It gives us a working submission, runtime estimate, output-schema check, and a starting point for local validation.

## Kaggle setup

1. Join the competition and create a new Kaggle notebook.
2. Upload or copy the contents of `notebooks/baseline_pipeline.ipynb`.
3. Attach the competition data through **Add Input → Competition**.
4. Run the inspection cells first. They confirm the input path, Zarr layout, and array shape before processing the full test set.
5. Run all cells and submit the generated `submission.csv`.

The notebook includes an optional online `zarr` install for exploration. For a final Kaggle submission with internet disabled, attach an offline wheel dataset and update the install cell as described in the notebook.

## Planned experiments

- Tune DoG scale and threshold on labeled training embryos.
- Add frame-to-frame motion gating and Hungarian assignment.
- Add one-frame gap closing.
- Detect divisions explicitly and measure Edge Jaccard versus Division Jaccard.
- Compare classical detection with a small 3D detector trained on centroid labels.

## Repository conventions

Do not commit competition data, Kaggle credentials, generated submissions, or notebook outputs containing large artifacts. Keep one experiment per branch or clearly named notebook version, and record public scores in `RESULTS.md` once we have them.

