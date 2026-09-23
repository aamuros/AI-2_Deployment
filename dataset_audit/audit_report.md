# FDM Defect Segmentation Dataset Audit

## Audit status

**FIX BEFORE TRAINING — DATASET NOT AVAILABLE FOR INSPECTION**

This status does not mean that defects were found in the dataset itself. It means the requested audit could not be completed because no dataset was present in the inspected workspace.

## Inspection scope

- Workspace: `/Users/aamuros/Documents/AI2-FinalProject/fdm-defect-segmentation`
- Mode: inspection-only
- Dataset files modified: none
- Search result: no `data.yaml` or `data.yml` and no train, validation, or test image/label folders were found
- Existing data-related folders: `model/` and `sample_images/`, both empty at inspection time

## 1. Dataset structure

Image counts, label counts, `data.yaml` contents, and configured class IDs could not be verified because the dataset is absent.

Expected classes supplied by the project brief:

| Class ID | Class name |
|---:|---|
| 0 | Cracking |
| 1 | Layer_Shifting |
| 2 | Stringing |
| 3 | Warping |

These expected values could not be compared with an actual `data.yaml` file.

## 2. Class distribution

Class distributions could not be calculated. Missing-class checks could not be performed.

## 3. Annotation integrity

Missing labels, orphan labels, empty labels, malformed rows, invalid class IDs, out-of-range coordinates, multiclass files, and duplicate annotation rows could not be checked because no labels were available.

## 4. Image integrity

Image readability and SHA-256 duplicate checks could not be performed because no dataset images were available.

## 5. Source-group leakage

Exact cross-split duplication and likely filename-based source-group leakage could not be checked because no split files were available.

## 6. Split suitability

The claimed total of 1,400 original images and the requested 70/20/10 split could not be verified. If the dataset is balanced at 350 original images per class, the stated target is mathematically consistent:

| Split | Per class | Four-class total | Percentage |
|---|---:|---:|---:|
| Train | 245 | 980 | 70% |
| Validation | 70 | 280 | 20% |
| Test | 35 | 140 | 10% |
| Total | 350 | 1,400 | 100% |

Whether the actual dataset supports this split remains unknown. The files must be made available before training so counts, annotations, image integrity, exact duplicates, and related-source leakage can be verified.

## Required next step

Place or copy the dataset into the accessible project workspace, including `data.yaml` and the train/validation/test image and label directories, then rerun the audit. Do not train based on this incomplete audit.

## Output note

The accompanying CSV files contain headers only because emitting invented dataset rows or zero counts would be misleading.
