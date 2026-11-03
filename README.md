# KION watch feedback and BERT4Rec

Reproducible comparison of five ways to use film watch percentage as implicit feedback in a fixed compact BERT4Rec. [Инструкция на русском](README_RU.md).

The accompanying Russian manuscript is in `paper/manuscript_ru.md` and `.docx`. This repository is a prepared research artifact; a public GitHub URL and final author/publication details have not been assigned.

## What is compared

| Method | History | Masked-target weight, r = watch percentage / 100 |
|---|---|---|
| ALL | Kept | 1 |
| DELETE10 | Remove percentage ≤10 before truncation, including at inference | 1 |
| TARGET10 | Kept | 0 if r≤0.1; otherwise 1 |
| STEP10 | Kept | 1 if r≤0.1; otherwise 3 |
| SOFT | Kept | 0.2 + 0.8r |

The loss is `sum(weight * cross_entropy) / sum(weight)`. Low-percentage interactions are not labelled dislikes. STEP10's weight mapping follows the [RecTools guide](https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_advanced_training_guide.html); we compare that mapping in our own shared training implementation, not RecTools' full pipeline.

One experiment: 5 methods × 2 chronological cuts × seeds 17, 29, 43 = 30 models. Each cut independently samples 50,000 training-eligible users (at least 10 distinct films, sampling seed 20261031), before method-specific filtering. Maximum training window 20; inference ≤19 known items + MASK; mask probability 0.35; 20,000 updates; validation every 1,000 updates. Configuration is in `config/scientific.json`; `kion_watch_feedback.py` is the unchanged implementation used for the reported run.

## Run on Kaggle

Import `notebooks/KION_WatchFeedback_5methods_Kaggle.ipynb`, enable Internet and a GPU, and Run All. It embeds the full frozen implementation; no repository upload or Google Drive is required. See `docs/KAGGLE_RU.md` for checkpoints and continuation.

## Run from a command line

Use a fresh Python environment. `requirements.txt` is a portable dependency range. For the recorded numerical stack see `requirements-paper-cu128.txt` and `reference_results/environment.json` (Python 3.13.15, torch 2.11.0+cu128, NumPy 2.1.3, pandas 2.3.3, Tesla T4, FP32, two CPU threads). The exact plotting version and full transitive environment were not recorded. Changed runtime/hardware can change numerical results and timing; a fresh series records its actual environment and receives its own plan hash. Continuation requires the same frozen environment.

```bash
python -m pip install -r requirements.txt
python reproduce.py --output outputs/full
```

Scientific training requires CUDA. The official, pinned KION archive is downloaded automatically and verified by SHA-256. An existing copy can be supplied:

```bash
python reproduce.py --output outputs/full --data-archive /path/to/data_en.zip
python reproduce.py --output outputs/full --max-runs 5
python reproduce.py --output outputs/continued --resume-from /path/to/saved/full
```

`--max-runs` limits new models in an invocation, not the 30-model design. Test evaluation waits until the entire queue is finished. Existing complete runs are checked and reused. Save the full output folder with `runs/*/last.pt` and `best.pt` to continue; the light results ZIP deliberately excludes weights. Checkpoints are saved every 1,000 updates; after interruption up to 1,000 unsaved updates can repeat.

CPU technical check (80 users/cut, two updates/model, not paper results):

```bash
python -m unittest discover -s tests -v
python reproduce.py --smoke --output outputs/cpu-smoke
```

The smoke option still uses the official archive. Unit tests use synthetic data and do not download it. Preparation without training: `python reproduce.py --prepare-only --output outputs/prepared` (scientific mode still checks CUDA). No full GPU run is performed by CI.

## Rebuild figures and independently check results

No training is needed for the supplied aggregate results:

```bash
python analysis/plot_results.py --results-dir reference_results --output-dir outputs/figures
python analysis/check_release.py
```

To rebuild the editable manuscript as well, install `python-docx>=1.1,<2` and run `python analysis/build_manuscript.py`. Rebuilding changes release-file hashes; `check_release.py` verifies the distributed snapshot, not arbitrary local edits. Render and visually review the DOCX again after rebuilding it.

For a new complete scientific run, use its light results ZIP and the official source data. This independent verifier rebuilds cohorts, windows and future targets, recomputes NDCG/Recall from ranked IDs, and recomputes all 32 bootstrap intervals. It does not import or execute training code from the ZIP and does not inspect checkpoint contents.

```bash
python analysis/verify_results.py --archive outputs/full_results.zip --data-archive /path/to/data_en.zip --output-dir outputs/audit
python analysis/plot_results.py --results-dir outputs/audit --output-dir outputs/verified-figures
```

The verifier targets the scientific 50k/S20/20k protocol and its report schema; it is not a verifier for arbitrary hyperparameter studies or the CPU smoke run. For new raw results, the plot script also accepts the run directory's `results_by_method.csv` and `results_per_run.csv` directly.

## Evaluation and interpretation

| Cut | Train through | Validation | Test |
|---|---|---|---|
| main | 2021-08-01 | 2021-08-02…08-08 | 2021-08-09…08-15 |
| replication | 2021-07-18 | 2021-07-19…07-25 | 2021-07-26…08-01 |

Targets are the known-catalog films of the first raw future film day, chosen before removing OOV or missing-percentage records; the target day is never shifted. Test contexts use observations through the validation endpoint; model weights are not refit. All previously seen films are excluded even under DELETE10. Same-day ordering in the input uses item ID, not an observed within-day order. The catalog is based on training users before any method-specific filtering.

Primary outcome: binary NDCG@10 for future films with recorded percentage ≥50. Also report Recall@10 and the separate, larger population of any valid future film. All models use the first checkpoint attaining maximum validation high-percentage NDCG. Four prespecified contrasts: SOFT−ALL and SOFT−STEP10 on each cut. Pair users, average their metrics over three seeds first, then 5,000 bootstrap resamples (seed 20261031). Primary intervals are 98.75% (nominal Bonferroni correction for four contrasts); other intervals are descriptive 95%. These intervals condition on the trained models; seed SD is separate. The temporal cuts are correlated, not independent datasets.

SOFT vs ALL high-percentage NDCG: +4.43% (main), +4.46% (replication). SOFT is not a universal winner: DELETE10 has the largest replication mean and the SOFT−STEP10 replication interval crosses zero. See the manuscript and `reference_results/` for complete comparisons. The aggregate results contain no individual user IDs. Model weights and raw KION are not bundled; the original light archive was numerically audited, not used to verify weight contents.

## Exploratory viewer and film analysis

The additional analysis compares SOFT and DELETE10 in fixed groups of past history length, recent weak-view share and context shortening. Film groups use training frequency only. It was specified after the aggregate results were known and is exploratory: all new 95% user intervals are descriptive and have no multiple-comparison correction. The original four primary comparisons remain unchanged. No new model training or test-based routing policy is claimed.

The analysis localizes the mean difference mainly in popular films and does not establish a transferable policy based only on history length or weak-view share. The detailed report and aggregate inputs are in `reference_results/posthoc/POSTHOC_ANALYSIS_RU.md`. A film's additive NDCG contribution is calculated with the original full user IDCG; it is not a retraining ablation or a separately renormalized subgroup NDCG. No user identifiers or individual recommendation rows are distributed.

```bash
python analysis/plot_posthoc.py --results-dir reference_results/posthoc --output-dir outputs/posthoc_figures
```

To recompute from a complete scientific results ZIP and official raw data:

```bash
python analysis/posthoc_analysis.py --archive outputs/full_results.zip --data-archive /path/to/data_en.zip --output-dir outputs/posthoc
python analysis/write_posthoc_report.py --results-dir outputs/posthoc
python analysis/plot_posthoc.py --results-dir outputs/posthoc --output-dir outputs/posthoc/figures
```

The reporting helper's named film illustration (`item_id=9728`) describes the supplied run; it should be reviewed if you apply the script to a new run. Every film's contribution is saved, rather than only selected positive examples. The processing and plots are reproducible; interpretative prose requires review against new results.

## Data and reuse

[KION dataset paper](https://arxiv.org/abs/2209.00325), [official pinned archive](https://raw.githubusercontent.com/irsafilo/KION_DATASET/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_en.zip).

SHA-256: `fddc5b2215bfff49bcfa21e464441f1fc6dc30f8f4afe41d3f23bf3fc45f3542`.

KION records aggregate multiple views at the last interaction date. Recorded percentage is not a per-session completion time, satisfaction measure, causal debiasing target, or revenue outcome. See `DATA_SOURCES.md`. Code is MIT licensed; this does not relicense the third-party dataset, cited papers or dependencies. Authors and publication identifiers should be added once confirmed; no invented citation DOI is supplied.
