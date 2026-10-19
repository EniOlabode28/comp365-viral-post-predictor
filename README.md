# Why Did This Post Go Viral? Predicting Facebook Post Engagement Before Publishing

**Enioluwa Olabode · COMP 365 Individual Project · Final reproducible materials**

Can the information a page manager knows *before* posting (post type, category, timing, whether it's paid, page size) predict whether a Facebook post will land in the top 25% for total interactions? This repository rebuilds every table and figure in the paper with one command.

## Quick start

```bash
git clone <this repo>
cd <repo folder>
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```

The run takes about 25 seconds on a laptop CPU. It writes:

| Output | Contents |
|---|---|
| `results/output.txt` | Full log of every test, with the package versions used |
| `results/table_time_test.csv` | **Main result:** test-month scores for every model, with bootstrap 95% CIs |
| `results/table_selection_validation.csv` | Validation scores used to choose each model's feature set and threshold |
| `results/table_time_test_posthoc_default_threshold.csv` | Post hoc check at threshold 0.50 (not used for any choice) |
| `results/table_time_test_by_type.csv`, `table_time_test_missed.csv` | Weak-case analysis |
| `results/table_monthly.csv` | Posts, median interactions and page size by month |
| `results/table_random_cv.csv`, `table_random_cv_ablation.csv` | Secondary random-split cross-validation (Checkpoint 2 setup) |
| `results/summary.json` | Headline numbers |
| `figures/fig1_time_split.png` … `fig4_test_by_type.png` | The four figures in the paper |

## Environment

| | Version |
|---|---|
| Python | 3.13.7 (3.11 or newer works with the pinned packages) |
| numpy | 2.4.1 |
| pandas | 3.0.0 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.1 |
| matplotlib | 3.11.1 |

- **Hardware:** any modern CPU. No GPU needed. Tested on an Apple Mac laptop.
- **Randomness:** every random step uses seed 42. The Random Forest uses `n_jobs=1`.
- **Verified:** running in a fresh virtual environment from `requirements.txt` produced identical output and byte-identical CSVs.

## Data

`data/dataset_Facebook.csv` is the **Facebook Metrics** dataset from the UCI Machine Learning Repository (dataset #368), released under CC BY 4.0:

> Moro, S., Rita, P., & Vala, B. (2016). Predicting social media performance metrics and evaluation of the impact on brand building: A data mining approach. *Journal of Business Research*, 69(9), 3341–3351.

It contains 500 posts from one cosmetics brand's Facebook page in 2014. One row with a missing "Paid" value is dropped, leaving 499.

To download it yourself: https://archive.ics.uci.edu/dataset/368/facebook+metrics (`dataset_Facebook.csv`, semicolon-separated). SHA-256 of the file used: `666ec82952b7ce90b464f5a87ec4f0fe92d9a63208f2fcb2d0443cb5f31c3f4c`.

## What the script does

**Features (known before posting):** Page total likes, Type (one-hot), Category, Post Month, Post Weekday, Post Hour, Paid. All "Lifetime …", comment, like and share columns are excluded, because they are only known after posting.

**Label:** high engagement = Total Interactions above the 75th percentile of the *training* posts.

**Models:**

| Model | Settings |
|---|---|
| Majority-class baseline | `DummyClassifier(strategy="most_frequent")` |
| Rule-of-thumb baseline | HIGH if paid, video or status update (no learning) |
| Decision Tree | max_depth 4, min_samples_leaf 10, balanced class weights |
| Logistic Regression | StandardScaler, balanced class weights, max_iter 1000 |
| Random Forest | 300 trees, max_depth 6, min_samples_leaf 5, balanced class weights |

All models are trained from scratch. Nothing is pretrained.

### Part 1: time-ordered evaluation (main result)

The posts are put in publication order. The file lists December first, so it is reversed, and the script checks that months only increase.

| Months | Role |
|---|---|
| 1–6 (222 posts) | Fit candidate models |
| 7–9 (122 posts) | Choose each model's feature set and decision threshold by F1 |
| 10–12 (155 posts) | Test set, scored once |

- **Feature sets compared:** all 7 features, or without Page total likes and Post Month (which mainly track time).
- **Threshold grid:** 0.05 to 0.95 in steps of 0.01.
- **Final models:** each model is refit on months 1–9 with its chosen settings. The cutoff is then recomputed from months 1–9 only, and the models are scored on months 10–12.
- **Statistics:** 2000-resample bootstrap confidence intervals, and a paired bootstrap comparing the selected model with the rule of thumb.

### Part 2: random-split cross-validation (comparison with Checkpoint 2)

- Repeated stratified 5-fold cross-validation (5 repeats).
- **Corrected resampled t-test** (Nadeau & Bengio, 2003), because the repeated folds share posts.
- **200-shuffle label-permutation test**, using scikit-learn's `permutation_test_score`.
- Feature ablation.

Here the cutoff uses all posts, as in Checkpoint 2, so this part is for comparison only.

## Changes since Checkpoint 2

- Added the time-ordered evaluation with training-only cutoffs and validation-only selection.
- Replaced "won 18 of 25 folds" and the 5-shuffle check with the corrected t-test and the 200-permutation test.
- Pinned all package versions and fixed the Random Forest seed and threads. Checkpoint 2's Random Forest numbers came from an unpinned environment, so they will not match exactly. The pinned values are in `results/table_random_cv.csv`.

## Credits and AI-use disclosure

- **Libraries:** scikit-learn, pandas, NumPy, SciPy, Matplotlib.
- **Data:** UCI / Moro et al. (2016).
- **AI assistance:** Claude (Anthropic) was used to help write, debug and clean up `run_all.py`, and to draft this README and the reproducibility setup. The experimental design follows the instructor's Checkpoint 2 feedback. All results come from actually running the code and can be reproduced with the command above. The paper itself was written without AI assistance.
