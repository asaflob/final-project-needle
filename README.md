# Where is the Hidden Value in the NBA Draft?

Final project for 67978 — A Needle in a Data Haystack (HUJI).

## Main question

**Where is the hidden value in the NBA draft?** Broken into three sub-questions:

1. What is each draft pick actually worth? (player outcomes measured fairly, using each
   player's **first 4 seasons** — the rookie-contract window — so career length does not
   bias the comparison)
2. Can we predict a prospect's NBA value from pre-draft data better than the draft order
   itself does — and which player profiles get systematically over- or under-drafted?
3. Does media hype help or hurt? (does hype predict where a player gets picked better
   than it predicts how good he becomes?)

## Project structure

```
├── data/
│   ├── raw/           # downloaded source files (never edited by hand)
│   └── processed/     # clean tables produced by src/data_prep.py
├── src/
│   ├── download_data.py   # shared: downloads the draft data into data/raw/
│   ├── data_prep.py       # shared: builds the first-4-seasons WS table (core metric)
│   ├── audit.py           # shared: data representativeness checks
│   ├── download_combine.py# optional: combine measurements for q2's ADVANCED set
│   ├── q1_value_curve.py  # Q1: pick value curve, bust/contributor/star rates
│   ├── q2_model.py        # Q2: prospect model vs draft-order baseline (SIMPLE
│   │                      #     features always; ADVANCED if combine data exists)
│   ├── q2_clusters.py     # (next) Q2: player archetypes
│   └── q3_hype.py         # (next) Q3: hype score + hype gap
├── figures/           # every script saves its PNGs here (used in the writeup)
└── app.py             # (next) Streamlit app: Player Lookup + Draft Map
```

## How to run

```
pip install -r requirements.txt
python src/download_data.py     # downloads draft.csv; prints Kaggle instructions
python src/data_prep.py         # builds data/processed/draft_value.csv
python src/audit.py             # coverage + missingness report, saves audit figure
python src/q1_value_curve.py    # Q1 figures + expected value per pick
```

## Data sources

1. **Draft history (2000-2020, both rounds)** — draft_full.csv, scraped from the
   official Basketball-Reference draft pages by `src/download_data.py`.
   (The milestone's draft.csv covered only 2000-2009 round 1 and had broken
   encoding for accented names — documented in the writeup's data-issues section.)
2. **Season-level advanced stats** — `Advanced.csv` from the Kaggle dataset
   "NBA Stats (1947-present)" by Sumitro Datta. Needed for per-season Win Shares
   (the raw draft table only has career totals). Download manually from Kaggle and
   place in `data/raw/Advanced.csv`.
