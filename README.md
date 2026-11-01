# Phishing website classification with multilayer perceptrons

Code, data and experiment outputs for Project 2 (UCI Phishing Websites, id 327).

`phishing_websites.csv` is included. No separate download needed.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python run_experiments.py baselines grid encodings overfit vanishing
python run_experiments.py final costs local_optima datasize missing dedup
```

Quick check:
```bash
python run_experiments.py final
```

Compare new files under `results/` with the ones shipped here. Selected MLP test accuracy is about 0.974; Random Forest on the same split about 0.976.

## Layout
- `phishing_websites.csv` — dataset
- `common.py` — load, split, MLP, train, metrics
- `run_experiments.py` — experiment stages
- `analysis.py`, `analyze_grid.py`, `oscillation.py`, `test_reference.py` — helpers
- `results/` — metrics from the reported runs
- `figures/` — plots
