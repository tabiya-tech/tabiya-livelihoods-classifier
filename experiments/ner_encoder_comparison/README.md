# Spanish NER encoder comparison

Which encoder model (the pretrained network we fine-tune) finds **Occupation**, **Skill** and
**Qualification** phrases best in Spanish job ads? This folder trains each candidate on the
Peru job-ads corpus and scores it the same way, so we can choose on numbers.

| Config file | Model | Why it is in the race |
|---|---|---|
| `configs/runs/xlmr-base.yaml` | `FacebookAI/xlm-roberta-base` | Baseline: the multilingual version of the current English model |
| `configs/runs/mroberta.yaml` | `BSC-LT/mRoBERTa` | Same design, trained on more Spanish |
| `configs/runs/mmbert-base.yaml` | `jhu-clsp/mmBERT-base` | Newer design (ModernBERT), multilingual |
| `configs/runs/mrbert-es.yaml` | `BSC-LT/MrBERT-es` | Newer design (ModernBERT), Spanish-focused |

The full reasoning is in the experiment design doc. This README covers how to run it.

## What is in git and what is not

In git: the code, the configs, the tests and `splits/` (the frozen list of which ads are
train, dev and test). Not in git, rebuilt or downloaded on your machine: `data/` (the
corpus), `runs/` (one folder per training run), `reports/`, and the models from Hugging Face.

## One-time setup (windows, cuda)
If you're using an nvidia GPU, install the CUDA build of PyTorch. (the CPU-only build is much slower. macbook m-series gpus are autodetected ) 

Then install this package and its dependencies.
Use PowerShell, starting in this folder (`experiments\ner_encoder_comparison`). Python 3.10 to
3.12 works.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
# If PowerShell refuses to run the activate script, run this once, then activate again:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

python -m pip install --upgrade pip

# 1) The GPU (CUDA) build of torch FIRST. Plain "pip install torch" gives the CPU-only build.
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2) This package and everything else. -e (editable) is required: it lets the code find
#    data/, splits/ and runs/ in this folder, whichever folder you run from.
pip install -e ".[dev]"

# 3) Check the GPU is visible. It should print True and the card's name.
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4) Optional: keep downloaded models inside this folder (it is git-ignored).
#    Set it in every new terminal, or add it to your user environment variables.
$env:HF_HOME = "$PWD\.hf-cache"
```

Run the tests (about a second): `pytest`

## Steps, in order

### 1. Prepare the data (once, about a minute)

```powershell
python -m ner_experiments.cli.prepare_data --config configs/prepare.yaml
```

This downloads the Peru corpus to `data/raw/`. It drops two ads that are word-for-word copies
of others (it prints which), converts the Peru tags to our three types, and writes
`data/processed/{train,dev,test}.jsonl` using the frozen split in `splits/`. You should see
exactly these counts. If they differ, stop: something changed in the data.

| split | ads | sentences | Occupation | Skill | Qualification |
|---|---|---|---|---|---|
| train | 558 | 8,971 | 910 | 2,909 | 1,854 |
| dev | 80 | 1,359 | 112 | 443 | 320 |
| test | 160 | 2,654 | 247 | 817 | 563 |

Phrases that were moved from Skill to Qualification (like "Técnico" or "postgrado") are
listed in `data/processed/moved_to_qualification.csv`. Skim it once.

### 2. Check the tokenizers (once, about a minute)

```powershell
python -m ner_experiments.cli.check_tokenizers
```

This makes sure each model's tokenizer (the part that splits words into pieces) lines up
with our words. All four should print `PASS`.

### 3. Train: one run at a time, set by hand

Each run is one command with one config file:

```powershell
python -m ner_experiments.cli.run_experiment --config configs/runs/xlmr-base.yaml
```

Before each run, edit the values in the config file (learning rate, seed, and so on), or
override them for that one terminal with environment variables. An environment variable
always wins over the file:

```powershell
$env:NER_EXP_SEED = "2"
$env:NER_EXP_EVALUATE_ON_TEST = "true"
python -m ner_experiments.cli.run_experiment --config configs/runs/mrbert-es.yaml
Remove-Item Env:NER_EXP_SEED, Env:NER_EXP_EVALUATE_ON_TEST   # clear them afterwards
```

Every setting has a variable: `NER_EXP_` plus the setting name in capitals. Nested ones use
a double underscore, e.g. `NER_EXP_PATHS__RUNS_DIR = "D:\ner-runs"`. You can also point to
the config with `NER_EXP_CONFIG_FILE` instead of `--config`. Every setting, with what it
does, is in `src/ner_experiments/config.py` (`ExperimentConfig`). A misspelt setting name
is rejected rather than ignored.

The run prints which of its 7 stages it is in. During training it prints the progress of
each pass over the data (epoch), the validation (dev) score after each pass, the best
score so far and the time left. Expect roughly 10 to 20 minutes per run on the 4060.

Each run gets its own folder, `runs/<run name>/`. The name is built from the settings, e.g.
`mrbert-es-lr3e-05-seed2`. The folder holds:

- `effective_config.yaml`: every setting the run actually used
- `result.json`: scores, score after each pass, timing
- `predictions_dev.jsonl` (and `predictions_test.jsonl` for final runs): the model's tags,
  used later for the significance test
- `charts/`: training loss and dev score per pass, and scores per phrase type
- `model/`: only if `keep_model: true`

A run whose folder already has a `result.json` will not start again. This protects
finished runs. To redo one, set `overwrite: true`. A run that crashed halfway can simply be
started again.

### 4. Look at the results (any time, seconds)

```powershell
python -m ner_experiments.cli.report                  # table + charts in reports/latest/
python -m ner_experiments.cli.report --include-test   # also the test-set chart (final runs)
python -m ner_experiments.cli.compare --split test    # are the gaps real or luck?
```

`report` writes `summary.csv` / `summary.md` (one row per run) and these charts:

- models
- learning-rate sweep
- learning curve

Which charts we keep is still open. New charts go in `src/ner_experiments/reporting/charts.py`.

`compare` runs the paired bootstrap (it re-scores many random resamples of the test ads) on
the final runs. It says, for each pair of models, whether the gap is a real difference or
too close to call.

## The run plan

Do the steps in order. You decide each run by editing the config. There is deliberately no
script that runs everything.

**Step A. Learning-rate sweep (12 runs, dev only).** For each of the 4 models, train with
`learning_rate` 2e-5, 3e-5 and 5e-5, with `seed: 1` and `evaluate_on_test: false`. Then
run `report` and, for each model, pick the learning rate with the highest dev macro F1
(`learning_rate_sweep.png`).

**Step B. Final runs (20 runs, test scored).** For each model, at its chosen learning rate,
train seeds 1 to 5 with `evaluate_on_test: true`. Seed 1 already exists from step A, but
without test scores, so redo it with `overwrite: true`. (The design doc counts 16 here; the
4 extra runs keep the test set untouched until this step, which is the point of locking it.)
Then:

```powershell
python -m ner_experiments.cli.report --include-test
python -m ner_experiments.cli.compare --split test
```

**Step C. Learning curve (6 runs, dev only).** Does more labelled data still help? Take the
winner and XLM-R at their chosen learning rate, with `train_fraction` 0.25, 0.5 and 0.75,
`seed: 1` and `evaluate_on_test: false`. The 100% point comes from step B. If the line is
still climbing at 100%, more labelled ads would pay off.

**Step D. Speed check (optional, ties only).** If two models are tied, retrain one seed of
each with `keep_model: true`, then time them on the CPU:

```powershell
python -m ner_experiments.cli.benchmark_latency --model-dir runs/<run name>/model --threads 4
```

### Rules that keep the comparison fair

- Never pick settings by looking at test scores. Test is only scored in step B.
- Don't change `splits/`. If you ever need a new split (`prepare_data --new-split`), all
  earlier runs stop being comparable.
- Change one thing at a time between runs, and let the run name record it.

## Running more than one at a time

There is one GPU, so the fastest plan is usually one training run at a time. Runs never
share files, so it is safe to start a second run in another terminal. They will each go
slower, though. A pair of the smaller models may fit together in the 4060's 8 GB. If you
see "CUDA out of memory", go back to one at a time, or use `batch_size: 8` with
`gradient_accumulation_steps: 2`. `report` and `compare` are cheap and can run while a
training run is going.

## Troubleshooting

- **"CUDA was requested but PyTorch can't see an NVIDIA GPU"**: torch is the CPU-only build.
  Run `pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall`.
- **Model download stalls or warns about Xet**: `hf_xet` is installed by `pip install -e .`.
  Just re-run the command; downloads resume.
- **Counts in step 1 differ**: don't train. Check that `splits/` is unchanged (`git status`).
- **Data loader errors on Windows**: keep `dataloader_workers: 0` (the default).

## Code layout

```
src/ner_experiments/
  config.py            typed settings (pydantic), YAML + environment variables
  schemas.py           typed records: sentences, predictions, scores, run results
  tags.py              our 7 tags and phrase extraction
  progress.py          stage banners and training progress
  run_files.py         reading and writing a run folder
  corpus/              Peru corpus download, tag conversion, split, file reading
  modeling/            tokenizer alignment, training, prediction, device choice
  evaluation/          exact and partial matching, scoring, bootstrap, CPU latency
  reporting/           matplotlib charts, summary table
  cli/                 the commands above, one file each
tests/                 pytest
```
