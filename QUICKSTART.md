# Quick Start Guide

## The Problem: Path with Colon

Your project path contains `:` which Python's venv doesn't support.

## The Solution (Pick One)

### Option 1: Rename Directory ⭐ RECOMMENDED

```bash
# One command to rename:
mv "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses : Code" \
   "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses_Code"

# Then setup:
cd "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses_Code/TCN"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run:
python run_all.py
```

### Option 2: Use Conda

```bash
conda create -n tcn python=3.10 -y
conda activate tcn
cd "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses : Code/TCN"
pip install -r requirements.txt
python run_all.py
```

### Option 3: Create Symlink

```bash
ln -s "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses : Code/TCN" ~/tcn
cd ~/tcn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

---

## After Setup

```bash
# Run the complete pipeline
python run_all.py

# Or step-by-step
python run_layer1.py      # Data & features (5-10 min)
python run_layer2.py      # Windowing (1 min)
python run_layer3.py      # Training (5-15 min)
python run_backtest.py    # Backtest (2 min)
```

Results appear in `results/figures/`

---

## That's It!

For detailed instructions, see **SETUP.md** and **USAGE.md**.

