# Setup Instructions

## ⚠️ Important: Path Issue

Your project path contains a colon `:` in the directory name "Courses : Code", which Python's `venv` module doesn't support. 

**Choose one of the solutions below:**

---

## Solution 1: Rename Directory (Recommended)

Rename the parent directory to remove the colon:

```bash
# Current path:
# /Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses : Code/TCN

# Rename to:
mv "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses : Code" \
   "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses_Code"

# New path:
# /Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses_Code/TCN
```

Then create the venv:
```bash
cd "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses_Code/TCN"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Solution 2: Use Conda (Alternative)

Conda doesn't have the path restriction:

```bash
# Install Conda if you don't have it (https://docs.conda.io/en/latest/miniconda.html)

# Create conda environment
conda create -n tcn-trading python=3.10 -y
conda activate tcn-trading

# Navigate to project and install
cd "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses : Code/TCN"
pip install -r requirements.txt
```

To use in the future:
```bash
conda activate tcn-trading
cd "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses : Code/TCN"
python run_all.py
```

---

## Solution 3: System-Wide Install (Not Recommended)

Install packages system-wide (not isolated):

```bash
cd "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses : Code/TCN"
pip3 install --user -r requirements.txt
python3 run_all.py
```

⚠️ **Warning**: This installs packages globally and may cause conflicts with other projects.

---

## Solution 4: Create Symlink

Create a symlink without the colon:

```bash
# Create symlink
ln -s "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses : Code/TCN" \
      ~/tcn-trading

# Use the symlink
cd ~/tcn-trading
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then always work through the symlink:
```bash
cd ~/tcn-trading
source .venv/bin/activate
python run_all.py
```

---

## Recommended Approach

**I recommend Solution 1 (rename directory)** as it's the cleanest and most maintainable.

Here's the exact command:

```bash
# Rename the directory
mv "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses : Code" \
   "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses_Code"

# Navigate to new location
cd "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses_Code/TCN"

# Create venv
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python -c "import torch; import pandas; print('✓ Setup successful!')"
```

---

## After Setup

Once your environment is ready, you can run:

```bash
# Activate environment (if not already)
source .venv/bin/activate  # or: conda activate tcn-trading

# Run complete pipeline
python run_all.py

# Or run individual layers
python run_layer1.py      # Data & features
python run_layer2.py      # Windowing
python run_layer3.py      # Training
python run_backtest.py    # Backtest
```

---

## IDE Setup

### VS Code

1. Open the project folder (use the new path if renamed)
2. Press `Cmd+Shift+P`
3. Type "Python: Select Interpreter"
4. Choose `.venv/bin/python` or the conda environment

### PyCharm

1. Open project (use new path if renamed)
2. Preferences → Project → Python Interpreter
3. Add → Existing environment
4. Select `.venv/bin/python` or conda environment

---

## Verifying Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Test imports
python -c "import torch; import pandas; import numpy; print('✓ All dependencies installed')"

# Quick test
python -c "from utils import load_config; print('✓ Project imports working')"
```

---

## What Gets Installed

After setup, you'll have these packages:

- **torch** (2.0+) - Deep learning
- **pandas** (2.0+) - Data manipulation
- **numpy** (1.24+) - Numerical computing
- **scikit-learn** (1.3+) - ML metrics
- **matplotlib/seaborn** - Visualization
- **yfinance** - Market data
- **PyYAML** - Configuration
- And more (see `requirements.txt`)

---

## Troubleshooting

### Still getting path errors?

Use **Solution 2 (Conda)** or **Solution 4 (Symlink)** instead.

### Import errors after installation?

```bash
# Make sure environment is activated
which python  # Should point to .venv/bin/python or conda env

# Reinstall if needed
pip install --force-reinstall -r requirements.txt
```

### Permission errors?

```bash
# Don't use sudo! Instead:
pip install --user -r requirements.txt
```

---

## Quick Reference

**Activate environment:**
- venv: `source .venv/bin/activate`
- conda: `conda activate tcn-trading`

**Deactivate:**
- venv: `deactivate`
- conda: `conda deactivate`

**Run system:**
```bash
python run_all.py          # Complete pipeline
python run_layer1.py       # Just data prep
python run_backtest.py     # Just backtest
```

---

## Next Steps

1. ✓ Choose and complete setup solution
2. ✓ Verify installation
3. → Edit `config/params.yaml`
4. → Run `python run_all.py`
5. → Review results in `results/figures/`

See **USAGE.md** for detailed instructions!
