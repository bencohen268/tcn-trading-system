#!/bin/bash
# Quick activation script for the virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Run ./setup.sh first to create the environment."
    exit 1
fi

echo "Activating TCN virtual environment..."
source "$SCRIPT_DIR/.venv/bin/activate"

echo "✓ Virtual environment activated"
echo ""
echo "Ready to run:"
echo "  python run_all.py         # Complete pipeline"
echo "  python run_layer1.py      # Data & features"
echo "  python run_layer2.py      # Windowing"
echo "  python run_layer3.py      # Training"
echo "  python run_backtest.py    # Backtest"

