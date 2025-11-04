#!/bin/bash
# Setup script for TCN Trading System

echo "=================================================="
echo "TCN Trading System - Setup"
echo "=================================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "Working directory: $SCRIPT_DIR"
echo ""

# Create virtual environment
echo "[1/3] Creating virtual environment..."
python3 -m venv .venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    echo "Try running: python3 -m pip install --upgrade pip"
    exit 1
fi

echo "✓ Virtual environment created in .venv/"

# Activate virtual environment
echo ""
echo "[2/3] Activating virtual environment..."
source .venv/bin/activate

echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "[3/3] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "=================================================="
echo "✓ Setup complete!"
echo "=================================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the complete pipeline:"
echo "  python run_all.py"
echo ""
echo "To run individual layers:"
echo "  python run_layer1.py      # Data & features"
echo "  python run_layer2.py      # Windowing"
echo "  python run_layer3.py      # Training"
echo "  python run_backtest.py    # Backtest"
echo ""
echo "See USAGE.md for detailed instructions."
echo "=================================================="

