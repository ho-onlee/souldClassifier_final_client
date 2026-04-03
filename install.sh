#!/bin/bash

# Create a new virtual environment
echo "Creating virtual environment..."
python -m venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/Scripts/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install \
    torch \
    numpy \
    librosa \
    sounddevice

echo "Installation complete!"
echo "To activate the virtual environment in the future, run:"
echo "  source .venv/Scripts/activate"
source .venv/Scripts/activate