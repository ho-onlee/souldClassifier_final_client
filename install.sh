#!/bin/bash

# Check if virtual environment already exists
if [ ! -d ".venv" ]; then
    # Create a new virtual environment
    echo "Creating virtual environment..."
    python -m venv .venv
else
    echo "Virtual environment already exists."
fi
sudo apt update
sudo apt install portaudio19-dev portaudio19-doc
# Activate the virtual environment
echo "Activating virtual environment..."

source .venv/bin/activate

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
echo "  source .venv/bin/activate"
source .venv/bin/activate