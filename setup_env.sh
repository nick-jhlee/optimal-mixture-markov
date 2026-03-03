#!/bin/bash
set -e

echo "=========================================="
echo "Setting up environment for optimal-mixture-markov"
echo "=========================================="

# Check if uv is installed, if not, install it
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    echo "uv installed successfully!"
else
    echo "uv is already installed."
fi

# Create virtual environment named 'markov'
echo ""
echo "Creating virtual environment 'markov'..."
uv venv markov --python 3.11

# Activate the environment and install dependencies
echo ""
echo "Installing dependencies..."
source markov/bin/activate

# Install packages directly (not as an editable package)
uv pip install numpy scipy pandas matplotlib scikit-learn numba joblib tqdm cloudpickle gymnasium farama-notifications tensorflow

# create results directory
mkdir -p results

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source markov/bin/activate"
echo ""
echo "Then you can run the scripts:"
echo "  python main_synthetic.py"
echo "  python ablation1.py"
echo "  python ablation2.py"
echo "  python ablation3.py"
echo "  python ablation4.py"
echo "  python ablation5.py"
echo ""
