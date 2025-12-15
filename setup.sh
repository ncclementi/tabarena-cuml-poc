#!/bin/bash
set -e  # Exit on error

echo "=== TabArena cuML PoC Setup Script ==="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Commit hashes (set to empty string to use latest main)
AUTOGLUON_COMMIT="708849b"  # Example: "abc123def456"
TABARENA_COMMIT="aeff2d8"   # Example: "789ghi012jkl"

# Step 0: Check and install uv if needed
echo "[Step 0/6] Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the environment to make uv available in current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo "ERROR: Failed to install uv. Please install manually."
        exit 1
    fi
    echo "uv installed successfully"
else
    echo "uv is already installed ($(uv --version))"
fi

# Step 1: Create uv environment with Python 3.12
echo ""
echo "[Step 1/6] Creating uv environment with Python 3.12..."
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

uv venv --python 3.12 
echo "Virtual environment created"

# Activate the virtual environment
source .venv/bin/activate

# Step 2: Check for CUDA 13
echo ""
echo "[Step 2/6] Checking for CUDA 13..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
    echo "CUDA version $CUDA_VERSION detected"
    if [ "$CUDA_MAJOR" != "13" ]; then
        echo "ERROR: Expected CUDA 13 but found CUDA $CUDA_VERSION."
        exit 1
    fi
else
    echo "ERROR: nvcc not found. CUDA 13 must be installed and in PATH."
    exit 1
fi

# Step 3: Install PyTorch
echo ""
echo "[Step 3/6] Installing PyTorch..."
uv pip install \
    --index-url https://download.pytorch.org/whl/cu130 \
    torch \
    torchvision \
    && echo "torch installed"

uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    pytorch-metric-learning \
    && echo "pytorch-metric-learning installed"

# Step 4: Clone and install AutoGluon
echo ""
echo "[Step 4/6] Installing AutoGluon..."
AUTOGLUON_REPO="https://github.com/csadorf/autogluon"
if [ -d "autogluon" ]; then
    echo "AutoGluon directory exists. Checking remote..."
    cd autogluon
    CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
    if [ "$CURRENT_REMOTE" != "$AUTOGLUON_REPO" ] && [ "$CURRENT_REMOTE" != "${AUTOGLUON_REPO}.git" ]; then
        echo "Remote mismatch. Deleting and re-cloning..."
        cd "$SCRIPT_DIR"
        rm -rf autogluon
        git clone "$AUTOGLUON_REPO"
        cd autogluon
    else
        echo "Pulling latest from master..."
        git pull origin master
    fi
else
    echo "Cloning AutoGluon from $AUTOGLUON_REPO..."
    git clone "$AUTOGLUON_REPO"
    cd autogluon
fi

# Checkout specific commit if specified
if [ -n "$AUTOGLUON_COMMIT" ]; then
    echo "Checking out commit $AUTOGLUON_COMMIT..."
    git checkout "$AUTOGLUON_COMMIT"
fi

echo "Running full_install.sh..."
chmod +x full_install.sh
./full_install.sh

cd "$SCRIPT_DIR"
echo "AutoGluon installed"

# Step 5: Clone and install TabArena
echo ""
echo "[Step 5/6] Installing TabArena..."
TABARENA_REPO="https://github.com/csadorf/tabarena"
if [ -d "tabarena" ]; then
    echo "TabArena directory exists. Checking remote..."
    cd tabarena
    CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
    if [ "$CURRENT_REMOTE" != "$TABARENA_REPO" ] && [ "$CURRENT_REMOTE" != "${TABARENA_REPO}.git" ]; then
        echo "Remote mismatch. Deleting and re-cloning..."
        cd "$SCRIPT_DIR"
        rm -rf tabarena
        git clone "$TABARENA_REPO"
        cd tabarena
    else
        echo "Pulling latest from main..."
        git pull origin main
    fi
else
    echo "Cloning TabArena from $TABARENA_REPO..."
    git clone "$TABARENA_REPO"
    cd tabarena
fi

# Checkout specific commit if specified
if [ -n "$TABARENA_COMMIT" ]; then
    echo "Checking out commit $TABARENA_COMMIT..."
    git checkout "$TABARENA_COMMIT"
fi

echo "Installing TabArena with benchmark extras..."
uv pip install -e tabarena/[benchmark]

cd "$SCRIPT_DIR"
echo "TabArena installed"

# Step 6: Install cuML
echo ""
echo "[Step 6/6] Installing cuML..."
uv pip install "cuml-cu13==25.12.00"
echo "cuML installed"

echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""


