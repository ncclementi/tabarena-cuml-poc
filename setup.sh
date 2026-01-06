#!/bin/bash
set -e  # Exit on error

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Setup script for TabArena cuML PoC environment."
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message and exit"
    echo "  --ignore-cuda-warning   Bypass CUDA 13 version check"
    echo ""
    echo "This script will:"
    echo "  1. Install uv (if not present)"
    echo "  2. Create a Python 3.12 virtual environment"
    echo "  3. Check for CUDA 13"
    echo "  4. Install PyTorch with CUDA 13 support"
    echo "  5. Clone and install AutoGluon"
    echo "  6. Clone and install TabArena"
    echo "  7. Install cuML"
    exit 0
}

# Parse command line arguments
IGNORE_CUDA_WARNING=false
for arg in "$@"; do
    case $arg in
        -h|--help)
            usage
            ;;
        --ignore-cuda-warning)
            IGNORE_CUDA_WARNING=true
            shift
            ;;
    esac
done

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
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+')
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
    echo "CUDA version $CUDA_VERSION detected (from nvidia-smi)"
    if [ "$CUDA_MAJOR" != "13" ]; then
        if [ "$IGNORE_CUDA_WARNING" = true ]; then
            echo "WARNING: Expected CUDA 13 but found CUDA $CUDA_VERSION. Continuing anyway (--ignore-cuda-warning)."
        else
            echo "ERROR: Expected CUDA 13 but found CUDA $CUDA_VERSION."
            echo "Use --ignore-cuda-warning to bypass this check."
            exit 1
        fi
    fi
else
    if [ "$IGNORE_CUDA_WARNING" = true ]; then
        echo "WARNING: nvidia-smi not found. Continuing anyway (--ignore-cuda-warning)."
    else
        echo "ERROR: nvidia-smi not found. NVIDIA driver with CUDA 13 support must be installed."
        echo "Use --ignore-cuda-warning to bypass this check."
        exit 1
    fi
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
uv pip install --prerelease=allow -e ./tabarena[benchmark]

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


