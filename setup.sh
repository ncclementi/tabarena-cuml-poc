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
    echo "  7. Build and install treelite from source (if TREELITE_COMMIT is set)"
    echo "  8. Install cuML"
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

# Directory for third-party dependencies
THIRDPARTY_DIR="$SCRIPT_DIR/src/thirdparty"
mkdir -p "$THIRDPARTY_DIR"

# Commit hashes (set to empty string to use latest main/master)
AUTOGLUON_COMMIT="c124491"  # Example: "abc123def456"
TABARENA_COMMIT="aeff2d8"   # Example: "789ghi012jkl"
# TREELITE_REPO="https://github.com/dmlc/treelite"
TREELITE_REPO="https://github.com/dantegd/treelite.git"
TREELITE_COMMIT="0c46c84b9d72174de9b9a6c59e15865d895c6137"          # Set to build treelite from source, e.g., "abc123def456"

# Determine total steps based on whether treelite will be built
if [ -n "$TREELITE_COMMIT" ]; then
    TOTAL_STEPS=7
else
    TOTAL_STEPS=6
fi

# Step 0: Check and install uv if needed
echo "[Step 0/$TOTAL_STEPS] Checking for uv..."
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
echo "[Step 1/$TOTAL_STEPS] Creating uv environment with Python 3.12..."
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
echo "[Step 2/$TOTAL_STEPS] Checking for CUDA 13..."
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
echo "[Step 3/$TOTAL_STEPS] Installing PyTorch..."
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
echo "[Step 4/$TOTAL_STEPS] Installing AutoGluon..."
AUTOGLUON_REPO="https://github.com/csadorf/autogluon"
AUTOGLUON_DIR="$THIRDPARTY_DIR/autogluon"
if [ -d "$AUTOGLUON_DIR" ]; then
    echo "AutoGluon directory exists. Checking remote..."
    cd "$AUTOGLUON_DIR"
    CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
    if [ "$CURRENT_REMOTE" != "$AUTOGLUON_REPO" ] && [ "$CURRENT_REMOTE" != "${AUTOGLUON_REPO}.git" ]; then
        echo "Remote mismatch. Deleting and re-cloning..."
        rm -rf "$AUTOGLUON_DIR"
        git clone "$AUTOGLUON_REPO" "$AUTOGLUON_DIR"
        cd "$AUTOGLUON_DIR"
    else
        echo "Pulling latest from master..."
        git pull origin master
    fi
else
    echo "Cloning AutoGluon from $AUTOGLUON_REPO..."
    git clone "$AUTOGLUON_REPO" "$AUTOGLUON_DIR"
    cd "$AUTOGLUON_DIR"
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
echo "[Step 5/$TOTAL_STEPS] Installing TabArena..."
TABARENA_REPO="https://github.com/csadorf/tabarena"
TABARENA_DIR="$THIRDPARTY_DIR/tabarena"
if [ -d "$TABARENA_DIR" ]; then
    echo "TabArena directory exists. Checking remote..."
    cd "$TABARENA_DIR"
    CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
    if [ "$CURRENT_REMOTE" != "$TABARENA_REPO" ] && [ "$CURRENT_REMOTE" != "${TABARENA_REPO}.git" ]; then
        echo "Remote mismatch. Deleting and re-cloning..."
        rm -rf "$TABARENA_DIR"
        git clone "$TABARENA_REPO" "$TABARENA_DIR"
        cd "$TABARENA_DIR"
    else
        echo "Pulling latest from main..."
        git pull origin main
    fi
else
    echo "Cloning TabArena from $TABARENA_REPO..."
    git clone "$TABARENA_REPO" "$TABARENA_DIR"
    cd "$TABARENA_DIR"
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

# Step 6: Build and install treelite from source (optional)
if [ -n "$TREELITE_COMMIT" ]; then
    echo ""
    echo "[Step 6/$TOTAL_STEPS] Building treelite from source..."
    
    # Check for cmake
    if ! command -v cmake &> /dev/null; then
        echo "ERROR: cmake is required to build treelite but was not found."
        echo "Please install cmake (e.g., 'apt install cmake' or 'conda install cmake')"
        exit 1
    fi
    
    # Check for make
    if ! command -v make &> /dev/null; then
        echo "ERROR: make is required to build treelite but was not found."
        echo "Please install make (e.g., 'apt install build-essential')"
        exit 1
    fi
    
    TREELITE_DIR="$THIRDPARTY_DIR/treelite"
    if [ -d "$TREELITE_DIR" ]; then
        echo "treelite directory exists. Checking remote..."
        cd "$TREELITE_DIR"
        CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
        if [ "$CURRENT_REMOTE" != "$TREELITE_REPO" ] && [ "$CURRENT_REMOTE" != "${TREELITE_REPO}.git" ]; then
            echo "Remote mismatch. Deleting and re-cloning..."
            rm -rf "$TREELITE_DIR"
            git clone --recursive "$TREELITE_REPO" "$TREELITE_DIR"
            cd "$TREELITE_DIR"
        else
            echo "Fetching latest..."
            git fetch origin
        fi
    else
        echo "Cloning treelite from $TREELITE_REPO..."
        git clone --recursive "$TREELITE_REPO" "$TREELITE_DIR"
        cd "$TREELITE_DIR"
    fi
    
    # Checkout specific commit
    echo "Checking out commit $TREELITE_COMMIT..."
    git checkout "$TREELITE_COMMIT"
    git submodule update --init --recursive
    
    # Build shared libraries
    echo "Building treelite shared libraries..."
    rm -rf build
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    
    # Install Python package
    echo "Installing treelite Python package..."
    cd ../python
    uv pip install .
    
    cd "$SCRIPT_DIR"
    echo "treelite installed from source (commit: $TREELITE_COMMIT)"
fi

# Step 6 or 7: Install cuML (depending on whether treelite was built)
if [ -n "$TREELITE_COMMIT" ]; then
    CUML_STEP=7
else
    CUML_STEP=6
fi
echo ""
echo "[Step $CUML_STEP/$TOTAL_STEPS] Installing cuML..."
uv pip install "cuml-cu13==25.12.00"
echo "cuML installed"

echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""


