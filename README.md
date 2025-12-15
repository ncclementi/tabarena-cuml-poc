# tabarena-cuml-poc
Benchmarking sklearn estimators accelerated via cuml accel 

## Running the cuML-accelerated benchmark quickstart

To test cuML-accelerated estimators with TabArena, you can use the pre-configured example, copy the file in the tabarena repo location indicated below after installation is done. 

```bash
cp run_quickstart_tabarena_cuml.py tabarena/examples/benchmarking/
cd ./tabarena/examples/benchmarking
python run_quickstart_tabarena_cuml.py
```

This will run the cuML Random Forest benchmark on a few small datasets.  
You can modify the script to experiment with other models or datasets as needed.



## Setup

To set up the environment in a reproducible way, run the provided setup script:

```bash
chmod +x setup.sh
./setup.sh
```

**Requirements:**
- CUDA 13 must be installed and `nvcc` must be in your PATH
- The script will automatically install `uv` if not already present

**What the script does:**
1. Checks for and installs `uv` if needed
2. Creates a Python 3.12 virtual environment
3. Verifies CUDA 13 is available
4. Installs cuML-cu13 version 25.12.00
5. Clones/updates AutoGluon from https://github.com/csadorf/autogluon
6. Clones/updates TabArena from https://github.com/csadorf/tabarena

After setup completes, activate the environment:
```bash
source .venv/bin/activate
```

## TODO: 
- [x] Setup script to install everything in a reproducible way using uv 
    - [x] cuml from nightly, specific version
    - [x] autogluon from https://github.com/csadorf/autogluon main
    - [x] tabarena from https://github.com/csadorf/tabarena main
- [ ] Separately create a test for installation for LR, KNN, RF
    - [ ] one simple dataset that runs with pure autogluon cpu 
    - [ ] one that runs with cuml accel POC 


Notes to create the script. 
0. Install uv 
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
1. Create a uv environment with python 3.12
2. Check we have cuda 13 
3. pip install \
    "cudf-cu13==25.12.00" "cuml-cu13==25.12.00"
4. Install autogluon:

```bash
git clone https://github.com/csadorf/autogluon
cd autogluon && ./full_install.sh
cd ..
```
5. Install tabarena

```bash
git clone https://github.com/csadorf/tabarena
cd tabarena
uv pip install -e tabarena/[benchmark]
```

