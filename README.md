# Usage
To run the test suite, provide reference inputs and fluxes, as specified in `test_suite.py` and run:
```
python test_suite.py -n [NUMBER OF INPUTS TO RANDOMLY SAMPLE FROM REFERENCES] -o [OUTPUT DIRECTORY]
```
If you already have a directory of TGLF inputs and simulated outputs that you would like to compute fluxes on (post-processing and plotting only), comment out the following function calls at the bottom of `test_suite.py`:
```
setup_tglf_tests(...)
run_tglf(...)
```
I'll clean this up when I get a chance to be a little less ugly; I have a midterm to attend to today...

# Installation
Clone the git submodules with GACODE and Pyrokinetics libraries:
```
git pull
git submodule update --init --recursive
```

Create a new conda environment and install requirements using pip:
```
conda create --name gacode-sim-verification
conda activate gacode-sim-verification
pip install -r requirements.txt
```

For a fresh CGYRO and TGLF compilation, use:
```
python setup_gacode.py
```

