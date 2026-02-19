# Usage
To run the test suite, provide reference inputs and fluxes, as specified in `test_suite.py` and run:
```
python test_suite.py -n [NUMBER OF INPUTS TO RANDOMLY SAMPLE FROM REFERENCES] -o [OUTPUT DIRECTORY]
```

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

