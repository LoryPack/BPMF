# BPMF

Python implementation of Bayesian Probabilistic Matrix Factorization, following: 

http://www.cs.toronto.edu/~rsalakhu/papers/bpmf.pdf

## Run simulations: 

To run simulations with different dataset or parameters, you have to modify the the `script_bpmf.py` file. It calls the BPFM function in file `bpmf.py`, together with some other functions from the `utilities.py` file. To run it, simply type: 

`python script_bpmf.py`

The script handles with dataset in `.mtx` format. Some example input data is contained in `data/`
