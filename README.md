A Julia / Python toolbox for simulating open quantum systems with the implemented physics following the setup explained in the paper https://arxiv.org/abs/2501.13675.

---

## Usage

A folder in the DAGS folder should be created with the integer number of the experiment. For example for the first experiment we start with 1 which is already provided.

Within a given experiment, we create and run a main.py file which is also provided as an example in experiment 1.

The main.py file will create a DAG (directed acyclic graph) of jobs with specified inputs saved in an h5 file. The inputs and their options are also explained in the example provided for main.py. 

Running this DAG using condor_submit_dag run.dag will trigger the run.sub file for each specified job within the DAG.

The run.sub file then calls the run.jl file where our time evolution simulation is performed making use of the utilities.jl file. 

For plotting results the main.py file can further be used. The plotting functions are included in the example already provided.

---
