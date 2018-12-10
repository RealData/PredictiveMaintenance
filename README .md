Prerequisites: 
 - `azureml-defaults` package should be installed 
 - Python >= 3.6 
 - Spark 
 - `pyspark` package 

There are two ways to run Predictive Maintenance code: 
1) From notebook 
2) Via job submission 

In any case, we will need to create Azure ML Workspace where all Experiments, Runs, Logs, Models, Computes etc will be stored. 
Workspace can be created from Azure Dashboard, with use of Python SDK, using ML extension to Azure CLI or using Azure AI extention to Visual Studio Code. 

`PrepareWorkspace.ipynb` notebook creates new Azure ML Workspace within new resource group. It also writes workspace configuration, 
`aml_config\PredictiveMaintenanceWSConfig.json`, file which wil then be used for loading workspace object into Python code. 
This workspace will be used for both running the `Predictive Maintenance` notebooks from the `Code` folder as well as for 
submitting jobs to different Compute Targets. 


## Running Notebooks  
In this case, the notebook will be running locally or on remote server, for example, Azure Data Science Virtual Machine. 
Each notebook will be considered as separate experiment. Each run of a notebook will create new Run of corresponding Experiment.  
All needed information will be logged into workspace. 

To run Predictive Maintenance notebooks we need to have Python >= 3.6 as well as Spark installed 
with corresponding environmrnt variables. Jupyter kernel should be edited accordingly. For example, 
on Azure Data Science VM `SPARK 3 PYTHON LOCAL` kernel may be used. Corresponding kernel configuration file, 
`/usr/local/share/jupyter/kernels/spark-3-python`, should be edited to reference to Python 3.6 or newer. 


## Submitting jobs 
Job can be submitted to different Targets: 
 - Local machine (it is also possible to use Azure ML as a local machine) 
 - Remote server, for example, Azure VM 
 - Auto created AML compute which will be created for each Run 
 - Predefined cluster which can scale from 0 to predefined number of nodes 

 For each of this configurations corresponding Run Configuration file is created: 
 - `local.runconfig`  
 - `amlcompute.runconfig`  
 - `cluster.runconfig` 

Training model script is in `train.py` file. To run job on a Compute Target `SubmitRun.ipynb` notebook can be used. 
