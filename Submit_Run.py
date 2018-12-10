import time 
from azureml.core import ScriptRunConfig, RunConfiguration
from azureml.core import Workspace, Experiment 

ws = Workspace.from_config(path = './aml_config/PredictiveMaintenanceWSConfig.json') 

exp = Experiment(name = 'TrainModel', workspace = ws) 

#run_config = RunConfiguration.load(name = 'local', path = '.') 
run_config = RunConfiguration.load(name = 'amlcompute', path = '.') 
#run_config = RunConfiguration.load(name = 'cluster', path = '.') # `cluster` Compute Target should be created within Azure ML Workspace 

print(run_config) 

script_run_config = ScriptRunConfig(source_directory = '.', script = 'train.py', run_config = run_config) 

run = exp.submit(script_run_config) 

print(run.get_portal_url())

run.log('Starting Submission', time.asctime(time.localtime(time.time()))) 

run.wait_for_completion(show_output = True) 
