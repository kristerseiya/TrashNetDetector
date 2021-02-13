
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='trashnet-ssd')

parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--output_path', type=str, default='outputs/detector.pth')
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--batch', type=int, required=True)

args = parser.parse_args()

ws = Workspace.from_config()
datastore = ws.get_default_datastore()
dataset = Dataset.File.from_files(path=(datastore, args.data_dir))

experiment = Experiment(workspace=ws, name=args.id)

config = ScriptRunConfig(source_directory='.',
                         script='train.py',
                         compute_target='gpu-cluster',
                         arguments=['--data_dir', dataset.as_named_input('input').as_mount(),
                                    '--export', args.output_path,
                                    '--n_epoch', args.epoch,
                                    '--batch_size', args.batch])

env = Environment.from_conda_specification(
    name='pytorch-env',
    file_path='.azureml/pytorch_env.yml'
)
config.run_config.environment = env

run = experiment.submit(config)

aml_url = run.get_portal_url()
print(aml_url)
