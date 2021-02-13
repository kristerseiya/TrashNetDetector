
from azureml.core import Workspace
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--target_dir', type=str, required=True)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

ws = Workspace.from_config()
datastore = ws.get_default_datastore()
datastore.upload(src_dir=args.data_dir,
                 target_path=args.target_dir,
                 overwrite=args.overwrite)
