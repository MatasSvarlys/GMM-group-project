import json
from fairchem.core.models.model_registry import model_name_to_local_file

model_name = 'GemNet-OC-S2EFS-OC20+OC22'


# get the pretrained model
from fairchem.core import OCPCalculator

calc = OCPCalculator(
    model_name=model_name,
    local_cache="fairchem_checkpoints",
    cpu=True,
    seed=50,
    trainer='forces'
)

# get the data into an lmdb

from fairchem.core.scripts.download_data import get_data
import os

if not os.path.exists('./data'):
    get_data(datadir='./data', task='s2ef', split='200k', del_intmd_files=True)

# train the model

# from fairchem.core.trainers import OCPTrainer
from omegaconf import OmegaConf

config_path = "./configs/oc20/s2ef/200k/gemnet/gemnet-oc.yml"


from fairchem.core.common.tutorial_utils import generate_yml_config
path_to_yaml = generate_yml_config(checkpoint_path="fairchem_checkpoints/gnoc_oc22_oc20_all_s2ef.pt")

from fairchem.core.common.utils import build_config

from argparse import Namespace

args = Namespace(
    config_yml=path_to_yaml,
    mode="train",
    identifier="gnoc_test",
    timestamp_id="12345",
    seed=50,
    debug=False,
    run_dir="./outputs",
    print_every=100,
    amp=True,
    checkpoint="./fairchem_checkpoints/gnoc_oc22_oc20_all_s2ef.pt",
    cpu=True,
    submit=False,
    summit=False,
    num_nodes=1,
    num_gpus=1,
    gp_gpus=1,
)


config = build_config(args=args, args_override=[
                                                "dataset.train.src=data/s2ef/200k/train/data.0000.lmdb",
                                                "dataset.src=data/s2ef/200k/train/data.0000.lmdb"
    ], include_paths=None)

cfg = OmegaConf.create(config)  # convert dict to OmegaConf object

# cfg = OmegaConf.load(path_to_yaml)


from fairchem.core._cli import runner_wrapper

# this will train and evaluate given the right config
runner_wrapper(cfg)

# only evaluate

# from fairchem.core.modules import Evaluator

# evaluator = Evaluator(task="s2ef")
# perf = evaluator.eval(prediction, target)
