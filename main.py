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

from fairchem.core.datasets.oc22_lmdb_dataset import OC22LmdbDataset

dataset = OC22LmdbDataset(
    path='./data/oc22_lmdb',
    split='train',
    transform=None,
    key_mapping=None,
    lin_ref=None,
    oc20_ref=None,
    config=None,
    use_cache=False,
    cache_dir=None,
    cache_size=0,
    cache_mode="r",
    cache_type="lmdb",
    cache_path="./cache",
)


# train the model

from fairchem.core.trainers import OCPTrainer

trainer = OCPTrainer(
    
)



# evaluate the model

# show the results

