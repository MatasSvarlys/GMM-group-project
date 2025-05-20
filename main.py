from fairchem.core.models.model_registry import available_pretrained_models
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.scripts import download_large_files
import json
from fairchem.core import FAIRChemCalculator

#show all available pretrained models

# print(available_pretrained_models)



#get the pretrained model

model_name = 'GemNet-OC-S2EFS-OC20+OC22'
checkpoint_path = model_name_to_local_file(model_name, local_cache='./fairchem_checkpoints/')
print(checkpoint_path)

calc = FAIRChemCalculator(checkpoint_path=checkpoint_path, device="cpu", task_name="oc20")

# read the data into an lmdb

from argparse import Namespace
from fairchem.scripts.dataset import preprocess_ef

args = Namespace(
    input_path="/data/oc22_uncompressed",
    output_path="/data/oc22_lmdb",
    task="s2ef",
    dataset="oc22",
    num_workers=8,
    val_split=None,  # Optional: add validation split
    train_size=None  # Optional: for subset
)

preprocess_ef.main(args)


# train the model

# evaluate the model

# show the results

