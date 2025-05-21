import json
# from fairchem.core import FAIRChemCalculator

#get the pretrained model

# model_name = 'GemNet-OC-S2EFS-OC20+OC22'
# checkpoint_path = model_name_to_local_file(model_name, local_cache='./fairchem_checkpoints/')
# print(checkpoint_path)

# calc = FAIRChemCalculator(checkpoint_path='./fairchem_checkpoints/gnoc_oc22_oc20_all_s2ef.pt', device="cpu", task_name="oc20")

# get the data into an lmdb

from fairchem.core.scripts.download_data import get_data
import os

if not os.path.exists('./data'):
    get_data(datadir='./data', task='s2ef', split='200k', del_intmd_files=True)


# train the model

# evaluate the model

# show the results

