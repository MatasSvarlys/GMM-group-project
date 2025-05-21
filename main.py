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

# calc = FAIRChemCalculator(checkpoint_path='./fairchem_checkpoints/gnoc_oc22_oc20_all_s2ef.pt', device="cpu", task_name="oc20")

# get the data into an lmdb

from fairchem.core.scripts.download_data import get_data
 
get_data(datadir='./data', task='s2ef', split='200k', del_intmd_files=True)


# train the model

# evaluate the model

# show the results

