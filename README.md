download the fairchem-core with 
```
pip install git+https://github.com/facebookresearch/fairchem.git@fairchem_core-2.0.0#subdirectory=packages/fairchem-core
```

download ase with `pip install ase`


for unzipping `py ./fairchem/src/fairchem/core/scripts/uncompress.py --ipdir ./15/15/ --opdir ./data/oc22_uncompressed/ --num-workers 8`

--ipdir is where you downloaded the data
--opdir is where you will unzip the data to

for making into a db `python ./fairchem/src/fairchem/core/scripts/preprocess_ef.py --data-path ./data/oc22_uncompressed/ --out-path ./data/oc22_lmdb/ --num-workers 8`