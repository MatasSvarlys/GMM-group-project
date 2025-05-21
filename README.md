## download the fairchem-core with 
1. enable symlinks with
`git config --global core.symlinks true`
2. enable windows developer mode
3. run these commands:
```
git clone -c core.symlinks=true https://github.com/FAIR-Chem/fairchem.git
cd fairchem/packages/fairchem-core/
pip install .
```

## download ase with 
`pip install ase`


## for unzipping 
`py ./fairchem/src/fairchem/core/scripts/uncompress.py --ipdir ./15/15/ --opdir ./data/oc22_uncompressed/ --num-workers 8`

--ipdir is where you downloaded the data

--opdir is where you will unzip the data to

## for making into a db 
`py ./fairchem/src/fairchem/core/scripts/preprocess_ef.py --data-path ./data/oc22_uncompressed/ --out-path ./data/oc22_lmdb/ --num-workers 8`