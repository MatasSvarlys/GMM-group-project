# SOME PACKAGES DONT WORK ON WINDOWS SO USE WSL

## download the fairchem-core with 
1. enable symlinks with (only if ur on windows)
`git config --global core.symlinks true`
2. enable windows developer mode (only if ur on windows)
3. run these commands:
```
git clone -c core.symlinks=true https://github.com/FAIR-Chem/fairchem.git
cd fairchem/packages/fairchem-core/
pip install .
```

in wsl you can just run `pip install fairchem-core` for the newer version

i did it with `pip install fairchem-core==1.10` cus the newest one has not implemented AtomsToGraphs yet

### note: you need to setup venv before pip installing on wsl
```
python3 -m venv .venv
source .venv/bin/activate
```


## if importing doesnt work due to faulty imports use

```
from fairchem.core.scripts
```

## download ase with 
`pip install ase`

# Windows (should not use)

## for unzipping 
`py ./fairchem/src/fairchem/core/scripts/uncompress.py --ipdir ./15/15/ --opdir ./data/oc22_uncompressed/ --num-workers 8`

--ipdir is where you downloaded the data

--opdir is where you will unzip the data to

## for making into a db 
`py ./fairchem/src/fairchem/core/scripts/preprocess_ef.py --data-path ./data/oc22_uncompressed/ --out-path ./data/oc22_lmdb/ --num-workers 8`




# What i did to make it download the dataset

1. download wsl with Ubuntu
2. copy the repository to the wsl
3. enable venv with 
```
python3 -m venv .venv
source .venv/bin/activate
```
4. ran `pip install fairchem-core==1.10`
5. changed the download_data.py file
```
from: import uncompress 
to: from fairchem.core.scripts import uncompress

from: import preprocess_ef as preprocess
to: from fairchem.core.scripts import preprocess_ef as preprocess
```
6. ran the main.py