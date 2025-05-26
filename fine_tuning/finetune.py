import argparse, json, os, sys, time, multiprocessing
import glob

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator
from fairchem.core.common.tutorial_utils import train_test_val_split, generate_yml_config
from fairchem.core import OCPCalculator
import fairchem.core._cli as cli

# Paths & models
GRAPHS_DIR = "./data/graphs"
DB_PATH = "./db/oxides.db"
CONFIG_DIR = "./config"

# Change these to your model and generated model file
MODEL_NAME = "GemNet-OC-S2EFS-OC20+OC22"
MODEL_FILE = "gnoc_oc22_oc20_all_s2ef.pt"

MODEL_DIR = f"./model/is2re/{MODEL_FILE}"
TMP_MODEL_DIR = "./model/is2re"

# choose a checkpoint path (generated during training)
DEFAULT_CHECKPOINT = f"./outputs/checkpoints/2025-05-25-06-44-00-ft-oxides/best_checkpoint.pt"
SUPPORT_JSON = "./data/info/supporting-information.json"

MAX_EPOCHS = 3
NUM_WORKERS = 4  # Number of workers for data loading


def ensure_dirs():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(f"{GRAPHS_DIR}/{MODEL_NAME}", exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)


def load_support_data():
    with open(SUPPORT_JSON, 'r') as f:
        data = json.load(f)
    oxides = list(data.keys())
    polymorphs = list(data['TiO2'].keys())
    print(oxides)
    print(polymorphs)
    c = data['TiO2']['rutile']['PBE']['EOS']['calculations'][0]
    atoms = Atoms(symbols=c['atoms']['symbols'],
                  positions=c['atoms']['positions'],
                  cell=c['atoms']['cell'],
                  pbc=c['atoms']['pbc'])
    atoms.set_tags(np.ones(len(atoms)))
    print(atoms, c['data']['total_energy'], c['data']['forces'])
    return data, oxides, polymorphs


def compute_eos(calc, data, oxides, polymorphs):
    t0 = time.time()
    eos = {}
    for oxide in oxides:
        eos[oxide] = {}
        for polymorph in polymorphs:
            vols, dft, ocp = [], [], []
            calculations = data[oxide][polymorph]['PBE']['EOS']['calculations']
            for c in calculations:
                atoms = Atoms(symbols=c['atoms']['symbols'],
                              positions=c['atoms']['positions'],
                              cell=c['atoms']['cell'],
                              pbc=c['atoms']['pbc'])
                atoms.set_tags(np.ones(len(atoms)))
                atoms.calc = calc
                ocp += [atoms.get_potential_energy() / len(atoms)]
                dft += [c['data']['total_energy'] / len(atoms)]
                vols += [atoms.get_volume()]
            eos[oxide][polymorph] = (vols, dft, ocp)
    print(f"EoS compute time: {time.time() - t0:1.1f}s")
    return eos


def plot_general(eos, outfile, title):
    all_dft, all_ocp = [], []
    for ox in eos:
        for poly in eos[ox]:
            vols, dft, ocp = eos[ox][poly]
            plt.plot(dft, ocp, marker='s' if ox == 'VO2' else '.', alpha=0.5, label=f'{ox}-{poly}')
            all_dft += dft;
            all_ocp += ocp
    plt.xlabel('DFT (eV/atom)');
    plt.ylabel('OCP (eV/atom)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
    mae = np.mean(np.abs(np.array(all_dft) - np.array(all_ocp)))
    print(f"{title} MAE = {mae:1.3f} eV/atom")
    plt.savefig(outfile, dpi=300, bbox_inches='tight');
    plt.clf()


def plot_vo2(eos, oxide, polymorph, outfile, title):
    vols, dft, ocp = eos[oxide][polymorph]
    plt.plot(vols, dft, label='dft')
    plt.plot(vols, ocp, label='ocp')
    plt.title(f"{title} {oxide}-{polymorph}")
    plt.legend()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.clf()


def build_database(data, oxides, polymorphs):
    if os.path.exists(DB_PATH): os.remove(DB_PATH)
    for n in ['train.db', 'test.db', 'val.db']:
        if os.path.exists(n): os.remove(n)
    print(f"Creating database {DB_PATH}...")
    db = connect(DB_PATH)
    for oxide in oxides:
        for polymorph in polymorphs:
            for c in data[oxide][polymorph]['PBE']['EOS']['calculations']:
                atoms = Atoms(symbols=c['atoms']['symbols'],
                              positions=c['atoms']['positions'],
                              cell=c['atoms']['cell'],
                              pbc=c['atoms']['pbc'])
                atoms.set_tags(np.ones(len(atoms)))
                calc = SinglePointCalculator(atoms,
                                             energy=c['data']['total_energy'],
                                             forces=c['data']['forces'])
                atoms.set_calculator(calc)
                db.write(atoms)
    print(f"Database created with {len(db)} entries.")
    return train_test_val_split(DB_PATH)


def generate_config():
    yml = generate_yml_config(
        MODEL_DIR, f"{CONFIG_DIR}/config.yml",
        delete=['slurm', 'cmd', 'logger', 'task', 'model_attributes', 'dataset', 'test_dataset', 'val_dataset'],
        update={
            'gpus': 1, 'task.dataset': 'ase_db', 'optim.eval_every': 1, 'optim.max_epochs': MAX_EPOCHS,
            # training parameters
            'dataset.train.src': 'train.db', 'dataset.train.num_workers': NUM_WORKERS,
            'dataset.train.a2g_args.r_energy': True, 'dataset.train.a2g_args.r_forces': True,
            # test parameters
            'dataset.test.src': 'test.db', 'dataset.test.a2g_args.r_energy': False,
            'dataset.test.a2g_args.r_forces': False,
            # validation parameters
            'dataset.val.src': 'val.db', 'dataset.val.a2g_args.r_energy': True, 'dataset.val.a2g_args.r_forces': True
        }
    )
    print(yml)


def run_training(onCpu):
    sys.argv = [
                   "fairchem", "--mode", "train", "--config-yml", f"{CONFIG_DIR}/config.yml",
                   "--checkpoint", MODEL_DIR, "--run-dir", "outputs", "--identifier", "ft-oxides",
                   "--seed", "50", "--print-every", "100"
               ] + (["--cpu"] if onCpu else ["--num-gpus", "1"])
    print("Running OCP training...")
    cli.main()
    print("Training finished.")


def run_evaluation(ckpt, onCpu, data, oxides, polymorphs):
    print(f"Loading model from {ckpt}...")
    calc = OCPCalculator(
        local_cache=TMP_MODEL_DIR,  # dir to save checkpoints
        checkpoint_path=ckpt,  # path to the checkpoint file
        trainer='forces', cpu=onCpu, seed=50)
    eos = compute_eos(calc, data, oxides, polymorphs)
    plot_general(eos, f"{GRAPHS_DIR}/{MODEL_NAME}/trained_eos.png", "fine-tuned")
    plot_vo2(eos, 'VO2', 'fluorite', f"{GRAPHS_DIR}/{MODEL_NAME}/trained_eos_VO2_fluorite.png", "fine-tuned")


def main():
    train_from_scratch = True
    onCpu = True

    ensure_dirs()
    data, oxides, polymorphs = load_support_data()

    # pretrained
    pre_calc = OCPCalculator(
        model_name=MODEL_NAME,  # download the model if not present
        local_cache=TMP_MODEL_DIR,
        trainer='forces', cpu=onCpu, seed=50)
    eos_pre = compute_eos(pre_calc, data, oxides, polymorphs)
    plot_general(eos_pre, f"{GRAPHS_DIR}/{MODEL_NAME}/pretrained_eos.png", "pretrained")

    plot_vo2(eos_pre, 'VO2', 'fluorite', f"{GRAPHS_DIR}/{MODEL_NAME}/pretrained_eos_VO2_fluorite.png", "pretrained")

    # DB + config
    build_database(data, oxides, polymorphs)
    print("train, test, val split finished.")
    generate_config()

    # train vs eval
    if train_from_scratch:
        run_training(onCpu=onCpu)

        # load latest trained checkpoint from outputs directory
        run_dirs = glob.glob("outputs/checkpoints/*")
        latest_run = max(run_dirs, key=os.path.getmtime)
        newckpt = os.path.join(latest_run, "checkpoint.pt")
        print("Loading checkpoint from", newckpt)

        run_evaluation(newckpt, onCpu, data, oxides, polymorphs)
    else:
        print("Skipping training, running evaluation only.")
        run_evaluation(DEFAULT_CHECKPOINT, onCpu, data, oxides, polymorphs)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
