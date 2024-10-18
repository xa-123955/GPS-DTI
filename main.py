comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='dgl.heterograph')
from models import GPSDTI
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloder import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="GPS-DTI for DTI prediction")
parser.add_argument('--cfg',  help="path to config file", type=str)
parser.add_argument('--data',  type=str, metavar='TASK', help='dataset')
parser.add_argument('--split', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster'])



args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    if not cfg.DA.TASK:
        train_path = os.path.join(dataFolder, 'train.csv')
        val_path = os.path.join(dataFolder, "valid.csv")
        test_path = os.path.join(dataFolder, "test.csv")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        train_dataset = DTIDataset(df_train.index.values, df_train)
        val_dataset = DTIDataset(df_val.index.values, df_val)
        test_dataset = DTIDataset(df_test.index.values, df_test)
    else:
        train_source_path = os.path.join(dataFolder, 'source_train.csv')
        train_target_path = os.path.join(dataFolder, 'target_train.csv')
        test_target_path = os.path.join(dataFolder, 'target_test.csv')
        df_train_source = pd.read_csv(train_source_path)
        df_train_target = pd.read_csv(train_target_path)
        df_test_target = pd.read_csv(test_target_path)

        train_dataset = DTIDataset(df_train_source.index.values, df_train_source)
        train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
        test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    if not cfg.DA.TASK:
        val_generator = DataLoader(val_dataset, **params)
        test_generator = DataLoader(test_dataset, **params)
    else:
        val_generator = DataLoader(train_target_dataset, **params)
        test_generator = DataLoader(test_target_dataset, **params)

    model = GPSDTI(**cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.LR,weight_decay=5e-5)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None,
                          discriminator=None,
                          experiment=experiment, **cfg)
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
