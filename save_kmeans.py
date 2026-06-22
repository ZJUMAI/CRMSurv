import os

import csv
import time
import numpy as np

from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from utils.options import parse_args
from utils.util import get_split_loader, set_seed

from utils.loss import define_loss
from utils.optimizer import define_optimizer, define_optimizer1
from utils.scheduler import define_scheduler
import gc
import torch

def main(args):
    # set random seed for reproduction
    set_seed(args.seed)

    results_dir = "./save_cluster_{cluster}/{dataset}/[{model}]-[{fusion}]-[{alpha}]-[{time}]".format(
        dataset=args.dataset,
        model=args.model,
        fusion=args.fusion,
        alpha=args.alpha,
        time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        cluster=args.cluster
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    args.results_dir = results_dir
    # 5-fold cross validation
    header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
    best_epoch = ["best epoch"]
    best_score = ["best cindex"]

    # start 5-fold CV evaluation.
    for fold in range(1):
        # build dataset
        dataset = Generic_MIL_Survival_Dataset(
            csv_path=".Data/csv/tcga_blca_all_clean/%s_all_clean.csv.zip" % (args.dataset),
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=args.data_root_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
            cancer_type=args.cancer_type,
            # latent_model=args.latent_model
        )
        split_dir = os.path.join("./splits", args.dataset)
        train_dataset, val_dataset = dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}_filtered.csv".format(split_dir,6)
        )
        train_loader = get_split_loader(
            train_dataset,
            training=False,
            weighted=args.weighted_sample,
            modal=args.modal,
            batch_size=args.batch_size,
        )
        val_loader = get_split_loader(
            val_dataset, modal=args.modal, batch_size=args.batch_size
        )
        print(
            "training: {}, validation: {}".format(len(train_dataset), len(val_dataset))
        )

        # build model, criterion, optimizer, schedular
        if args.model == "crmsurv":
            from models.crmsurv.kmeans_save import Network
            from models.crmsurv.engine_save import Engine
            print(train_dataset.omic_sizes)
            model_dict = {
                "cluster":args.cluster,
                "seed":args.seed,
            }
            print("use_txt", args.use_txt)
            print("use_rl_loss",args.use_rl_loss)
            print("use_compression", args.use_compression)
            model = Network(**model_dict)#TransMIL(**model_dict)

            if args.use_rl_loss:
                criterion = define_loss(args, model.wsi_bank, model.omic_bank)
            else:
                criterion = define_loss(args, None, None)
            optimizer = None
            scheduler = None
            engine = Engine(args, results_dir, fold)
        else:
            raise NotImplementedError(
                "Model [{}] is not implemented".format(args.model)
            )
        # start training
        engine.learning(
            model, train_loader, val_loader, criterion, optimizer, scheduler
        )

        del model
        del optimizer
        del scheduler
        del criterion
        del engine
        del train_loader
        del val_loader
        del train_dataset
        del val_dataset
        del dataset

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")