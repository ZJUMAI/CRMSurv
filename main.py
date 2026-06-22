import os

import csv
import time
import numpy as np

from datasets.dataset_survival1 import Generic_MIL_Survival_Dataset
from utils.options import parse_args
from utils.util import get_split_loader, set_seed

from utils.loss import define_loss
from utils.optimizer import define_optimizer, define_optimizer1
from utils.scheduler import define_scheduler
import gc
import torch


def main(args):
    set_seed(args.seed)

    # create results directory
    results_dir = "./CRMSurv/{dataset}/[{model}]-[{fusion}]-[{alpha}]-[{time}]".format(
        dataset=args.dataset,
        model=args.model,
        fusion=args.fusion,
        alpha=args.alpha,
        cluster=args.cluster,
        time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        beta=args.beta
    )

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    args.results_dir = results_dir

    # 5-fold cross validation
    header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
    best_epoch = ["best epoch"]
    best_score = ["best cindex"]

    # start 5-fold CV evaluation.
    for fold in range(5):
        # build dataset

        dataset = Generic_MIL_Survival_Dataset(
            csv_path="./Data/csv/tcga_blca_all_clean/%s_all_clean.csv" % (args.dataset),
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
            cluster=args.cluster
        )
        split_dir = os.path.join("./splits", args.dataset)
        train_dataset, val_dataset = dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}_filtered.csv".format(split_dir, fold)
        )

        train_loader = get_split_loader(
            train_dataset,
            training=True,
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

        # build model, criterion, optimizer, schedular CRMSurv
        if args.model == "crmsurv":
            from models.crmsurv.CRMSurv import Network
            from models.crmsurv.engine import Engine

            print(train_dataset.omic_sizes)
            model_dict = {
                "omic_sizes":train_dataset.omic_sizes,
                'input_dim': 512,
                "dim" : 256,
                "wsi_latent_dim" :512 ,
                "gene_latent_dim" :768 , 
                "emb_dim":  768,
                "wsi_queries": 24,
                "gene_queries" : 16,
                "depth" : 1,
                "num_classes" : 4,
                "cluster":args.cluster,
                "seed":args.seed,
                "use_txt": args.use_txt,
                "use_rl_loss":args.use_rl_loss,
                "use_compression":args.use_compression,
                "evaluate": args.evaluate
            }
            print("use_txt", args.use_txt)
            print("use_rl_loss",args.use_rl_loss)
            print("use_compression", args.use_compression)

            model = Network(**model_dict)
            model = model.cuda()

            if args.use_rl_loss:
                criterion = define_loss(args, model.wsi_bank, model.omic_bank)
            else:
                criterion = define_loss(args, None, None)

            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)
            engine = Engine(args, results_dir, fold)
        else:
            raise NotImplementedError(
                "Model [{}] is not implemented".format(args.model)
            )
        # start training
        score, epoch = engine.learning(
            model, train_loader, val_loader, criterion, optimizer, scheduler
        )
        # save best score and epoch for each fold
        best_epoch.append(epoch)
        best_score.append(score)

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


    # finish training
    # mean and std
    best_epoch.append("~")
    best_epoch.append("~")
    best_score.append(np.mean(best_score[1:6]))
    best_score.append(np.std(best_score[1:6]))

    csv_path = os.path.join(results_dir, "results.csv")
    print("############", csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerow(best_epoch)
        writer.writerow(best_score)


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")