import argparse


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Configurations for Survival Analysis on TCGA Data.")
    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument(
        "--data_root_dir", type=str, default="path/to/data_root_dir", help="Data directory to WSI features (extracted via CLAM"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible experiment (default: 1)")
    parser.add_argument(
        "--which_splits", type=str, default="5foldcv", help="Which splits folder to use in ./splits/ (Default: ./splits/5foldcv"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tcga_blca",#"tcga_blca_100",
        help='Which cancer type within ./splits/<which_dataset> to use for training. Used synonymously for "task" (Default: tcga_blca_100)',
    )
    parser.add_argument(
        "--cancer_type",
        type=str,
        default="blca",
    )
    parser.add_argument("--log_data", action="store_true", default=True, help="Log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", default=False, dest="evaluate", help="Evaluate model on test set")
    parser.add_argument("--resume", type=str, default="", metavar="PATH", help="Path to latest checkpoint (default: none)")

    parser.add_argument("--OOM", type=int, default=0, help="Ramdomly sampling some patches to avoid OOM error")

    # Model Parameters.
    parser.add_argument(
        "--model",
        type=str,
        default="crmsurv",
        help="Type of model (Default: mcat)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=[
            "small",
            "large",
        ],
        default="small",
        help="Size of some models (Transformer)",
    )
    parser.add_argument(
        "--modal",
        type=str,
        choices=["omic", "path", "pathomic", "cluster", "coattn", "triattn"],
        default="triattn",
        help="Specifies which modalities to use / collate function in dataloader.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["concat", "bilinear"],
        default="concat",
        help="Modality fuison strategy",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="hyper-parameter of loss function")
    parser.add_argument("--beta", type=float, default=0.1, help="hyper-parameter of loss function")
     parser.add_argument(
        "--save_cluster_path",
        type=str,
        default="./Data",
        help="Specifies which modalities to use / collate function in dataloader.",
    )

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--num_epoch", type=int, default=30, help="Maximum number of epochs to train (default: 30)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size (Default: 1, due to varying bag sizes)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--loss",
        type=str,
        default="nll_surv",
        help="slide-level classification loss function (default: ce)",
    )
    parser.add_argument("--weighted_sample", action="store_true", default=True, help="Enable weighted sampling")
    parser.add_argument("--use_rl_loss", action="store_true", help="Enable rl loss")
    parser.add_argument("--use_txt", action="store_true", help="Enable txt")
    parser.add_argument("--use_compression", action="store_true", help="Enable compression")
    parser.add_argument("--cluster", type=int, default=8, help="numbers of cluster centers")
    parser.set_defaults(use_rl_loss=False)
    parser.set_defaults(use_txt=False)
    parser.set_defaults(use_compression=False)

    args = parser.parse_args()
    return args
