import numpy as np
np.Inf = np.inf
from executer_TP import Execute_TP
import pytorch_lightning as pl
import argparse
import os
import copy
try:
    import optuna
except ModuleNotFoundError:
    optuna = None

from pytorch_lightning import Trainer
#seed_everything(42, workers=True)

current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir,"data_TP")

def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())

    #my editing
    # --- Model ablation flags (can be overridden by presets below) ---
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed for reproducibility")
    parser.add_argument("--do_ablation", type=lambda s: str(s).lower() in ["1", "true", "yes"], default=False,
                        help="Run a small ablation grid instead of a single training run")

    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "huber"],
                        help="Loss for normalized endpoints: l1 or huber")
    parser.add_argument("--huber_beta", type=float, default=0.5,
                        help="Huber beta (only used if loss_type=huber)")
    parser.add_argument("--use_interaction", type=lambda s: str(s).lower() in ["1", "true", "yes"], default=False,
                        help="If true, add |h - t| to inputs")
    # --- Extra knobs ---
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Optimizer learning rate")
    parser.add_argument("--end_weight", type=float, default=1.0,
                        help="Extra weight on end-year loss term")
    parser.add_argument("--extra_order_pen", type=float, default=0.0,
                        help="Penalty for predicted end < start (applied in normalized space)")
    parser.add_argument("--use_prod", type=lambda s: str(s).lower() in ["1", "true", "yes"], default=False,
                        help="If true, add elementwise h*t interaction to features")
    # embedding noise (tiny Gaussian noise on E/R/T embeddings during training)
    parser.add_argument("--emb_noise", type=float, default=0.0,
                        help="Stddev of Gaussian noise added to entity/relation embeddings during training (0.0 disables).")

    # --- Optuna (Bayesian) search flags ---
    parser.add_argument("--optuna_trials", type=int, default=0,
                        help="Number of Optuna trials; 0 disables tuning")
    parser.add_argument("--optuna_timeout", type=int, default=0,
                        help="Global timeout in seconds for Optuna (0 = no timeout)")
    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Hidden width of the MLP trunk (e.g., 512, 1024).")
    parser.add_argument("--dropout", type=float, default=0.10,
                        help="Dropout prob used in the trunk.")
    parser.add_argument("--gauss_sigma_idx", type=float, default=1.25,
                        help="Gaussian width (in index units) for CE targets.")
    parser.add_argument("--t_dim", type=int, default=None)
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--k_experts", type=int, default=None)
    parser.add_argument("--gate_temp_start", type=float, default=None)
    parser.add_argument("--gate_balance", type=float, default=None)
    parser.add_argument("--run_optuna", type=int, default=0,
                        help="Set to 1 to run Optuna sweep instead of normal training.")
    # === bands / duration-prior flags ===
    parser.add_argument("--use_bands", type=int, default=0, help="Enable per-relation year bands (1/0)")
    parser.add_argument("--use_prior", type=int, default=0, help="Enable relation-wise duration prior (1/0)")
    parser.add_argument("--band_margin", type=float, default=1.5, help="Observed band margin in index units")
    parser.add_argument("--prior_weight", type=float, default=0.0, help="Loss weight for duration prior")

    # === Ablation loss weights ===
    parser.add_argument("--w_ce", type=float, default=0.08,
                        help="Weight for Gaussian calendar CE loss.")
    parser.add_argument("--w_iou", type=float, default=0.20,
                        help="Weight for soft-IoU interval objective.")
    parser.add_argument("--w_consistency", type=float, default=0.01,
                        help="Weight for consistency loss between regression and calendar heads.")
    parser.add_argument("--w_end_boost", type=float, default=0.02,
                        help="Weight for auxiliary end-before-start calendar penalty.")
    parser.add_argument("--w_order", type=float, default=0.01,
                        help="Weight for auxiliary order penalty.")
    parser.add_argument("--w_align", type=float, default=0.01,
                        help="Weight for q_s/q_e alignment regularizer.")
    parser.add_argument("--w_smooth", type=float, default=0.010,
                        help="Weight for calendar smoothness regularizer.")
    parser.add_argument("--w_ridge", type=float, default=0.0003,
                        help="Weight for calendar L2/ridge regularizer.")
    parser.add_argument("--w_reg", type=float, default=1.0,
                        help="Weight for regression head loss.")

    parser.add_argument("--pred_head", type=str, default="calendar",
                        choices=["calendar", "regression"],
                        help="Which head to use for final evaluation: calendar or regression.")

    #parser.add_argument("--path_train_dataset", type=str, required=True, help="data_TP/dbpedia124k/")
    # Paths.
    parser.add_argument("--path_dataset_folder", type=str, default='data_TP/') #The folder path where your dataset is located

    parser.add_argument("--storage_path", type=str, default='HYBRID_Storage') #Location for storing model outputs.
    parser.add_argument("--eval_dataset", type=str, default='Dbpedia124k',
                        help="Available datasets: Dbpedia124k, Yago3K") #Specifies the dataset to use
    # FactBench, BPDP,Dbpedia34k,
    #TODO: To be added later for factbench dataset in particular
    parser.add_argument("--sub_dataset_path", type=str, default=None,
                        help="TODO: Available subpaths: bpdp/, domain/, domainrange/, mix/, property/, random/, range/,")

    #TODO: To be added later for factbench dataset in particular
    parser.add_argument("--prop", type=str, default=None,
                        help="TODO: Available properties (only for FactBench dataset if available): architect, artist, author, commander, director, musicComposer, producer, None")

    parser.add_argument("--negative_triple_generation", type=str, default="False",
                        help="Available approaches: corrupted-triple-based, corrupted-time-based, False")

    parser.add_argument("--complete_dataset", type=bool, default=True)
    parser.add_argument("--include_veracity", type=bool, default=True)

    # parser.add_argument("--auto_scale_batch_size", type=bool, default=True)
    # parser.add_argument("--deserialize_flag", type=str, default=None, help='Path of a folder for deserialization.')

    # Models. select temporal model for time point prediction!!
    parser.add_argument("--model", type=str, default='temporal-prediction-model',
                        help="Available models:temporal-prediction-model, temporal-full-hybrid") #Defines which model to use


    parser.add_argument("--task", type=str, default='time-prediction',
                        help="Available datasets:   time-prediction, fact-checking") #Specifies the task
                        # help="Available models:Hybrid, ConEx, TransE, Hybrid, ComplEx, RDF2Vec")

    parser.add_argument("--emb_type", type=str, default='dihedron',
                        help="Available TKG embeddings: dihedron, None")

    # Hyperparameters pertaining to number of parameters.
    parser.add_argument('--embedding_dim', type=int, default=100) #Hyperparameters. define the training setup
    parser.add_argument('--valid_ratio', type=int, default=20)
    parser.add_argument('--sentence_dim', type=int, default=768)
    parser.add_argument("--max_num_epochs", type=int, default=50) #Hyperparameters. define the training setup
    parser.add_argument("--min_num_epochs", type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512) #Hyperparameters. define the training setup
    parser.add_argument('--val_batch_size', type=int, default=1000)
    # parser.add_argument('--negative_sample_ratio', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1, help='Number of cpus used during batching')
    parser.add_argument("--check_val_every_n_epochs", type=int, default=1)
    # parser.add_argument('--enable_checkpointing', type=bool, default=True)
    # parser.add_argument('--deterministic', type=bool, default=True)
    # parser.add_argument('--fast_dev_run', type=bool, default=False)
    # parser.add_argument("--accumulate_grad_batches", type=int, default=3)
    # PREPROCESS DATASETS
    parser.add_argument("--preprocess", type=str, default='False',
                        help="Available options: False, Concat, SentEmb, TrainTestTriplesCreate") #data preprocessing operations -> Concat for concatenating embeddings or SentEmb for sentence embeddings

    parser.add_argument("--ids_only", type=str, default=False)
    parser.add_argument("--checkpoint_dir_folder", type=str, default='2024-08-01 15:32:27.650994', choices=["all","YYYY-MM-DD HH:MM:SS.XXXXXX"], help="check hybrid storage folder")
    parser.add_argument(
        "--checkpoint_dataset_folder", default="dataset/", choices=["dataset/"], help="folder in which all resultant models are stored"
    ) #arguments define where the model checkpoints will be saved during training. The model's performance will also be evaluated periodically.

    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)

import copy

def run_optuna(args):
    try:
        import optuna
    except ModuleNotFoundError:
        raise RuntimeError("Install Optuna: pip install 'optuna>=3,<4'")

    def objective(trial):
        targs = copy.deepcopy(args)

        # ========= NEW / EXPANDED SUGGESTIONS =========

        # model capacity that your select_model already passes:
        targs.hidden_dim = trial.suggest_categorical("hidden_dim", [512, 1024])
        targs.dropout = trial.suggest_float("dropout", 0.10, 0.18)

        # === MoE + calendar knobs ===
        targs.t_dim = int(trial.suggest_categorical("t_dim", [64, 96, 128]))
        targs.num_experts = int(trial.suggest_categorical("num_experts", [3, 4, 5]))
        targs.k_experts = int(trial.suggest_categorical("k_experts", [1, 2, 3]))
        if targs.k_experts > targs.num_experts:
            targs.k_experts = targs.num_experts  # safety

        targs.gate_temp_start = float(trial.suggest_float("gate_temp_start", 1.2, 2.0))
        targs.gate_balance = float(trial.suggest_float("gate_balance", 0.01, 0.05))

        # already used by your training_step (scalar CE sigma):
        targs.gauss_sigma_idx = trial.suggest_float("gauss_sigma_idx", 1.10, 1.60)

        # regularizers you already wire:
        targs.emb_noise = trial.suggest_float("emb_noise", 0.00, 0.015)

        # (Optional toggles, default OFF in your best recipe; let Optuna test)
        targs.use_bands = trial.suggest_categorical("use_bands", [0, 1])
        targs.use_prior = trial.suggest_categorical("use_prior", [0, 1])

        # only meaningful if bands/prior are ON; still safe to set:
        targs.band_margin = trial.suggest_float("band_margin", 0.8, 2.0)
        targs.prior_weight = trial.suggest_float("prior_weight", 0.02, 0.06)

        # --- sampled hyperparams ---
        targs.lr = trial.suggest_float("lr", 7e-4, 2e-3, log=True)
        targs.loss_type = "huber"
        targs.huber_beta = trial.suggest_float("huber_beta", 0.60, 1.00)
        targs.end_weight = trial.suggest_float("end_weight", 0.85, 1.4)
        targs.extra_order_pen = trial.suggest_float("extra_order_pen", 0.02, 0.12)
        targs.use_interaction = trial.suggest_categorical("use_interaction", [0, 1])
        targs.use_prod        = trial.suggest_categorical("use_prod", [0, 1])

        # --- keep trials short (set BEFORE creating Execute_TP) ---
        targs.max_num_epochs = min(getattr(args, "max_num_epochs", 100), 60)
        targs.min_num_epochs = getattr(args, "min_num_epochs", 1)
        # optional: make sure we validate every epoch during tuning
        targs.check_val_every_n_epochs = 1

        # train once and return best val
        ex = Execute_TP(targs)
        _, best_val = ex.fit_return_best()
        return best_val

    # optional: make results reproducible
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=args.optuna_trials,
        timeout=args.optuna_timeout if args.optuna_timeout > 0 else None,
        show_progress_bar=True,
    )
    print("Optuna best params:", study.best_trial.params)
    print("Optuna best val_loss:", study.best_value)
    return study

def run_ablation_grid(base_args):
    """
    Run a tiny matrix of configs to validate that certain switches help.
    Results will go to separate run folders and into results.csv (your code already writes it).
    """
    from copy import deepcopy
    from executer_TP import Execute_TP

    # Keep runs short but comparable
    base_epochs = getattr(base_args, "max_num_epochs", 100)
    short_epochs = min(base_epochs, 30)

    grid = [
        # (A) Your tuned "best" config (reference)
        dict(name="best", use_prod=True,  extra_order_pen=base_args.extra_order_pen,
             end_weight=base_args.end_weight, loss_type=base_args.loss_type,
             huber_beta=base_args.huber_beta, lr=base_args.lr),

        # (B) Turn off use_prod only
        dict(name="no_prod", use_prod=False),

        # (C) Remove order penalty only
        dict(name="no_order_pen", extra_order_pen=0.0),

        # (D) L1 vs Huber check (only if your best uses huber)
        dict(name="l1_check", loss_type="l1"),
    ]

    for spec in grid:
        a = deepcopy(base_args)
        # apply overrides
        for k,v in spec.items():
            if k == "name":
                continue
            setattr(a, k, v)

        # keep runs short & deterministic
        a.max_num_epochs = short_epochs
        a.min_num_epochs = getattr(a, "min_num_epochs", 1)
        a.check_val_every_n_epochs = 1
        # give each run its own folder suffix
        a.storage_path = f"{base_args.storage_path}/{spec['name']}"

        print(f"\n[ABLATION] Running {spec['name']} with overrides: "
              f"{ {k:v for k,v in spec.items() if k!='name'} }")
        ex = Execute_TP(a)
        ex.start()


if __name__ == "__main__":
    args = argparse_default()

    if args.optuna_trials and args.optuna_trials > 0:
        run_optuna(args)
    elif getattr(args, "do_ablation", False):
        run_ablation_grid(args)
    else:
        ex = Execute_TP(args)
        ex.start()




#if __name__ == '__main__':
 #   args = argparse_default()
  #  print("Parsed Arguments:", args)  # my editing for checking where my code gets killed?

    # ===== A/B/C/D PRESETS: uncomment ONE block you want to run =====

    # --- A: Baseline (L1, no |h-t|) ---
    #args.loss_type = "l1"
    #args.use_interaction = False
    # args.huber_beta = 0.5  # (ignored for L1)

    # --- B: L1 + |h-t| ---
   # args.loss_type = "l1"
    #args.use_interaction = True
   # args.use_prod = False  # turn on h ⊙ r
    #args.end_weight = 1.0  # asymmetric: weight end more
    #args.extra_order_pen = 0.0  # small extra order penalty

    # --- C: Huber, no |h-t| ---
    #args.loss_type = "huber"
    #args.huber_beta = 0.5
    #args.use_interaction = False

    # --- D: Huber + |h-t| ---
    #args.loss_type = "huber"
    #args.huber_beta = 0.5
    #args.use_interaction = True
    # ================================================================

  #  exc = Execute_TP(args)
   # exc.start()



    # Yago al
    # if args.eval_dataset == "Yago3K":
    #     args.ids_only = True

    # Preprocess dataset if flag is True!
    # if args.preprocess != 'False':
    #     if args.preprocess == 'Concat':
    #         if args.eval_dataset == "Dbpedia124k":
    #             DBpedia34kDataset = True
    #             # ConcatEmbeddings(args, path_dataset_folder=args.path_dataset_folder,DBpedia34k=DBpedia34kDataset)
    #         print("concat done")
    #     elif args.preprocess == 'SentEmb':
    #         print("sentence vectors creation is done")
    #     elif args.preprocess == 'TrainTestTriplesCreation':
    #         print("Train and Test triples creation is complete")

    # if args.eval_dataset == "Dbpedia124k" or args.eval_dataset == "Yago3K":
    # else:
    #     print("Please specify a valid dataset")
    #     exit(1)
