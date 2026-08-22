import warnings

import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, classification_report, precision_score, auc
import nn_models_TP
from utils_TP.dataset_classes import StandardDataModule
from data_TP import Data
from utils_TP.static_funcs import *
import time
import torch
from sklearn.model_selection import KFold
# from pytorch_lightning_kfold.validation import KFoldCrossValidator

import json
import pytorch_lightning as pl
#from pytorch_lightning.plugins import DDPPlugin,DataParallelPlugin
# from utils.dataset_classes import StandardDataModule
from pytorch_lightning import Trainer
#seed_everything(42, workers=True)


class Execute_TP:
    def __init__(self, args):
        args = preprocesses_input_args(args)
        sanity_checking_with_arguments(args)
        self.args = args
        from pytorch_lightning import seed_everything
        seed_everything(getattr(args, "seed", 42), workers=True)
        # 1. Create an instance of KG.
        self.args.dataset = Data(args=args)

        # === QUICK SPLIT SANITY ===
        def _split_stats(name, idx_list, num_times):
            if not idx_list:
                print(f"[SANITY] {name}: EMPTY")
                return
            import numpy as np
            X = np.asarray(idx_list, dtype=np.int64)
            if X.ndim != 2 or X.shape[1] < 5:
                print(f"[SANITY] {name}: unexpected shape {X.shape} (want Nx5)")
                return
            y1 = X[:, 3]; y2 = X[:, 4]
            n = X.shape[0]
            bad_lo = (y1 < 0).sum() + (y2 < 0).sum()
            bad_hi = (y1 >= num_times).sum() + (y2 >= num_times).sum()
            bad_ord = (y1 > y2).sum()
            print(f"[SANITY] {name}: n={n}  "
                  f"y1[min,max]=({y1.min()},{y1.max()})  "
                  f"y2[min,max]=({y2.min()},{y2.max()})  "
                  f"out_of_range={bad_lo+bad_hi}  "
                  f"violations(y1>y2)={bad_ord}")

        _split_stats("train", self.args.dataset.idx_train_set, self.args.dataset.num_times)
        _split_stats("valid", self.args.dataset.idx_valid_set, self.args.dataset.num_times)
        _split_stats("test",  self.args.dataset.idx_test_set,  self.args.dataset.num_times)


        print(f"[DEBUG] idx_time_dict keys: {list(self.args.dataset.idx_time_dict.keys())[:10]}")  # Print first 10 keys
        self.args.idx_time_dict = self.args.dataset.idx_time_dict  # Add this line to store the dictionary

        # 2. Create a storage path  + Serialize dataset object.
        self.storage_path = create_experiment_folder(folder_name=args.storage_path)
        # self.eval_model = True if self.args.eval == 1 else False

        # 3. Save Create folder to serialize data. This two numerical value will be used in embedding initialization.
        self.args.num_entities, self.args.num_relations, self.args.num_times = self.args.dataset.num_entities, self.args.dataset.num_relations, self.args.dataset.num_times


        # 4. Create logger
        self.logger = create_logger(name=self.args.model, p=self.storage_path)

        # 5. KGE related parameters


    def store(self, trained_model) -> None:
        """
        Store trained_model model and save embeddings into csv file.
        :param trained_model:
        :return:
        """
        self.logger.info('Store full model.')
        # Save Torch model.
        torch.save(trained_model.state_dict(), self.storage_path + '/model.pt')
        self.args.dataset = ""
        with open(self.storage_path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(self.args)
            temp.pop('gpus')
            temp.pop('tpu_cores')
            json.dump(temp, file_descriptor)

        self.logger.info('Stored data.')


    def start(self) -> None:
        """
        Train and/or Evaluate Model
        Store Mode
        """
        print("Training Started...")    # my editing for checking where my code gets killed?
        start_time = time.time()
        # 1. Train and Evaluate
        trained_model = self.train_and_eval()
        print("Training and evaluation Complete.")      # my editing for checking where my code gets killed?
        # 2. Store trained model
        self.store(trained_model)
        #
        total_runtime = time.time() - start_time

        print(f"Total Runtime: {total_runtime} seconds")        # my editing for checking where my code gets killed?

        if 60 * 60 > total_runtime:
            message = f'{total_runtime / 60:.3f} minutes'
        else:
            message = f'{total_runtime / (60 ** 2):.3f} hours'

        self.logger.info(f'Total Runtime:{message}')

    def train_and_eval(self) -> nn_models_TP.BaseKGE:
        """
        Training and evaluation procedure
        """
        print("Training and evaluation Started...")         # my editing for checking where my code gets killed?

        self.logger.info('--- Parameters are parsed for training ---')


        # trainer = pl.Trainer.from_argparse_args(Namespace(**dict(train_config)), early_stop_callback=early_stop_callback)

        # 3. Init ModelCheckpoint callback, monitoring 'val_loss'
        # self.args.enable_checkpointing = True
        self.args.checkpoint_callback = True
        if self.args.sub_dataset_path==None:
            pth = self.args.eval_dataset
        else:
            pth = self.args.eval_dataset + "-" + self.args.sub_dataset_path.replace('/','')
        mdl = self.args.model

        # saves a checkpint model file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.storage_path,
            filename="sample-{"+(str(mdl).lower())+"}--{"+(str(pth).lower())+"}--"+(str(self.args.emb_type).lower())+"--"+(str(self.args.negative_triple_generation).lower())+"--{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
            mode="min",
        )
        # 1. Create Pytorch-lightning Trainer object from input configuration
        # print(torch.cuda.device_count())
        # new: explicit single-process trainer (no dp/ddp_spawn)
        # --- single-process trainer, compatible with old & new PL ---
        # compute robust epoch args
        max_epochs = getattr(self.args, "max_num_epochs", getattr(self.args, "max_epochs", 80))
        pat = max(15, int(0.4 * max_epochs))  # e.g., 40% of max epochs
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=pat, mode="min")
        min_epochs = getattr(self.args, "min_num_epochs", getattr(self.args, "min_epochs", 1))

        try:
            # newer PL
            self.trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=max_epochs,
                min_epochs=min_epochs,
                callbacks=[early_stopping_callback, checkpoint],
                enable_checkpointing=True,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
                logger=self.args.logger,
                num_sanity_val_steps=0,
                benchmark=False,
                deterministic=False,
            )
        except TypeError:
            # PL ≤ 1.5 fallback
            self.trainer = pl.Trainer(
                gpus=1 if torch.cuda.is_available() else None,
                max_epochs=max_epochs,
                min_epochs=min_epochs,
                callbacks=[early_stopping_callback, checkpoint],
                checkpoint_callback=True,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
                logger=self.args.logger,
                num_sanity_val_steps=0,
                benchmark=False,
                deterministic=False,
            )

        # 2. Check whether validation and test datasets are available.
        #if self.args.dataset.is_valid_test_available():
        trained_model = self.training()
        print("Training fininshed")         # my editing for checking where my code gets killed?
        self.logger.info('--- Training is completed  ---')

        # print(self.args.checkpoint_callback.best_model_path)

        return trained_model

    def fit_return_best(self):
        """
        Lightweight training for hyperparameter search.
        Trains and returns (model, best_val_softIoU) without running test/eval/store.
        """
        # 1) Build model + loaders (unchanged)
        model, form_of_labelling = select_model(self.args)
        model.gauss_sigma_idx = float(self.args.gauss_sigma_idx)

        if not self.args.batch_size:
            self.args.batch_size = int(len(self.args.dataset.idx_train_set) / 3) + 1
        if not self.args.val_batch_size:
            self.args.val_batch_size = int(len(self.args.dataset.idx_valid_set) / 2) + 1

        dataset = StandardDataModule(
            train_set_idx=self.args.dataset.idx_train_set,
            valid_set_idx=self.args.dataset.idx_valid_set,
            test_set_idx=self.args.dataset.idx_test_set,
            entities_count=self.args.dataset.num_entities,
            relations_count=self.args.dataset.num_relations,
            times_count=self.args.dataset.num_times,
            form=form_of_labelling,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
        train_loader = dataset.train_dataloader(batch_size1=self.args.batch_size)
        val_loader = dataset.val_dataloader(batch_size1=self.args.val_batch_size)

        # 2) Optuna-friendly callbacks: checkpoint on IoU (maximize), early-stop on val_loss (minimize)
        ckpt_cb_iou = ModelCheckpoint(
            monitor="val_softIoU", mode="max", save_top_k=1,
            filename="optuna-iou-{epoch:02d}-{val_softIoU:.4f}"
        )
        early_cb = EarlyStopping(monitor="val_loss", mode="min", patience=3)

        # 3) Trainer (keeps your compatibility branch)
        max_epochs = getattr(self.args, "max_num_epochs", getattr(self.args, "max_epochs", 100))
        min_epochs = getattr(self.args, "min_num_epochs", getattr(self.args, "min_epochs", 1))
        check_val_every_n_epoch = getattr(
            self.args, "check_val_every_n_epoch",
            getattr(self.args, "check_val_every_n_epochs", 1)
        )

        try:
            tuner_trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=max_epochs,
                min_epochs=min_epochs,
                check_val_every_n_epoch=check_val_every_n_epoch,
                enable_checkpointing=True,
                logger=False,
                enable_model_summary=False,
                callbacks=[ckpt_cb_iou, early_cb],
                num_sanity_val_steps=0,
                deterministic=False,
            )
        except TypeError:
            tuner_trainer = Trainer(
                gpus=1 if torch.cuda.is_available() else None,
                max_epochs=max_epochs,
                min_epochs=min_epochs,
                check_val_every_n_epoch=check_val_every_n_epoch,
                callbacks=[ckpt_cb_iou, early_cb],
                checkpoint_callback=True,
                logger=False,
                num_sanity_val_steps=0,
                deterministic=False,
            )

        # 4) Fit once
        tuner_trainer.fit(model, train_loader, val_loader)

        # 5) Return the **best** IoU seen during training
        if ckpt_cb_iou.best_model_score is not None:
            best_iou = float(ckpt_cb_iou.best_model_score.detach().cpu().item())
        else:
            # fallback: last epoch's IoU if checkpoint didn’t trigger
            m = tuner_trainer.callback_metrics.get("val_softIoU", None)
            best_iou = float(m.detach().cpu().item()) if m is not None else 0.0

        return model, best_iou

    def run_optuna(self):
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner

        # Use CLI limits if provided
        n_trials = int(getattr(self.args, "optuna_trials", 30) or 30)
        timeout_s = int(getattr(self.args, "optuna_timeout", 0) or 0)  # 0 = no timeout

        def objective(trial: optuna.Trial):
            # ---- Suggest ONLY knobs your CLI / model already supports safely ----
            # (Keeps this simple; we can extend later.)
            self.args.hidden_dim = trial.suggest_categorical("hidden_dim", [512, 1024])
            self.args.dropout = trial.suggest_float("dropout", 0.10, 0.18)
            self.args.emb_noise = trial.suggest_float("emb_noise", 0.00, 0.015)
            self.args.lr = trial.suggest_float("lr", 1e-3, 2e-3, log=True)

            # existing loss knobs
            self.args.huber_beta = trial.suggest_float("huber_beta", 0.6, 1.0)
            self.args.end_weight = trial.suggest_float("end_weight", 0.85, 1.15)
            self.args.extra_order_pen = trial.suggest_float("extra_order_pen", 0.0, 0.06)

            # simple structural toggles you already have
            self.args.use_interaction = trial.suggest_categorical("use_interaction", [0, 1])
            self.args.use_prod = trial.suggest_categorical("use_prod", [0, 1])

            # Fit once, read best val_softIoU
            _, best_iou = self.fit_return_best()
            # Optuna maximizes when we return larger numbers directly
            return best_iou

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=int(getattr(self.args, "seed", 42))),
            pruner=MedianPruner(n_startup_trials=5),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout_s if timeout_s > 0 else None)

        print("\n[OPTUNA] Best trial:")
        print("  value (val_softIoU):", study.best_value)
        print("  params:", study.best_params)

    def training(self):
        """
        Train models with KvsAll or NegativeSampling
        :return:
        """
        # 1. Select model and labelling : triple prediction.
        print("Loading data")       # my editing for checking where my code gets killed?

        model, form_of_labelling = select_model(self.args)
        # make Optuna/CLI value visible to the model’s training_step
        model.gauss_sigma_idx = float(self.args.gauss_sigma_idx)
        if not self.args.batch_size:
            self.args.batch_size = int(len(self.args.dataset.idx_train_set) / 3) + 1
        if not self.args.val_batch_size:
            self.args.val_batch_size = int(len(self.args.dataset.idx_valid_set) / 2) + 1

        self.args.fast_dev_run=False
        self.args.accumulate_grad_batches = 1
        self.args.deterministic=True

        self.logger.info(f' Standard training starts: {model.name}-labeling:{form_of_labelling}')

        print("  >> #valid triples:", len(self.args.dataset.idx_valid_set))

      #  print(" Execute_TP sees dataset.paired_train_idx", getattr(self.args.dataset, "paired_train_idx", None))
        # 2. Create training data.
        dataset = StandardDataModule(train_set_idx=self.args.dataset.idx_train_set,
                                     valid_set_idx=self.args.dataset.idx_valid_set,
                                     test_set_idx=self.args.dataset.idx_test_set,
                                     entities_count=self.args.dataset.num_entities,
                                     relations_count=self.args.dataset.num_relations,
                                     times_count=self.args.dataset.num_times,
                                     form=form_of_labelling,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_workers)

        print("Data loaded successfully")       # my editing for checking where my code gets killed?

        # 3. Display the selected model's architecture.
        self.logger.info(model)

        train_data = dataset.train_dataloader(batch_size1=self.args.batch_size)
        #batch = next(iter(train_data))
        #print(f"[DEBUG] range batch sampe:", batch)
        val_data = dataset.val_dataloader(batch_size1=self.args.val_batch_size)

        # peek at a tiny batch
        try:
            b = next(iter(train_data))
            h, r, t, y1, y2 = b
            print(f"[PEEK] train batch shapes: h={h.shape}, r={r.shape}, t={t.shape}, "
                  f"y1[min,max]=({int(y1.min())},{int(y1.max())}), "
                  f"y2[min,max]=({int(y2.min())},{int(y2.max())})")
        except Exception as e:
            print(f"[PEEK] could not sample a train batch: {e}")


     #   print(f"Training Data Size: {len(train_data.dataset)}")         # my editing for checking where my code gets killed?
     #   print(f"Validation Data Size: {len(val_data.dataset)}")         # my editing for checking where my code gets killed?

        # Create a KFoldCrossValidator instance
        # validator = KFoldCrossValidator(model, train_data, val_data, k=5)
        # self.trainer.add_callback(validator)
        # 5. Train model
        self.trainer.fit(model, train_data,val_data)
        # 6. Test model on validation and test sets if possible.
        #if self.args.task != 'range-prediction':
        #self.trainer.test(ckpt_path='best',test_dataloaders=dataset.dataloaders(len(self.args.dataset.idx_test_set)))
        test_results = self.trainer.test(ckpt_path='best',
                                         dataloaders=dataset.dataloaders(len(self.args.dataset.idx_test_set)))
        test_loss = float(test_results[0].get("test_loss", float("nan")))
        #self.evaluate(model, dataset.train_set_idx, 'Evaluation of Train data: ' + form_of_labelling)
        #self.evaluate(model, dataset.test_set_idx, 'Evaluation of Test data: ' + form_of_labelling)
        if self.args.task == 'range-prediction':
            train_metrics = self.evaluate(model, self.args.dataset.idx_train_set, 'Evaluation of Train data: ' + form_of_labelling)
            test_metrics  = self.evaluate(model, self.args.dataset.idx_test_set, 'Evaluation of Test data: '+ form_of_labelling)

            #FOR PUTTING EVERYTHING IN THE CSV file
            # Also log best val/test losses if you want
            # You can read the last logged values from self.trainer.callback_metrics if needed
            from pathlib import Path, PurePath
            import csv, datetime

            results_path = Path(self.storage_path) / "results.csv"
            header = [
                "timestamp", "run_folder", "preset", "seed",
                "loss_type", "huber_beta", "use_interaction", "use_prod",
                "extra_order_pen", "end_weight",
                "embedding_dim", "batch_size", "lr",
                "val_loss_best", "test_loss",
                "train_exact_match", "train_mae_start", "train_mae_end", "train_iou",
                "test_exact_match", "test_mae_start", "test_mae_end", "test_iou"
            ]

            # pull some values
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            run_folder = self.storage_path
            loss_type = getattr(self.args, "loss_type", "l1")
            huber_beta = getattr(self.args, "huber_beta", 0.5)
            use_interaction = getattr(self.args, "use_interaction", False)
            use_prod = getattr(self.args, "use_prod", False)
            extra_order_pen = getattr(self.args, "extra_order_pen", 0.0)
            end_weight = getattr(self.args, "end_weight", 1.0)
            seed = getattr(self.args, "seed", 42)
            embedding_dim = self.args.embedding_dim
            batch_size = self.args.batch_size
            lr = getattr(model, "lr", getattr(self.args, "lr", 1e-3))

            # read from checkpoint filename or callback metrics if available
            #val_loss_best = float(self.trainer.callback_metrics.get("val_loss", torch.tensor(float('nan'))))
            #val_loss_best = float(self.trainer.callback_metrics.get("val_loss", torch.tensor(float("nan"))))
            # After self.trainer.test(...), test loss is in self.trainer.callback_metrics as well
            #test_loss = float(self.trainer.callback_metrics.get("test_loss", torch.tensor(float('nan'))))

           # row = [
            #    ts, run_folder, loss_type, huber_beta, use_interaction,
            #    order_penalty_lambda, embedding_dim, batch_size, lr,
            #    val_loss_best, test_loss,
           #     train_metrics["exact_match"], train_metrics["mae_start_years"], train_metrics["mae_end_years"],
            #    test_metrics["exact_match"], test_metrics["mae_start_years"], test_metrics["mae_end_years"],
           # ]

            # best val_loss from checkpoint callback (more accurate than last logged)
            best_val = None
            for cb in self.trainer.callbacks:
                if hasattr(cb, "best_model_score") and cb.best_model_score is not None:
                    best_val = cb.best_model_score
            val_loss_best = float(best_val.item()) if best_val is not None else float("nan")
# test_loss was captured earlier from self.trainer.test() return value
# already stored in `test_loss` variable

# pull actual learning rate from model (default fallback if missing)
            #lr = getattr(model, "lr", 1e-4)
# optional: preset label for convenience
            preset = f"{loss_type}_{'prod' if use_prod else 'base'}"

            row = [
                ts, run_folder, preset, seed,
                loss_type, huber_beta, use_interaction, use_prod,
                extra_order_pen, end_weight,
                embedding_dim, batch_size, lr,
                val_loss_best, test_loss,
                train_metrics["exact_match"], train_metrics["mae_start_years"], train_metrics["mae_end_years"], train_metrics["iou"],
                test_metrics["exact_match"], test_metrics["mae_start_years"], test_metrics["mae_end_years"], test_metrics["iou"],
            ]
            file_exists = results_path.exists()
            with open(results_path, "a", newline="") as f:
                w = csv.writer(f)
                if not file_exists:
                    w.writerow(header)
                w.writerow(row)
            # FOR PUTTING EVERYTHING IN THE CSV file

        else:
            self.evaluate(model, self.args.dataset.idx_train_set, 'Evaluation of Train data: ' + form_of_labelling)
            self.evaluate(model, self.args.dataset.idx_test_set, 'Evaluation of Test data: '+ form_of_labelling)
        return model

    def mrr_score2(self, predictions, labels):
        # Convert predictions and labels to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Compute the reciprocal rank for each query
        reciprocal_ranks = []
        for query_index in range(len(predictions)):
            # Get the prediction and label for the current query
            prediction = predictions[query_index]
            label = labels[query_index]

            # Find the rank of the highest ranked relevant item
            rank = np.where(prediction == label)[0][0] + 1
            reciprocal_rank = 1.0 / rank
            reciprocal_ranks.append(reciprocal_rank)

        # Return the mean of all the reciprocal ranks
        return np.mean(reciprocal_ranks)
    def mrr_score(self, y_true, y_pred):
        """
        Calculate MRR (Mean Reciprocal Rank) for a list of predictions.

        Parameters:
        y_true (array): An array of true target values.
        y_pred (array): An array of predicted target values.

        Returns:
        float: The MRR score.
        """
        ranks = []
        for yt, yp in zip(y_true, y_pred):
            rank = np.where(yp == yt)[0][0] + 1
            ranks.append(1 / rank)

        return np.mean(ranks)
    def evaluate(self, model, triple_idx, info):
        dev = next(model.parameters()).device
        print("evaluation")
        model.eval()
        self.logger.info(info)
        #self.logger.info(f'Num of triples {len(triple_idx)}')
        self.logger.info(f'Num of records {len(triple_idx)}')
        """
        X_test = np.array(triple_idx)[:, :6]
        y_test = np.array(triple_idx)[:, -1]

        # label = model.time_embeddings(y_test)
        label = y_test
        X_test_tensor = torch.Tensor(X_test).long()
        Y_test_tensor = torch.Tensor(y_test).long()
        idx_s, idx_p, idx_o, t_idx, s_idx, v_data = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2], X_test_tensor[:, 3], X_test_tensor[:, 4], X_test_tensor[:, 5]
        # 2. Prediction score
        if info.__contains__("Test"):
            prob = model.forward_triples(idx_s, idx_p, idx_o, t_idx, s_idx, v_data,type="test")
        else:
            prob = model.forward_triples(idx_s, idx_p, idx_o, t_idx, s_idx, v_data)
        """
        #load into a single tensor
        X = np.array(triple_idx)
        X_tensor = torch.LongTensor(X)
        if self.args.task == 'range-prediction':
            # 5-column input: (h, r, t, y1_idx, y2_idx)
            N = X_tensor.shape[0]
            dev = next(model.parameters()).device
            # choose a safe eval batch size (use val_batch_size if set; cap to avoid OOM)
            eval_bs = getattr(self.args, "val_batch_size", 512) or 512
            eval_bs = int(min(max(64, eval_bs), 2048))

            # running sums (to avoid holding huge tensors)
            exact_match_sum = 0.0
            mae_start_sum = 0.0
            mae_end_sum = 0.0
            acc_pm = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
            iou_sum = 0.0

            model.eval()
            with torch.no_grad():
                for s in range(0, N, eval_bs):
                    e = min(N, s + eval_bs)
                    B = e - s
                    batch = X_tensor[s:e].to(dev)
                    idx_s, idx_p, idx_o, y1_idx, y2_idx = (batch[:, i] for i in range(5))

                    # forward to index space (float), then clamp and round
                    start_idx_f, end_idx_f = model.forward_triples(idx_s, idx_p, idx_o, y1_idx, y2_idx,
                                                                   type="test" if "Test" in info else None)

                    if start_idx_f.dtype.is_floating_point:
                        pred_start = start_idx_f.round().clamp(0, self.args.num_times - 1).long()
                    else:
                        pred_start = start_idx_f.clamp(0, self.args.num_times - 1).long()

                    if end_idx_f.dtype.is_floating_point:
                        pred_end = end_idx_f.round().clamp(0, self.args.num_times - 1).long()
                    else:
                        pred_end = end_idx_f.clamp(0, self.args.num_times - 1).long()

                    # enforce start <= end
                    swap = pred_start > pred_end
                    if swap.any():
                        tmp = pred_start[swap].clone()
                        pred_start[swap] = pred_end[swap]
                        pred_end[swap] = tmp

                    # exact match on indices
                    exact_match_sum += float(((pred_start == y1_idx) & (pred_end == y2_idx)).float().sum().item())

                    # convert indices -> YEARS (vectorized over the batch with a fast list lookup)
                    # year_idx_dict: idx -> string year; cast to float
                    arr = model.idx_to_year
                    pred_start_years = torch.tensor([float(arr[int(p)]) for p in pred_start.tolist()], device=dev)
                    pred_end_years = torch.tensor([float(arr[int(p)]) for p in pred_end.tolist()], device=dev)
                    y1_years = torch.tensor([float(arr[int(i)]) for i in y1_idx.tolist()], device=dev)
                    y2_years = torch.tensor([float(arr[int(i)]) for i in y2_idx.tolist()], device=dev)

                    # MAE sums
                    mae_start_sum += float(torch.abs(pred_start_years - y1_years).sum().item())
                    mae_end_sum += float(torch.abs(pred_end_years - y2_years).sum().item())

                    # ±k accuracy sums
                    for k in (1, 3, 5, 10):
                        acc_pm[k] += float((
                                                   ((pred_start_years - y1_years).abs() <= k) &
                                                   ((pred_end_years - y2_years).abs() <= k)
                                           ).float().sum().item())

                    # Interval IoU (inclusive years; +1 to treat years as discrete)
                    inter_left = torch.max(pred_start_years, y1_years)
                    inter_right = torch.min(pred_end_years, y2_years)
                    inter = (inter_right - inter_left + 1).clamp(min=0)
                    union = (torch.max(pred_end_years, y2_years) - torch.min(pred_start_years, y1_years) + 1)
                    iou_sum += float((inter / union).sum().item())

            # aggregate
            exact_match = 100.0 * exact_match_sum / N
            mae_start = mae_start_sum / N
            mae_end = mae_end_sum / N
            iou = iou_sum / N
            print(f"[INFO] Eval {info!r} exact-match accuracy: {exact_match:.2f}%")
            print(f"[INFO] Eval {info!r} MAE (start year): {mae_start:.2f} years")
            print(f"[INFO] Eval {info!r} MAE (end year): {mae_end:.2f} years")
            for k in (1, 3, 5, 10):
                print(f"[INFO] Eval {info!r} ±{k}y accuracy: {100.0 * acc_pm[k] / N:.2f}%")
            print(f"[INFO] Eval {info!r} Interval IoU: {iou:.3f}")

            return {
                "exact_match": exact_match,
                "mae_start_years": mae_start,
                "mae_end_years": mae_end,
                "iou": iou,
            }

        else:
            #original 6 column path: (h, r, t, time, sent_idx, veracity)
            X6 = X_tensor[:, :6]
            #y = torch.LongTensor(X[:, -1]).long()
            idx_s, idx_p, idx_o, t_idx, s_idx, v_data = (X6[:, i] for i in range(6))
            prob = model.forward_triples(idx_s, idx_p, idx_o, t_idx, s_idx, v_data, type="test" if "Test" in info else None)
        # pred = (prob > 0.5).float()
            pred = prob.data.detach().numpy()
            max_pred = np.argmax(pred, axis=1)
            idx, sort_pred= torch.sort(prob,dim=1,descending=True)
            return None

        # test_mrr = self.mrr_score(label, sort_pred)
        # self.logger.info(test_mrr)
        # self.logger.info( accuracy_score(max_pred, label))
        # self.logger.info(classification_report(max_pred, label))


# true negatives are ignored
