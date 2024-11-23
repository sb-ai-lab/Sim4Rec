import os
import pickle
from collections import namedtuple
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchmetrics import AUROC
from torchmetrics.functional import accuracy, f1_score
from tqdm.auto import tqdm

try:
    import mlflow
except ImportError:
    pass

from .adversarial import AdversarialNCM
from .datasets import RecommendationData, PandasRecommendationData
from .sessionwise import (
    SCOT,
    AggregatedSlatewiseGRU,
    DummyTransformerGRU,
    SessionwiseGRU,
    SessionwiseTransformer,
    TransformerGRU,
)
from .slatewise import (
    DotProduct,
    LogisticRegression,
    NeuralClickModel,
    SlatewiseGRU,
    SlatewiseTransformer,
)
from .utils import create_loader

Metrics = namedtuple("metrics", ["rocauc", "f1", "accuracy"])


class ResponseModel:
    def __init__(
        self, model, embeddings, calibrator=None, log_to_mlflow=False, **kwargs
    ):
        """
        :param model: string name of model
        :param embeddings: exemplar of embeddings class
        :param calibrator: Sklearn-compatible calibration instance.
        If given, responses are generated according to predicted probabilities
        (i.e., with threshold 0.5 for determiinistic responses or sampled),
        otherwise theshold is fitted to raw model scores.
        """
        self._embeddings = embeddings
        self.model_name = model
        self.threshold = 0.5
        self._calibrator = calibrator
        self.auc = AUROC(task="binary")
        self.log_to_mlflow = log_to_mlflow
        if model == "DotProduct":
            self._model = DotProduct(embeddings, **kwargs)
        elif model == "LogisticRegression":
            self._model = LogisticRegression(embeddings, **kwargs)
        elif model == "SlatewiseTransformer":
            self._model = SlatewiseTransformer(embeddings, **kwargs)
        elif model == "SessionwiseTransformer":
            self._model = SessionwiseTransformer(embeddings, **kwargs)
        elif model == "DummyTransformerGRU":
            self._model = DummyTransformerGRU(embeddings, **kwargs)
        elif model == "TransformerGRU":
            self._model = TransformerGRU(embeddings, **kwargs)
        elif model == "SCOT":
            self._model = SCOT(embeddings, **kwargs)
        elif model == "SlatewiseGRU":
            self._model = SlatewiseGRU(embeddings, **kwargs)
        elif model == "AggregatedSlatewiseGRU":
            self._model = AggregatedSlatewiseGRU(embeddings, **kwargs)
        elif model == "SessionwiseGRU":
            self._model = SessionwiseGRU(embeddings, **kwargs)
        elif model == "NCMBase":
            self._model = NeuralClickModel(embeddings, **kwargs)
        elif model == "NCMDiffSample":
            self._model = NeuralClickModel(embeddings, readout="diff_sample", **kwargs)
        elif model == "AdversarialNCM":
            self._model = AdversarialNCM(embeddings)
        else:
            raise ValueError(f"unknown model {model}")
        if self.log_to_mlflow:
            mlflow.log_params({"model": self.model_name, **kwargs})

    def set_calibrator(self, calibrator):
        self._calibrator = calibrator

    def dump(self, path):
        """
        Saves model's parameters and weights checkpoint on a disk.
        :param path: where the model is saved.
        """
        params = {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "device": self.device,
            "calibrator": self._calibrator,
        }
        with open(os.path.join(path, "params.pkl"), "wb") as f:
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
        torch.save(self._model, os.path.join(path, "model.pt"))
        torch.save(self._embeddings, os.path.join(path, "embeddings.pt"))

    @classmethod
    def load(cls, path):
        """
        Loads model from files creqated by `dump` method.
        :param path: where data is located
        """
        embeddings = torch.load(os.path.join(path, "embeddings.pt"))
        with open(os.path.join(path, "params.pkl"), "rb") as f:
            params = pickle.load(f)
        model = cls(params["model_name"], embeddings)
        model.threshold = params["threshold"]
        model._model = torch.load(os.path.join(path, "model.pt"))
        model.device = params["device"]
        return model

    def _val_epoch(self, data_loader, silent=True):
        # run model on dataloader, compute auc
        self.auc.reset()
        self._model.eval()
        loss_accumulated = 0.0
        for batch in tqdm(data_loader, desc="evaluate:", disable=silent):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                scores = self._model(batch)
                prediction_probs = torch.sigmoid(scores)
            corrects = (batch["responses"] > 0).float()
            mask = batch["out_mask"]
            self.auc(prediction_probs[mask].cpu(), corrects[mask].cpu())

            criterion = nn.functional.binary_cross_entropy_with_logits
            loss_mask = batch["slates_mask"]
            loss = criterion(
                scores[loss_mask],
                corrects[loss_mask],
            )
            loss_accumulated += loss.cpu().item()

        self.val_loss = loss_accumulated

    def _train_epoch(self, data_loader, optimizer, criterion, silent=False):
        loss_accumulated = 0.0
        for batch in tqdm(data_loader, desc="train epoch:", disable=silent):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            mask = batch["slates_mask"]
            corrects = (batch["responses"] > 0).float()

            scores = self._model(batch)
            loss = criterion(
                scores[mask],
                corrects[mask],
            )
            loss_accumulated += loss.detach().cpu().item()
            loss.backward()
            clip_grad_norm_(self._model.parameters(), 1.0)
            optimizer.step()

            metric_mask = batch["out_mask"]
            self.auc(
                torch.sigmoid(scores[metric_mask]).detach().cpu(),
                corrects[metric_mask].detach().cpu(),
            )

        return loss_accumulated

    def _train(
        self,
        train_loader,
        val_loader,
        device="cuda",
        lr=1e-3,
        num_epochs=100,
        silent=False,
        early_stopping=7,
        debug=False,
    ):
        if early_stopping == 0:
            early_stopping = num_epochs

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        epochs_without_improvement = 0
        criterion = nn.functional.binary_cross_entropy_with_logits
        self.ebar = tqdm(range(num_epochs), desc="epoch")
        best_model = deepcopy(self._model)
        best_model_calibrator = deepcopy(self._calibrator)
        self.best_val_scores = self.evaluate(val_loader, silent=silent)
        best_val_loss = None
        best_epoch = 0

        for epoch in self.ebar:
            self._model.train()
            train_loss = self._train_epoch(
                train_loader, optimizer, criterion, silent=silent
            )
            preds, target = torch.cat(self.auc.preds), torch.cat(self.auc.target)
            self._fit_threshold_f1(preds, target)
            if self._calibrator:
                self._calibrator.fit(preds.numpy(), target.numpy())
            self.auc.reset()

            val_scores = self.evaluate(val_loader, silent=silent)
            epochs_without_improvement += 1
            # choosing best model based on roc_auc, then f1, then accuracy
            if val_scores >= self.best_val_scores:
                best_model = deepcopy(self._model)
                best_model_calibrator = deepcopy(self._calibrator)
                self.best_val_scores = val_scores
                best_epoch = epoch

            if not best_val_loss or best_val_loss > self.val_loss:
                epochs_without_improvement = 0
                best_val_loss = self.val_loss

            if self.log_to_mlflow:
                metrics = {
                    "val_auc": val_scores.rocauc.numpy().tolist(),
                    "val_f1": val_scores.f1.numpy().tolist(),
                    "val_accuracy": val_scores.accuracy.numpy().tolist(),
                    "best_val_auc": self.best_val_scores.rocauc.numpy().tolist(),
                    "best_val_f1": self.best_val_scores.f1.numpy().tolist(),
                    "best_val_accuracy": self.best_val_scores.accuracy.numpy().tolist(),
                    "threshold": self.threshold,
                    "epochs_without_improvement": epochs_without_improvement,
                    "train_loss": train_loss,
                    "val_loss": self.val_loss,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                }
                if self._calibrator is not None:
                    metrics.update(
                        {
                            "calibrator_a_": self._calibrator.a_,
                            "calibrator_b_": self._calibrator.b_,
                        }
                    )

                mlflow.log_metrics(metrics, step=epoch)

            # early stopping
            if epochs_without_improvement >= early_stopping or val_scores == (
                1.0,
                1.0,
                1.0,
            ):
                print("Early stopping")
                break
        self._model = best_model
        self._calibrator = best_model_calibrator

    def evaluate(self, datalaoder, silent=True) -> Metrics:
        self._val_epoch(datalaoder, silent=silent)
        preds, target = torch.cat(self.auc.preds), torch.cat(self.auc.target)
        metrics = Metrics(
            self.auc.compute(),
            f1_score(
                preds, target, task="binary", threshold=self.threshold, average="macro"
            ),
            accuracy(preds, target, task="binary", threshold=self.threshold),
        )
        if not silent:
            print(metrics)
        return metrics

    def _get_probs_and_responses(self, raw_scores, response_type="sample"):
        """
        Compute calibrated probabilities and predicted responses for given raw model scores.

        :param raw_scores: backbone model output, after sigmmoid
        :param resoponse_type: 'sample' or 'deterministic'
        """
        if self._calibrator is None:
            predicted_probs = raw_scores
            predicted_responses = (predicted_probs >= self.threshold).long()
        else:
            shp = raw_scores.shape
            predicted_probs = torch.tensor(
                self._calibrator.predict(raw_scores.cpu().flatten())
            ).reshape(shp)
            predicted_responses = (predicted_probs >= 0.5).long()
        if response_type == "sample":
            predicted_responses = torch.bernoulli(predicted_probs).long()
        elif response_type == "deterministic":
            pass
        else:
            raise ValueError(f"unkbnown response type {response_type}")
        return predicted_probs, predicted_responses

    def fit(
        self,
        train_data: RecommendationData,
        batch_size,
        device="cuda",
        silent=False,
        val_data: RecommendationData = None,
        **kwargs,
    ):
        """
        Fits model to given dataset.
        """

        self.to(device)

        # if validation dataset is not given, split train
        if val_data is None:
            train_data, val_data = train_data.split_by_users(0.8, seed=123)
        val_loader = create_loader(val_data, batch_size=batch_size)
        train_loader = create_loader(train_data, batch_size=batch_size)

        # dot product with svd or explicit embeddings has no params to fit
        param_num = sum(
            p.numel() for p in self._embeddings.parameters() if p.requires_grad
        )
        param_num += sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        if param_num == 0:
            self.best_model = deepcopy(self._model)
            self.best_val_scores = self.evaluate(val_loader)
        else:
            self._train(
                train_loader, val_loader, silent=silent, device=device, **kwargs
            )

    def _get_scores(
        self,
        dataset: RecommendationData,
        batch_size: int,
        **kwargs,
    ):
        """
        Run model on dataset, get predicted click probabilities.

        :return: (user_ids, timestamps, items, scores)
        """
        users, items, scores, timestamps = [], [], [], []

        loader = create_loader(dataset, batch_size=batch_size)
        for batch in loader:
            with torch.no_grad():
                # run model
                batch = {k: v.to(self.device) for k, v in batch.items()}
                mask = batch["slates_mask"]
                raw_scores = torch.sigmoid(self._model(batch))
                items.append(batch["item_indexes"][mask].detach().cpu().numpy())
                # create a view s.t. [user_1, ... ] -> [[user_1 x slate_size] x sequence_size, ... ]
                # then select by mask for items to allign with scores and item_idx sequences
                users.append(
                    batch["user_indexes"][:, None, None]
                    .expand_as(mask)[mask]
                    .detach()
                    .cpu()
                    .numpy()
                )
                scores.append(raw_scores[mask].detach().cpu().numpy())
                timestamps.append(batch["timestamps"][mask].detach().cpu().numpy())
            return users, timestamps, items, scores

    def transform(self, dataset, batch_size=128, **kwargs):
        """
        Returns a recommendation dataset with response probabilities provided.

        :param RecommendationData dataset: datset to operate on.

        """
        if type(dataset) is PandasRecommendationData:
            user_idx, timestamp, item_idx, score = self._get_scores(
                dataset, batch_size, **kwargs
            )
            score_df = pd.DataFrame(
                {
                    "user_index": np.concatenate(user_idx),
                    "__iter": np.concatenate(timestamp),
                    "item_index": np.concatenate(item_idx),
                    "score": np.concatenate(score),
                }
            )
            return deepcopy(dataset).apply_scoring(score_df)
        else:
            raise NotImplementedError

    def to(self, device: str):
        self._model = self._model.to(device)
        self.device = device

    def _fit_threshold_f1(self, preds, target):
        best_f1 = 0.0
        for thold in np.arange(0.0, 1.0, 0.01):
            f1 = f1_score(
                preds, target, task="binary", threshold=thold, average="macro"
            ).item()
            if f1 >= best_f1:
                self.threshold = thold
                best_f1 = f1
        return best_f1
