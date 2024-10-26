import torch
import torch.nn as nn
import numpy as np
import gc
from tqdm.notebook import tqdm
from copy import deepcopy
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from sklearn.linear_model import LogisticRegression
from torchmetrics import F1Score, AUROC, Accuracy
from torchmetrics.functional.classification import binary_f1_score, binary_accuracy


def evaluate_model(
    model,
    data_loader,
    device="cuda",
    threshold=0.5,
    silent=False,
    debug=False,
    **kwargs,
):
    # run model on dataloader, compute metrics
    f1 = F1Score(task="binary", average="macro", threshold=threshold).to(device)
    acc = Accuracy(task="binary", threshold=threshold).to(device)
    auc = AUROC(task="binary").to(device)

    model.to(device)
    model.eval()

    for batch in tqdm(data_loader, desc="evaluating...", disable=silent):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            prediction_scores = torch.sigmoid(model(batch))
        corrects = (batch["responses"] > 0).float()
        mask = batch["out_mask"]

        # # prediction_shape: (batch_size, max_sequence, 'max_slate, 2)
        f1(prediction_scores[mask], corrects[mask])
        auc(prediction_scores[mask], corrects[mask])
        acc(prediction_scores[mask], corrects[mask])
        if debug:
            print("\r", prediction_scores[mask], corrects[mask])

    gc.collect()
    return {
        "f1": f1.compute().item(),
        "roc-auc": auc.compute().item(),
        "accuracy": acc.compute().item(),
    }


def flatten(true, pred, mask, to_cpu=True):
    mask = mask.flatten()
    nnz_idx = mask.nonzero()[:, 0]
    true, pred = [x.flatten()[nnz_idx] for x in [true, pred]]
    if to_cpu:
        true, pred = [x.cpu().numpy() for x in [true, pred]]
    return true, pred


def fit_treshold(labels, scores):
    best_f1, best_thold, acc = 0.0, 0.01, 0.0
    for thold in np.arange(1e-2, 1 - 1e-2, 0.01):
        preds_labels = scores > thold
        f1 = binary_f1_score(preds_labels, labels)
        # print(f"{thold}: {f1}")
        if f1 > best_f1:
            acc = binary_accuracy(preds_labels, labels)
            best_f1, best_thold = f1, thold
    return best_f1, acc, best_thold


class Discriminator(nn.Module):
    def __init__(self, embedding):
        super(Discriminator, self).__init__()
        self.embedding = embedding
        self.emb_dim = embedding.embedding_dim
        self.rnn_layer = nn.GRU(
            input_size=self.emb_dim + 1, hidden_size=self.emb_dim, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, 10), nn.SELU(), nn.Linear(10, 1)
        )

    def forward(self, batch, gen_output):
        item_embs, user_embs = self.embedding(batch)

        items = torch.cat(
            [
                item_embs.flatten(0, 1),
                # item_embs.flatten(0,1)
            ],
            dim=-1,
        )
        h = user_embs.flatten(0, 1)[None, :, :]
        clicks = (batch["responses"].flatten(0, 1) > 0).int().clone()
        mask = batch["slates_mask"].flatten(0, 1).clone()  # padding mask

        # x = {}
        # x['items'] = torch.cat(
        #     [
        #         item_embs.flatten(0,1),
        #         # torch.zeros_like(item_embs.flatten(0,1)),
        #     ],
        #     dim = -1
        # )
        # if self.training:
        #     indices = (batch['length'] - 1)
        # else:
        #     indices = (batch['in_length'] - 1)
        # indices[indices<0] = 0
        # indices = indices[:, None, None].repeat(1, 1, user_embs.size(-1))
        # user_embs = user_embs.gather(1, indices).squeeze(-2).unsqueeze(0)
        # x['users'] = user_embs.repeat_interleave(max_sequence, 1)
        # x['clicks'] = (batch['responses'].flatten(0,1) > 0 ).int().clone()
        # x['mask'] = batch['slates_mask'].flatten(0,1).clone()

        # h = x['users']

        fake = gen_output * mask
        real = clicks
        fake = torch.cat([items.detach(), fake[:, :, None]], axis=2)
        real = torch.cat([items.detach(), real[:, :, None]], axis=2)
        fake_out, _ = self.rnn_layer(fake, h)
        real_out, _ = self.rnn_layer(real, h)
        fake_out = fake_out * mask[:, :, None]
        real_out = real_out * mask[:, :, None]
        fake_out = fake_out.mean(axis=1)
        real_out = real_out.mean(axis=1)
        fake_out = self.mlp(fake_out)[:, 0]
        real_out = self.mlp(real_out)[:, 0]

        return real_out - fake_out


class AdversarialNCM(nn.Module):
    """
    TODO: move this model to common NCM code.
    """

    def __init__(self, embedding, readout=False):
        super().__init__()
        self.embedding = embedding
        self.emb_dim = embedding.embedding_dim
        self.rnn_layer = nn.GRU(
            input_size=self.emb_dim, hidden_size=self.emb_dim, batch_first=True
        )
        self.out_layer = nn.Linear(self.emb_dim, 1)

        self.thr = -1.5
        self.readout = readout
        self.readout_mode = (
            "threshold"  # ['soft' ,'threshold', 'sample', 'diff_sample']
        )

        self.calibration = False
        self.w = 1
        self.b = 0

    def forward(self, batch, detach_embeddings=False, sample=None, to_reshape=True):
        item_embs, user_embs = self.embedding(batch)
        shp = item_embs.shape

        items = torch.cat(
            [
                item_embs.flatten(0, 1),
                # item_embs.flatten(0,1)
            ],
            dim=-1,
        )
        inputs = items.detach() if detach_embeddings else items
        h = user_embs.flatten(0, 1)[None, :, :]
        clicks = (batch["responses"].flatten(0, 1) > 0).int().clone()
        mask = batch["slates_mask"].flatten(0, 1).clone()  # padding mask

        inputs = F.dropout1d(inputs, p=0.1, training=self.training)
        h = F.dropout1d(h, p=0.1, training=self.training)
        rnn_out, _ = self.rnn_layer(inputs, h)
        y = self.out_layer(rnn_out)[:, :, 0]

        if self.training and sample is None:
            clicks_flat, logits_flat = flatten(clicks, y.detach(), mask)
            logreg = LogisticRegression()
            logreg.fit(logits_flat[:, None], clicks_flat)
            γ = 0.3
            self.w = (1 - γ) * self.w + γ * logreg.coef_[0, 0]
            self.b = (1 - γ) * self.b + γ * logreg.intercept_[0]
            y = self.w * y + self.b
        else:
            y = self.w * y + self.b

        if sample:
            eps = 1e-8
            gumbel_sample = -(
                (torch.rand_like(y) + eps).log() / (torch.rand_like(y) + eps).log()
                + eps
            ).log()
            T = 0.5
            bernoulli_sample = torch.sigmoid((nn.LogSigmoid()(y) + gumbel_sample) / T)
            hard_bernoulli_sample = (
                (bernoulli_sample > 0.5).to(torch.float32) - bernoulli_sample
            ).detach() + bernoulli_sample
            return bernoulli_sample if sample == "soft" else hard_bernoulli_sample

        else:
            return y.reshape(shp[:-1]) if to_reshape else y


def train_adversarial(
    model,
    discriminator,
    train_loader,
    val_loader,
    device="cuda",
    lr=1e-3,
    num_epochs=50,
    silent=False,
    early_stopping=None,
    debug=False,
    **kwargs,
):
    if early_stopping is None:
        early_stopping = num_epochs
    model.to(device)
    best_model = model

    auc = AUROC(task="binary").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    opt_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    epochs_without_improvement = 0
    best_val_scores = evaluate_model(
        model, val_loader, device=device, silent=silent, debug=debug
    )
    # best_test_scores = evaluate_model(model, test_loader, device=device, silent=silent, debug=debug)
    best_loss = 999.0

    # print(f"Test before learning: {best_test_scores}")
    ebar = tqdm(range(num_epochs), desc="train")

    for epoch in ebar:
        loss_accumulated = 0.0
        mean_grad_norm = 0.0
        model.train()

        labels = []
        preds = []

        gc.collect()
        # torch.cuda.empty_cache()

        if epoch > 10:
            # discriminator training
            for batch in tqdm(train_loader, desc=f"epoch {epoch}", disable=silent):
                batch = {k: v.to(device) for k, v in batch.items()}

                sample_gen = model(batch, sample="hard", to_reshape=False).detach()
                logits_dis = discriminator(batch, sample_gen)
                opt_dis.zero_grad()
                loss_dis = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits_dis, torch.ones(logits_dis.shape[0]).to(device)
                )
                loss_dis.backward()
                opt_dis.step()

        if epoch > 20:
            for g in optimizer.param_groups:
                g["lr"] = 0.0001

            # adversarial generator training
            for batch in tqdm(train_loader, desc=f"epoch {epoch}", disable=silent):
                batch = {k: v.to(device) for k, v in batch.items()}

                sample_gen = model(batch, detach_embeddings=True, sample="soft")
                logits_dis = discriminator(batch, sample_gen)

                optimizer.zero_grad()
                loss_gen = F.binary_cross_entropy_with_logits(
                    1 - logits_dis, torch.ones(logits_dis.shape[0]).to(device)
                )
                loss_gen.backward()
                optimizer.step()

        for g in optimizer.param_groups:
            g["lr"] = 0.001

        for batch in tqdm(train_loader, desc=f"epoch {epoch}", disable=silent):
            batch = {k: v.to(device) for k, v in batch.items()}
            raw_scores = model(batch)  ##################################
            prediction_scores = torch.sigmoid(raw_scores)
            corrects = (batch["responses"] > 0).float()
            mask = batch["slates_mask"]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                raw_scores[mask],
                corrects[mask],
            )  ######
            loss.backward()  ############
            mean_grad_norm += clip_grad_norm_(model.parameters(), 1).sum().item()
            optimizer.step()
            loss_accumulated += loss.detach().cpu().item()
            labels.append(corrects[batch["out_mask"]].detach().cpu())
            preds.append(prediction_scores[batch["out_mask"]].detach().cpu())
            auc(
                prediction_scores[batch["out_mask"]].detach().cpu(),
                corrects[batch["out_mask"]].detach().cpu(),
            )

        f1, acc, thold = fit_treshold(torch.cat(labels), torch.cat(preds))
        ebar.set_description(f"train... loss:{loss_accumulated}")
        val_m = evaluate_model(
            model,
            val_loader,
            device=device,
            threshold=thold,
            silent=silent,
            debug=debug,
            **kwargs,
        )
        if not silent:
            print(
                f"Train: epoch: {epoch} | accuracy: {acc} | "
                f"f1: {f1} | loss: {loss_accumulated} | "
                f"auc: {auc.compute()}  | thld {thold} | grad_norm: {mean_grad_norm / len(train_loader)}"
            )
            print(
                f"Val: epoch: {epoch} | accuracy: {val_m['accuracy']} | f1: {val_m['f1']} | auc: {val_m['roc-auc']}"
            )

        epochs_without_improvement += 1
        if (val_m["roc-auc"], val_m["f1"], val_m["accuracy"]) > (
            best_val_scores["roc-auc"],
            best_val_scores["f1"],
            best_val_scores["accuracy"],
        ):
            best_model = deepcopy(model)
            best_val_scores = val_m
            # best_test_scores = evaluate_model(model, test_loader, device=device, threshold=thold, silent=silent )
            print(
                f"Val update: epoch: {epoch} |"
                f"accuracy: {best_val_scores['accuracy']} | "
                f"f1: {best_val_scores['f1']} | "
                f"auc: {best_val_scores['roc-auc']} | "
                f"treshold: {thold}"
            )
            # print(f"Test: "
            #       f"accuracy: {best_test_scores['accuracy']} | "
            #       f"f1: {best_test_scores['f1']} | "
            #       f"auc: {best_test_scores['roc-auc']} | "
            # )

        auc.reset()

        if best_loss > loss_accumulated:
            epochs_without_improvement = 0
            best_loss = loss_accumulated

        if epochs_without_improvement >= early_stopping or (
            best_val_scores["roc-auc"] == 1.0
            and best_val_scores["f1"] == 1.0
            and best_val_scores["accuracy"] == 1.0
        ):
            break
    return best_model, best_val_scores, thold
