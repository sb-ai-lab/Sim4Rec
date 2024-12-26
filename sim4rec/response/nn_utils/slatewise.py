import torch

# import absч
import torch.nn as nn
from .embeddings import stack_embeddings


class DotProduct(torch.nn.Module):
    """
    Model whose prediction scoreis are just a dot product of user and item embeddings.
    """

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        scores = item_embs * user_embs[:, :, None, :].repeat(
            1, 1, item_embs.size(-2), 1
        )
        scores = scores.sum(-1)
        return scores


class LogisticRegression(torch.nn.Module):
    """
    Logistic Regression run on a concatenation of the user's and the item's embedding.
    """

    def __init__(self, embedding, output_dim=1):
        super().__init__()
        self.embedding = embedding
        self.linear = torch.nn.Linear(2 * embedding.embedding_dim, output_dim)

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        features = stack_embeddings(user_embs, item_embs)
        return self.linear(features).squeeze(-1)


class SlatewiseGRU(torch.nn.Module):
    """
    GRU acting on each slate independently.
    """

    def __init__(self, embedding, dropout=0, output_dim=1):
        super().__init__()
        self.embedding = embedding
        self.rnn_layer = torch.nn.GRU(
            input_size=embedding.embedding_dim,
            hidden_size=embedding.embedding_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.out_layer = torch.nn.Linear(embedding.embedding_dim, output_dim)

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        # (batch_size, session_len, slate_size)
        shp = item_embs.shape[:-1]

        # Reinterpreting session dim as an independent batch dim
        # now it is (batch * session, slate, embedding)
        item_embs = item_embs.flatten(0, 1)
        hidden = user_embs.flatten(0, 1)[None, ...].contiguous()
        rnn_out, _ = self.rnn_layer(
            item_embs,
            hidden,
        )
        return self.out_layer(rnn_out).reshape(shp)


class SlatewiseTransformer(torch.nn.Module):
    """
    Transformer acting on each slate independently.
    """

    def __init__(self, embedding, nheads=2, output_dim=1):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.attention = torch.nn.MultiheadAttention(
            2 * embedding.embedding_dim, num_heads=nheads, batch_first=True
        )
        self.out_layer = torch.nn.Linear(2 * embedding.embedding_dim, output_dim)

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        # (batch_size, session_len, slate_size)
        shp = item_embs.shape[:-1]

        # Reinterpreting session dim as an independent batch dim
        # now it is (batch * session, slate, embedding)
        features = stack_embeddings(user_embs, item_embs)
        features = features.flatten(0, 1)

        # adding a zero item
        features = torch.cat([torch.zeros_like(features[..., :1, :]), features], dim=-2)

        # key padding mask forbids attending to padding items
        key_padding_mask = batch["slates_mask"].flatten(0, 1)
        key_padding_mask = torch.cat(
            [torch.ones_like(key_padding_mask[..., :1]), key_padding_mask], dim=-1
        )

        # evaluating attention layer
        features, attn_map = self.attention(
            features, features, features, key_padding_mask=~key_padding_mask
        )

        # removing zero item
        features = features[..., 1:, :]

        out = self.out_layer(features)
        out = out.reshape(shp).squeeze(-1)
        return out


class NeuralClickModel(nn.Module):
    def __init__(self, embedding, readout=None, gumbel_temperature=1.0):
        """
        :param readout: can be one of None, 'soft' ,'threshold', 'sample' and 'diff_sample'
        """
        super().__init__()
        self.embedding = embedding
        self.embedding_dim = embedding.embedding_dim
        self.rnn_layer = nn.GRU(
            input_size=self.embedding_dim * 2,
            hidden_size=self.embedding_dim,
            batch_first=True,
        )
        self.out_layer = nn.Linear(self.embedding_dim, 1)
        self.readout = readout
        self.gumbel_temperature = gumbel_temperature

    def forward(self, batch, threshold=0.0):
        item_embs, user_embs = self.embedding(batch)
        shp = item_embs.shape
        slate_size = item_embs.shape[2]

        # duplicate item emneddings (second part will be reweighted later)
        items = torch.cat([item_embs.flatten(0, 1), item_embs.flatten(0, 1)], dim=-1)
        h = user_embs.flatten(0, 1)[None, :, :]
        clicks = torch.zeros_like(batch["responses"]).flatten(0, 1)

        if self.readout:
            res = []
            # iterate over recommended items in slate
            for i in range(slate_size):
                output, h = self.rnn_layer(items[:, [i], :], h)
                y = self.out_layer(output)[:, :, 0]
                if i + 1 == slate_size:
                    res.append(y)
                    break
                # update the last half of item embedding
                if self.readout == "threshold":
                    # hard readout
                    clicks = (y.detach()[:, :, None] > threshold).to(torch.float32)
                    items[:, [i + 1], self.embedding_dim :] *= clicks
                elif self.readout == "soft":
                    # soft readout, each item is added with weight equal to predicted click proba
                    items[:, [i + 1], self.embedding_dim :] *= torch.sigmoid(y)[
                        :, :, None
                    ]
                elif self.readout == "diff_sample" or self.readout_mode == "sample":
                    # gumbel-softmax trick
                    eps = 1e-8  # to avoid numerical instability
                    gumbel_sample = (torch.rand_like(y) + eps).log()
                    gumbel_sample /= (torch.rand_like(y) + eps).log() + eps
                    gumbel_sample = gumbel_sample.log()
                    gumbel_sample *= -1
                    bernoulli_sample = torch.sigmoid(
                        (nn.LogSigmoid()(y) + gumbel_sample) / self.gumbel_temperature
                    )
                    if self.readout == "sample":
                        bernoulli_sample = bernoulli_sample.detach()
                    items[:, i + 1, self.embedding_dim :] *= bernoulli_sample
                else:
                    raise NotImplementedError
                res.append(y)
            y = torch.cat(res, axis=1)
        else:
            items[:, 1:, self.embedding_dim :] *= clicks[:, :-1, None]
            rnn_out, _ = self.rnn_layer(items, h)
            y = self.out_layer(rnn_out)[:, :, 0]
        return y.reshape(shp[:-1])
