import torch
import numpy as np
import torch.nn as nn
from sklearn.utils.extmath import randomized_svd
from abc import ABC, abstractmethod


class EmbeddingBase(ABC, nn.Module):
    """
    Defines a common interface for all embeddings

    :param embedding_dim: desired dimensionality of the embedding layer.
    :param user_aggregate: if None, user embeddings will be produced
        independently from item embeddings. Otherwise, user embeddings are
        aggregations ('sum' or 'mean') of items previously consumed by this user.
        Default: 'mean'.
    """

    def __init__(self, embedding_dim: int, user_aggregate: str = "mean"):
        super().__init__()
        self.user_agg = user_aggregate
        self.embedding_dim = embedding_dim

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method returns embeddings for both items and users.

        If user aggregation is specified, the resulting user embeddings
        are obtained by aggregating consumed item embeddings. Otherwise,
        produce items and user embeddings independintly.

        Shapes of embedings are `(batch_size, sequence_len, slate_size, embedding_dim)`
        for items and `(batch_size, sequence_len, embedding_dim)` for users.

        :param batch: batched data (see :func:`utils.collate_recommendation_data`)
        :returns: a tuple of item and user embeddings.
        """
        item_embeddings = self._get_item_embeddings(batch)
        if not self.user_agg:
            user_embeddings = self._get_user_embeddings(batch)
        else:
            user_embeddings = self._aggregate_item_embeddings(item_embeddings, batch)
        return item_embeddings, user_embeddings

    @abstractmethod
    def _get_item_embeddings(self, batch):
        """
        This method produces embeddings for all items in the batch.

        :param batch: batched data (see :func:`utils.collate_recommendation_data`)
        :returns: tensor containing user embeddings for each recommendation with shape:
                                `(batch_size, sequence_len, slate_size, embedding_dim)`
        """
        pass

    @abstractmethod
    def _get_user_embeddings(self, batch):
        """
        This method produces embeddings for all items in the batch.

        :param batch: batched data (see :func:`utils.collate_recommendation_data`)
        :returns: tensor containing user embeddings for each recommendation with shape:
                                `(batch_size, sequence_len, embedding_dim)`
        """
        pass

    def _aggregate_item_embeddings(self, item_embeddings, batch):
        """
        Aggregate consumed item embeddings into user embeddings.

        For each user in batch, at interaction `t` user embedding is defined as
        aggregating all items during interactions `1 \hdots t-1`, which
        received positive responses. Aggregation can be "sum" or "mean",
        depending on the initialization parameter `user_aggregate.
        For the first interaction in each session (hence without any previous clicks),
        user embeddings are set to zero.

        :param item_embeddings: tensor with item embeddings with expected shape:
                                `(batch_size, sequence_len, slate_size, embedding dim)`.
        :param batch: batched data (see :func:`utils.collate_recommendation_data`)
        :returns: tensor containing user embeddings for each recommendation with shape:
                                `(batch_size, sequence_len, embedding_dim)`
        """
        batch_size, max_sequence = batch["responses"].shape[:2]

        # Shift the batch for 1 item to avoid leakage
        item_embeddings_shifted = torch.cat(
            [
                torch.zeros_like(item_embeddings[:, :1, :, :]),
                item_embeddings[:, :-1, :, :],
            ],
            dim=1,
        )

        responses_shifted = torch.cat(
            [
                torch.zeros_like(batch["responses"][:, :1, :]),
                batch["responses"][:, :-1, :],
            ],
            dim=1,
        )
        consumed_embedding = item_embeddings_shifted * responses_shifted[..., None]
        consumed_embedding = consumed_embedding.sum(-2).cumsum(-2)

        if self.user_agg == "sum":
            pass
        elif self.user_agg == "mean":
            total_responses = responses_shifted.sum(-1).cumsum(-1)
            nnz = total_responses > 0
            consumed_embedding[nnz] /= total_responses[nnz].unsqueeze(-1)
        else:
            raise ValueError(f"Unknown aggregation {self.agg}")

        return consumed_embedding


class NumericalEmbedding(EmbeddingBase):
    """
    Embeddings obtained from numerical features represented in the dataset.
    As models expect user and item embeddings to be equal in size, the
    features are projected into the same space with nn.Linear layers.

    It is assumed, that every batch contains the 'item_embeddings'
    and 'user_embeddings' keys with the value being torch.FloatTensor with shape
    (batch_size, sequence_len, slate_size, embedding_dim) for items and
    (batch_size, sequence_len, embedding_dim) for users.

    :param item_dim: dimensionality of item numerical features in your data.
    :param user_dim: dimensionality of user numerical features in your data.
    """

    def __init__(
        self,
        item_dim: int,
        user_dim: int = None,
        embedding_dim: int = 32,
        user_aggregate: str = "mean",
    ):
        super().__init__(embedding_dim, user_aggregate=user_aggregate)
        self.item_projection_layer = nn.Linear(item_dim, embedding_dim)
        if not user_aggregate:
            assert user_dim, "user embeddings size is undefined"
            self.user_projection_layer = nn.Linear(user_dim, embedding_dim)

    def _get_item_embeddings(self, batch):
        return self.item_projection_layer(batch["item_embeddings"])

    def _get_user_embeddings(self, batch):
        return self.user_projection_layer(batch["user_embeddings"])


class IndexEmbedding(EmbeddingBase):
    """
    Learnable nn.Embeddings for item and user indexes.

    It is assumed, that every batch contains the 'item_indexes'
    and 'user_indexes' keys with values being torch.tensor of shape
    (batch_size, sequence_len, slate_size) for items and
    (batch_size, sequence_len) for users.


    :param n_items: number of unique items.
    :param n_users: number of unique users.
    """

    def __init__(
        self,
        n_items: int,
        n_users: int = None,
        embedding_dim: int = 32,
        user_aggregate: str = "mean",
    ):
        super().__init__(embedding_dim, user_aggregate=user_aggregate)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        if not user_aggregate:
            assert n_users, "Number of users is undefined"
            self.user_embedding = nn.Embedding(n_users, embedding_dim)

    def _get_item_embeddings(self, batch):
        return self.item_embeddings(batch["item_indexes"])

    def _get_user_embeddings(self, batch):
        return self.user_embedding(batch["user_indexes"])


class SVDEmbedding(EmbeddingBase):
    """
    Static embeddings obtained from SVD decomposition
    of the user-item interaction matrix. Item embedding
    is the right matrix from decomposition, user embedding -
    the left matrix multiplied by the singular values eye matrix.

    It is assumed, that every batch contains the 'item_indexes'
    and 'user_indexes' keys with values being torch.tensor of shape
    (batch_size, sequence_len, slate_size) for items and
    (batch_size, sequence_len) for users, and theese indexes must align
    with indexing of the interaction matrix.


    :param user_item_matrix: interaction matrix with shape (n_users, n_items)
    """

    def __init__(self, user_item_matrix, embedding_dim=32, user_aggregate="mean"):
        super().__init__(embedding_dim=embedding_dim, user_aggregate=user_aggregate)
        self.user_embedding, singular_values, self.item_embedding = randomized_svd(
            user_item_matrix,
            n_components=embedding_dim,
            n_iter=4,
            power_iteration_normalizer="QR",
        )
        self.item_embedding = torch.tensor(self.item_embedding.T).float()
        self.user_embedding = torch.tensor(
            self.user_embedding * singular_values
        ).float()

    def _get_item_embeddings(self, batch):
        return self.item_embedding.to(batch["item_indexes"].device)[
            batch["item_indexes"]
        ]

    def _get_user_embeddings(self, batch):
        return self.user_embedding.to(batch["user_indexes"].device)[
            batch["user_indexes"]
        ]


class CategoricalEmbedding(EmbeddingBase):
    """
    Learnable nn.Embeddings for categorical feature indexes.

    It is assumed, that every batch contains the 'item_categorical'
    and 'user_indexes' keys with values being torch.tensor of shape
    (batch_size, sequence_len, slate_size, n_categorical_features) for items and
    (batch_size, sequence_len, n_categorical_features) for users.

    The overall dimensionality of user and item embeddings is equalized via linear projection.

    :param n_item_features: number of item categorical features.
    :param item_values_count: tuple of length `n_item_features` with number of each feature unique values
    :param n_user_features: number of item categorical features.
    :param user_values_count: tuple of length `n_item_features` with number of each feature unique values
    :param feature_embedding_dim: dimensionality of each feature embedding.
    """

    def __init__(
        self,
        n_item_features: int,
        item_values_count: tuple,
        n_user_features: int = None,
        user_values_count: tuple = (),
        feature_embedding_dim: int = 32,
        user_aggregate="mean",
        embedding_dim: int = 32,
    ):
        super().__init__(embedding_dim, user_aggregate=user_aggregate)

        # item embedding layers
        self.item_embedding = nn.ModuleList([])
        for num_values in item_values_count:
            self.item_embedding.append(nn.Embedding(num_values, feature_embedding_dim))
        self.item_projection_layer = nn.Linear(
            feature_embedding_dim * n_item_features, embedding_dim
        )
        # user embedding layers
        if not user_aggregate:
            assert n_user_features, "user embeddings size is undefined"
            self.user_embedding = nn.ModuleList([])
            for num_values in user_values_count:
                self.user_embedding.append(
                    nn.Embedding(num_values, feature_embedding_dim)
                )
            self.user_projection_layer = nn.Linear(
                n_user_features * feature_embedding_dim, embedding_dim
            )

    def _get_item_embeddings(self, batch):
        embeddings = []
        for i, layer in enumerate(self.item_embedding):
            embeddings.append(layer(batch["item_categorical"][..., i]))
        return self.item_projection_layer(torch.cat(embeddings, axis=-1))

    def _get_user_embeddings(self, batch):
        embeddings = []
        for i, layer in enumerate(self.user_embedding):
            embeddings.append(layer(batch["user_categorical"][..., i]))
        return self.user_projection_layer(torch.cat(embeddings, axis=-1))


def stack_embeddings(user_embs, item_embs):
    """Concatenate user and item embeddings"""
    return torch.cat(
        [item_embs, user_embs[:, :, None, :].repeat(1, 1, item_embs.size(-2), 1)],
        dim=-1,
    )


class MixedEmbedding(nn.Module):
    """
    Concatenates embeddings.

    :param embedding_modules: one or more modules derived from EmbeddingBase.
    """

    def __init__(self, *embedding_modules):
        super().__init__()
        self.embeddings = nn.ModuleList(embedding_modules)
        self.embedding_dim = sum([module.embedding_dim for module in self.embeddings])

    def forward(self, batch):
        item_embeddings = []
        user_embeddings = []
        for module in self.embeddings:
            items, users = module(batch)
            item_embeddings.append(items)
            user_embeddings.append(users)
        return torch.cat(item_embeddings, axis=-1), torch.cat(user_embeddings, axis=-1)


def add_zero_item(item_embeddings):
    """
    Adds an artificial zero item to a given item sequence
    Item embeddings are assumed to be of a shape
    (batch, sequence_len, embedding_dim) or (batch, sequence_len)
    """
    return torch.cat(
        [torch.zeros_like(item_embeddings[:, :1, ...]), item_embeddings], dim=1
    )
