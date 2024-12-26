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

def stack_embeddings(user_embs, item_embs):
    """Concatenate user and item embeddings"""
    return torch.cat(
        [item_embs, user_embs[:, :, None, :].repeat(1, 1, item_embs.size(-2), 1)],
        dim=-1,
    )

