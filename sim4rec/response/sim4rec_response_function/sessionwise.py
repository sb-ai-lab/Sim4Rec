import torch
import torch.nn as nn
from .embeddings import add_zero_item, stack_embeddings


class SessionwiseGRU(nn.Module):
    """GRU on all recommended items in session"""

    def __init__(self, embedding, output_dim=1, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.rnn_layer = nn.GRU(
            input_size=embedding.embedding_dim,
            hidden_size=embedding.embedding_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.out_layer = nn.Linear(embedding.embedding_dim, output_dim)

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        shp = item_embs.shape[:-1]  # (batch_size, session_len, slate_size)
        # flatening slates into one long sequence
        item_embs = item_embs.flatten(1, 2)
        # hidden is the user embedding before the first iteraction
        hidden = user_embs[None, :, 0, :].contiguous()
        rnn_out, _ = self.rnn_layer(
            item_embs,
            hidden,
        )
        return self.out_layer(rnn_out).reshape(shp)


class AggregatedSlatewiseGRU(nn.Module):
    """
    Slatewise GRU cell, whose hidden state initialized
    with aggregated hiddens over the previous slate
    """

    def __init__(self, embedding, output_dim=1, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.rnn_layer = torch.nn.GRU(
            input_size=embedding.embedding_dim,
            hidden_size=embedding.embedding_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.out_layer = nn.Linear(embedding.embedding_dim, output_dim)

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        shp = item_embs.shape[:-1]  # (batch_size, session_len, slate_size)
        session_len = shp[1]

        # initial hidden is the user embedding before the first iteraction
        hidden = user_embs[None, ..., 0, :].contiguous()
        preds = []

        # iterate over slates
        for slate_no in range(session_len):
            # run GRU on the current slate
            rnn_out, hidden = self.rnn_layer(
                item_embs[..., slate_no, :, :],
                hidden,
            )
            # save output for further prediction
            preds.append(rnn_out[..., None, :, :])
            # aggregate hiddens for the next slate
            hidden = rnn_out.mean(dim=1)[None, :, :]

        preds = torch.cat(preds, axis=1)
        return self.out_layer(preds).reshape(shp)


class SCOT(nn.Module):
    """
    Session-wise transformer, working on sequences of clicked-only items.
    """

    def __init__(self, embedding, nheads=2, output_dim=1, debug=False):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.nheads = nheads
        self.attention = nn.MultiheadAttention(
            2 * self.embedding_dim, num_heads=nheads, batch_first=True
        )
        self.out_layer = nn.Sequential(
            nn.LayerNorm(2 * embedding.embedding_dim),
            nn.Linear(embedding.embedding_dim * 2, embedding.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding.embedding_dim * 2, output_dim),
        )

    def forward(self, batch):
        # getting embeddings & flatening them into one sequence
        item_embs, user_embs = self.embedding(batch)
        item_embs = stack_embeddings(user_embs, item_embs)
        shp = item_embs.shape[:-1]
        device = item_embs.device
        slate_size = item_embs.size(-2)
        session_length = item_embs.size(1)
        batch_size = item_embs.size(0)

        # Tensor of shape (batch, session_length, slate_size)
        # indicating which iteration this item belongs to
        slate_num_for_item = torch.arange(session_length).to(device)
        slate_num_for_item = slate_num_for_item[None, :, None].repeat(
            batch_size, 1, slate_size
        )

        # Adding a dummy "zero item". It is required, pytorch
        # attention implementation will fail if there are sequences
        # with no keys in batch. We will drop out response on it later.
        item_embs = item_embs.flatten(1, 2)
        item_embs = add_zero_item(item_embs)
        slate_num_for_item = slate_num_for_item.flatten(1, 2) + 1
        slate_num_for_item = add_zero_item(slate_num_for_item)

        # gatghering clicked items
        keys = item_embs
        clicked_mask = batch["responses"].flatten(1, 2) > 0
        clicked_mask = ~add_zero_item(~clicked_mask)
        clicked_items_slateno, clicked_items = [], []
        for i in range(batch_size):
            clicked_items.append(keys[i][clicked_mask[i], :])
            clicked_items_slateno.append(slate_num_for_item[i][clicked_mask[i]])
        keys = nn.utils.rnn.pad_sequence(
            clicked_items, batch_first=True, padding_value=float("nan")
        )
        slate_num_clicked_items = nn.utils.rnn.pad_sequence(
            clicked_items_slateno, batch_first=True, padding_value=session_length + 1
        )
        key_padding_mask = keys.isnan().any(-1)
        keys = keys.nan_to_num(0)

        # Now `keys` is a sequence of all clicked items in each session.
        # We are constructing a mask to forbid model looking into future iteractions
        # Mask shape: (num_heads * bsize, all_items_sequence_length, clicked_sequence_len)
        # with True on position a pair (item, clicked_item) if `clicked_item` is recommended
        # after the `item`
        attn_mask = []
        for i in range(batch_size):
            slateno = slate_num_for_item[i]
            clicked_slateno = slate_num_clicked_items[i]
            mask = slateno[:, None] <= clicked_slateno[None, :]
            mask[:, 0] = False  # always can attend the 'zero item'
            attn_mask.append(mask)

        attn_mask = torch.nn.utils.rnn.pad_sequence(
            attn_mask, batch_first=True, padding_value=True
        )
        attn_mask.to(device)

        # Inference the model
        features, attn_map = self.attention(
            item_embs,
            keys,
            keys,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask.repeat_interleave(self.nheads, 0),
        )

        # removing artificial `zero item`
        features = features[:, 1:, :]
        return self.out_layer(features).reshape(shp).squeeze(-1)


class DummyTransformerGRU(nn.Module):
    """
    Output features of slatewise attention layer and
    sessionwise GRU layer are concatenated. This model
    id used for ablation study of Two-Stage Transformer+GRU model.
    """

    def __init__(self, embedding, nheads=2, output_dim=1):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.attention = nn.MultiheadAttention(
            2 * embedding.embedding_dim, num_heads=nheads, batch_first=True
        )
        self.rnn_layer = nn.GRU(
            input_size=embedding.embedding_dim,
            hidden_size=embedding.embedding_dim,
            batch_first=True,
        )
        self.out_layer = nn.Linear(3 * embedding.embedding_dim, output_dim)

    def get_attention_embeddings(self, item_embs, user_embs, slate_mask):
        shp = item_embs.shape[:-1]
        # reinterpret sequence dimension as batch dimension
        features = stack_embeddings(user_embs, item_embs).flatten(0, 1)
        key_padding_mask = slate_mask.flatten(0, 1)
        # add an artificial item
        features = add_zero_item(features)
        key_padding_mask = add_zero_item(~key_padding_mask)
        features, attn_map = self.attention(
            features, features, features, key_padding_mask=key_padding_mask
        )
        # drop the artificial item
        features = features[:, 1:, ...]
        features = features.reshape(shp + (self.embedding_dim * 2,))
        return features

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        slate_mask = batch["slates_mask"].clone()

        # slatewise attention
        att_features = self.get_attention_embeddings(item_embs, user_embs, slate_mask)

        # sequencewise gru
        gru_features, _ = self.rnn_layer(item_embs.flatten(1, 2))
        gru_features = gru_features.reshape(item_embs.shape)

        features = torch.cat([att_features, gru_features], dim=-1)

        return self.out_layer(features).squeeze(-1)


class SessionwiseTransformer(nn.Module):
    """
    Just a large transformer on sequence of items.
    """

    def __init__(self, embedding, nheads=2, output_dim=1):
        super().__init__()
        self.nheads = nheads
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.attention = nn.MultiheadAttention(
            2 * self.embedding_dim, num_heads=nheads, batch_first=True
        )
        self.out_layer = nn.Linear(2 * embedding.embedding_dim, output_dim)

    def forward(self, batch):
        # getting embeddings & flatening them into one sequence
        item_embs, user_embs = self.embedding(batch)
        item_embs = stack_embeddings(user_embs, item_embs)
        shp = item_embs.shape[:-1]
        device = item_embs.device
        slate_size = item_embs.size(-2)
        session_length = item_embs.size(1)
        batch_size = item_embs.size(0)

        # Tensor of shape (batch, session_length, slate_size)
        # indicating which iteration this item belongs to
        slate_num_for_item = torch.arange(session_length).to(device)
        slate_num_for_item = slate_num_for_item[None, :, None].repeat(
            batch_size, 1, slate_size
        )

        # Adding a dummy "zero item". It is required, pytorch
        # attention implementation will fail if there are sequences
        # with no keys in batch. We will drop out response on it later.
        keys = item_embs.flatten(1, 2)
        keys = add_zero_item(keys)
        slate_num_for_item = slate_num_for_item.flatten(1, 2) + 1
        slate_num_for_item = add_zero_item(slate_num_for_item)

        # forbid attending to the padding `items`
        key_padding_mask = batch["slates_mask"].flatten(1, 2).clone()
        key_padding_mask = add_zero_item(~key_padding_mask)

        # forbid model looking into future (and into current iteraction)
        attn_mask = []
        for i in range(batch_size):
            slateno = slate_num_for_item[i]
            clicked_slateno = slate_num_for_item[i]
            mask = slateno[:, None] <= clicked_slateno[None, :]
            mask[:, 0] = False  # always can attend the 'zero item'
            attn_mask.append(mask)
        future_mask = nn.utils.rnn.pad_sequence(
            attn_mask, batch_first=True, padding_value=True
        )
        future_mask.to(device)

        # calculating the attention layer
        features, attn_map = self.attention(
            keys,
            keys,
            keys,
            key_padding_mask=key_padding_mask,
            attn_mask=future_mask.repeat_interleave(self.nheads, 0),
        )

        # removing artificial `zero item`
        features = features[:, 1:, :]
        keys = keys[:, 1:, :]

        return self.out_layer(features).reshape(shp).squeeze(-1)


class TransformerGRU(nn.Module):
    """
    Two-stage model with attention layer operating on each slate independently,
    and a session-wise GRU layer to handle the preference drift.
    """

    def __init__(self, embedding, nheads=2, output_dim=1):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.attention = nn.MultiheadAttention(
            2 * embedding.embedding_dim, num_heads=nheads, batch_first=True
        )
        self.rnn_cell = nn.GRUCell(
            input_size=2 * embedding.embedding_dim,
            hidden_size=embedding.embedding_dim,
        )
        self.out_layer = nn.Linear(2 * embedding.embedding_dim, output_dim)

    def get_attention_embeddings(self, item_embs, user_embs, slate_mask):
        shp = item_embs.shape[:-1]
        # reinterpret sequence dimension as batch dimension
        features = stack_embeddings(user_embs, item_embs).flatten(0, 1)
        key_padding_mask = slate_mask.flatten(0, 1)
        # add an artificial item
        features = add_zero_item(features)
        key_padding_mask = add_zero_item(~key_padding_mask)
        features, attn_map = self.attention(
            features, features, features, key_padding_mask=key_padding_mask
        )
        # drop the artificial item
        features = features[:, 1:, ...]
        features = features.reshape(shp + (self.embedding_dim * 2,))
        return features

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        slate_mask = batch["slates_mask"].clone()
        session_length = item_embs.shape[1]

        preds, hidden = [], user_embs[..., 0, :]
        for slate_no in range(session_length):
            # select current slate, run slatewise attention on it
            att_features = self.get_attention_embeddings(
                item_embs[..., slate_no, :, :].unsqueeze(-3),
                hidden[..., None, :],
                slate_mask[..., slate_no, :].unsqueeze(-3),
            )
            # save the attention features
            preds.append(att_features)

            # run GRU cell on aggregated attention features
            hidden = self.rnn_cell(att_features.squeeze(-3).mean(-2), hidden)
        preds = torch.cat(preds, dim=-3)
        return self.out_layer(preds).squeeze(-1)
