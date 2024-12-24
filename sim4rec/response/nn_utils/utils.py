import torch
import numpy as np

from collections import Counter
from copy import deepcopy
from itertools import chain
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler


def pad_slates(arr, slate_size, padding_value):
    return np.pad(
        arr,
        ((0, 0), (0, slate_size - arr.shape[1])),
        mode="constant",
        constant_values=padding_value,
    )


def collate_rec_data(batch: list, padding_value=0):
    """
    Batch sessions of varying length: pad sequiences, prepare masks, etc.
    :param list bacth: list of data points (usually obtained from torch sampler).
    :param padding_value: value representing padding items/users.
    :return dict batch: bathed data in torch.tensor format.
    """

    # lengths
    batch_lengths = [b["length"] for b in batch]
    batch_lengths = torch.tensor(np.stack(batch_lengths), dtype=torch.long)

    # max_sequence_len = max([b["length"] for b in batch])
    max_slate_size = max([b["slate_size"] for b in batch])

    # if data contain slates of various length,
    # pad the slates before paddings the sequences
    for b in batch:
        if b["slate_size"] < max_slate_size:
            b["item_indexes"] = pad_slates(
                b["item_indexes"], max_slate_size, padding_value
            )
            b["slates_mask"] = pad_slates(
                b["slates_mask"], max_slate_size, padding_value
            )
            b["responses"] = pad_slates(b["responses"], max_slate_size, padding_value)
            b["timestamps"] = pad_slates(b["timestamps"], max_slate_size, padding_value)

    # user indexes
    # print([b["user_index"] for b in batch])
    user_indexes = torch.tensor([b["user_index"] for b in batch], dtype=torch.long)

    # item indexes
    # shape: batch_size, max_sequence_len, max_slate_size
    item_indexes = [torch.tensor(b["item_indexes"], dtype=torch.long) for b in batch]
    item_indexes = torch.nn.utils.rnn.pad_sequence(
        item_indexes, padding_value=padding_value, batch_first=True
    )

    # item mask: True for recommended items, False for paddings
    # (both sequence padding and slate padding)
    # shape: batch_size, max_sequence_len, max_slate_size
    slate_masks = [torch.tensor(b["slates_mask"], dtype=torch.bool) for b in batch]
    slate_masks = torch.nn.utils.rnn.pad_sequence(
        slate_masks, padding_value=False, batch_first=True
    )

    # responses: number of clicks per recommended item
    # shape: batch_size, max_sequence_len, max_slate_size
    responses = [torch.tensor(b["responses"], dtype=torch.long) for b in batch]
    responses = torch.nn.utils.rnn.pad_sequence(
        responses, padding_value=padding_value, batch_first=True
    )

    # timestamps: we assume that (user_id, timestamp) is an unique
    # identifier of slate, hence we need to pass it through model
    # for further decoding model outputs
    # shape: batch_size, max_sequence_len, max_slate_size
    timestamps = [torch.tensor(b["timestamps"], dtype=torch.int) for b in batch]
    timestamps = torch.nn.utils.rnn.pad_sequence(
        timestamps, padding_value=padding_value, batch_first=True
    )
    batch = {
        "item_indexes": item_indexes,  # recommended items indexes
        "slates_mask": slate_masks,  # batch mask, True for non-padding items
        "responses": responses,  # number of clicks for each item
        "timestamps": timestamps,  # interaction timestamp
        "length": batch_lengths,  # lenghts of each session in batch
        "user_indexes": user_indexes,  # indexes of users
        "out_mask": slate_masks,  # a mask for train-time metric computation
    }
    return batch


def concat_batch(left, right):
    """
    Concatenate two batches (before collating).
    In_lengths are summed.
    Recommendation_idx are concatenated.
    """
    sessionwise_fields = [
        "item_indexes",
        "slates_mask",
        "responses",
        "timestamps",
        "user_indexes",
        "recommendation_idx",
    ]
    assert len(left) == len(right)
    left_length = len(left)

    newbatch = deepcopy(left)
    for i in range(left_length):
        for key in sessionwise_fields:
            if left[i][key] is None:
                continue
            newbatch[i][key] = np.concatenate([left[i][key], right[i][key]], axis=0)
        newbatch[i]["in_length"] += right[i]["in_length"]
        newbatch[i]["length"] += right[i]["length"]
    return newbatch


def create_loader(dataset, batch_size, **kwargs):
    """Creates dataloader for recommendation dataset"""
    return DataLoader(
        dataset,
        sampler=BatchSampler(
            SubsetRandomSampler(dataset.users), batch_size, drop_last=False
        ),
        batch_size=None,
        collate_fn=collate_rec_data,
        **kwargs,
    )


class Indexer:
    """
    Handles mappings between some indexes and ids.
    Padding tokens will always have index 0, and
    unknown (rare) tokens will have index 1.
    """

    def __init__(self, pad_id=-1, unk_id=-2, **kwargs):
        """
        :param pad_id: - Id for padding item.
        :param unk_id: - Id for unknown item.
        """
        self.unk_id = unk_id
        self.pad_id = pad_id
        self._index2id = [pad_id, unk_id]
        # unknown indexes are mapped to dedicated 'unknown id' index
        self._id2index = {pad_id: 0, unk_id: 1}
        self._counter = Counter()

    def to_dict(self):
        """Raw id -> index mapping"""
        return self._id2index.copy()

    @property
    def n_objs(self):
        """Number of indexed IDs"""
        return len(self._index2id)

    def is_known_id(self, id_):
        return (id_ in self._id2index) and (id_ != self.unk_id)

    @classmethod
    def from_dict(cls, d):
        """
        Creates indexer from dictionary.

        :param d: id-to-index dictionary to update from
        """
        assert set(d.values()) == set(range(len(d))), "Given dict values aren't indexes"
        pad, unk = None, None
        for key, value in d.items():
            if value == 0:
                pad = key
            if value == 1:
                unk = key
        indexer = cls(pad_id=pad, unk_id=unk)
        indexer._id2index.update(d)
        indexer._index2id = [0 for i in d]
        for key, value in d.items():
            indexer._index2id[value] = key
        return indexer

    def update_from_iter(self, iterable, min_occurrences=1):
        """
        Builds mapping from given iterable.
        If called several times, updates the mapping with new values,
        preserving already allocated indexes. Note, that frequencies
        are preserved between calls.

        :param iterable: some collection with data.
        :param min_occurrences: frequency threshold for rare ids.
        """
        self._counter.update(iterable)

        known_ids = set(self._index2id)
        new_ids = [
            key
            for key in self._counter
            if self._counter[key] >= min_occurrences and key not in known_ids
        ]
        last_index = len(self._index2id)
        self._index2id += new_ids
        self._id2index.update(
            {key: index + last_index for (index, key) in enumerate(new_ids)}
        )

        return self

    def index_np(self, arr: np.array):
        """
        Transforms given array of IDs into array of indexes.
        Previously unseen IDs will be silently replaces with unk.

        :param arr: numpy array of ID's.
        :return: numpy array of the same shape filled with indexes.
        """
        unk_index = self._id2index[self.unk_id]
        vfunc = np.vectorize(lambda x: self._id2index.get(x, unk_index))
        return vfunc(arr)

    # def index_df(self, df, inputCol, outputCol):
    #     """
    #     Apply indexing to the whole spark dataframe colmn.
    #     """
    #     mapping_expr = create_map([lit(x) for x in chain(*self._id2index.items())])
    #     return df.withColumn(outputCol, coalesce(mapping_expr[col(inputCol)], lit(1)))

    def get_id(self, arr: np.array):
        """
        Transforms given array of indexes back into array of IDs.
        Throws an exception if found incorrect index.

        :param arr: numpy array of indexes's.
        :return: numpy array of the same shape filled with ids.
        """
        return np.vectorize(lambda x: self._index2id[x])(arr)
