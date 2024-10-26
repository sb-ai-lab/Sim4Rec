import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as sf

from abc import ABC, abstractmethod
from .utils import Indexer
from torch.utils.data import Dataset


class DatasetBase(Dataset, ABC):
    """
    The items and users are reindexed, because torch.nn.Embeddings
    and torch.Dataset requires integer indexes in 0...N. This class
    obtains indexes from keyword arguments (`item_id2index` and
    `user_id2index`) if specified. You probably want to do it,
    when usind different datasets obtained from one source.
    Otherwise, create new indexer.

    :param log: pySpark DataFrame with interaction log
    :param Indexer item_id2index: (Optinal) indexer for items
    :param Indexer users_id2index: (Optinal) indexer for users
    :param str padding_id: ID for padding item
    :param str unknown_id: ID for previously unseen items or users.
    :param int min_item_count: if item appears in logs less than `min_item_count`
                                   times, it will be indexed as "unknown item".
    """

    def __init__(
        self,
        log,
        item_indexer: Indexer = None,
        user_indexer: Indexer = None,
        padding_id=-1,
        unknown_id=-2,
    ):
        super().__init__()
        self._log = log
        if item_indexer:
            self._item_indexer = item_indexer
        else:
            self._item_indexer = Indexer(pad_id=padding_id, unk_id=unknown_id)

        if user_indexer:
            self._user_indexer = user_indexer
        else:
            # users always receive indexes
            self._user_indexer = Indexer(pad_id=padding_id, unk_id=unknown_id)

    @property
    def n_items(self):
        return self._item_indexer.n_objs

    @property
    def item_id2index(self):
        return self._item_indexer.to_dict()

    @property
    def n_users(self):
        return self._user_indexer.n_objs

    @property
    def user_id2index(self):
        return self._user_indexer.to_dict()

    @property
    def users(self):
        return self._users

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        if type(idx) is list:
            return self.__getitems__(idx)
        return self.__getitems__([idx])

    @abstractmethod
    def apply_scoring(self, score_df):
        """
        Apply scoring to the dataset. The scoring is applied to the
        dataset in place. The scoring dataframe must contain the
        following columns:
        * `user_idx` | int | user index
        * `item_idx` | int | item index
        * `iter`     | int | interaction number
        * `response_proba`    | float | score of the recommendation
        """
        pass

    @abstractmethod
    def _get_log_for_users(self, user_idxs):
        """
        Given a list of user indexes, return a list of rows cntaining
        aggregated data for given users. Each row corresponds to one interaction,
        i.e. each pair ('user_idx', '__iter') supposed to be unique in log.

        The rows must be sorted by ('user_idx', '__iter'). This will allow avoid
        using costlu groupby in further code.
        """
        pass

    def __getitems__(self, user_idxs: list) -> dict:
        """Get data points for users with ids in `user_idx`"""
        users_log = self._get_log_for_users(user_idxs)
        batch = []
        curr_user_log = []
        prev_user = -1
        # will it be faster if implemented with pandas udfs?
        for row in users_log:
            if prev_user == row["user_idx"] and prev_user != -1:
                curr_user_log.append(row)
            else:
                user_index = self._user_indexer.index_np(prev_user)
                batch.append(self._user_log_to_datapoint(curr_user_log, user_index))
                prev_user = row["user_idx"]
                curr_user_log = [row]
        return batch

    def get_empty_data(self, slate_size=10):
        """Empty datapont"""
        # everythoing is masked, hence it won't impact training nor metric computation
        return {
            "item_indexes": np.ones((1, slate_size), dtype=int),
            "user_index": 1,  # unknown index
            "slates_mask": np.zeros((1, slate_size), dtype=bool),
            "responses": np.zeros((1, slate_size), dtype=int),
            "length": 1,  # zero-length would cause problems with torch.nn.rnn_pad_sequences
            "slate_size": slate_size,
            "timestamps": np.ones((1, slate_size), dtype=int) * -(10**9),
        }

    def _user_log_to_datapoint(self, slates: list, user_index: int):
        """
        Gets one datapoint (a history of interactions for single user).
        In what follows, define:
            * R -- number of recommendations for this
            * S - slate size
            * Eu, Ei - embedding dim for users and items
        Datapoint is a dictionary with the following content:

        Item data:
            'item_indexes': np.array with shape (R, S). Same as previous,
                            but with indexes (0...N) instead if ids. Used to
                            index embeddings: nn.Embeddings nor scipy.sparse
                            can not be used with custom index.
        User data:
            'user_index': user index.
        Interaction data:
            'slate_mask': np.array with shape (R, S). True for recommended items,
                            False for placeholder.
            'responses': np.array with shape (R, S). Cell (i, j)
                            contains an id number of iteractions item
                            at j-th position of i-th slate.
            'length': int. R.
        """
        # Number of recommendations (R)
        R = len(slates)

        if R == 0:
            return self.get_empty_data()

        # Get the maximum slate size (S)
        S = max(len(s["item_idxs"]) for s in slates)

        # Prepare arrays to store the data
        item_idxs = np.zeros((R, S), dtype=object)
        slates_mask = np.zeros((R, S), dtype=bool)
        responses = np.zeros((R, S), dtype=int)
        timestamps = np.zeros((R, S), dtype=int)

        # Fill the data
        for i, slate in enumerate(slates):
            slate_size = len(slate["item_idxs"])
            item_idxs[i, :slate_size] = slate["item_idxs"]
            slates_mask[i, :slate_size] = [True] * slate_size
            responses[i, :slate_size] = slate["responses"]
            timestamps[i, :slate_size] = slate["__iter"] * slate_size

        # Create the output dictionary
        data_point = {
            "item_indexes": self._item_indexer.index_np(item_idxs),
            "user_index": user_index,
            "slates_mask": slates_mask,
            "responses": responses,
            "timestamps": timestamps,
            "length": R,
            "slate_size": S,
        }
        # print(data_point)
        return data_point


class RecommendationData(DatasetBase):
    """
    Recommednation dataset handler based on pySpark. Does not keep all the data in RAM.
    Recommendation data is initialized as spark DataFrame with required columns:
    * `user_idx`  | int | user identificator
    * `item_idx`  | int | item identificator
    * `__iter`    | int | timestamp

    In additional to required columns, the following columns are optional and
    used only in certain applications.
    * `response`        | int   | response, filled with 0s if not present
    * `response_proba`  | float | response probability, filled with 0.0 if not present
    * `slate_pos`       | int   | position of item in recommendation slate
    * `relevance`       | float | relevance of recommemded item in slate. This columns is used
                                  only sllate_pos is not present, and then slate_pos is
                                  assigned according to relevances.

    TODO: add embeddings.
    """

    def __init__(
        self,
        log: pyspark.sql.DataFrame,
        item_indexer=None,
        user_indexer=None,
        padding_id=-1,
        unknown_id=-2,
        min_item_count=1,
    ):
        """
        Initializes the dataset from `log` pyspark dataframe.
        """
        super().__init__(log, item_indexer, user_indexer, padding_id, unknown_id)
        if not item_indexer:
            self._item_indexer.update_from_iter(
                [
                    row["item_idx"]
                    for row in self._log.groupBy("item_idx")
                    .agg(sf.count("*").alias("count"))
                    .filter(sf.col("count") >= min_item_count)
                    .select("item_idx")
                    .collect()
                ]
            )

        # in _users only users which are actually present in data are stored, rather han all indexed users
        self._users = [
            row["user_idx"] for row in self._log.select("user_idx").distinct().collect()
        ]
        if not user_indexer:
            self._user_indexer.update_from_iter(self._users)

    def _get_log_for_users(self, user_idxs: list):
        users_log = self._log.filter(sf.col("user_idx").isin(user_idxs))
        users_log = (
            (
                users_log.groupBy("user_idx", "__iter").agg(
                    sf.collect_list("item_idx").alias("item_idxs"),
                    sf.collect_list("response").alias("responses"),
                )
            )
            .orderBy("user_idx", "__iter")
            .collect()
        )
        return users_log

    def apply_scoring(self, score_df):
        raise NotImplementedError("Not implemented yet")


class PandasRecommendationData(DatasetBase):
    """Temporary Dataset, normally used inside pandas user-defined functions"""

    def __init__(
        self,
        log: pd.DataFrame,
        item_indexer: Indexer = None,
        user_indexer: Indexer = None,
        padding_id=None,
        min_item_count=1,
    ):
        """
        Initializes the dataset from `log` pandas dataframe.
        """
        super().__init__(log, item_indexer, user_indexer)
        if not item_indexer:
            self._item_indexer.update_from_iter(self._log.item_idx.unique())
        if not item_indexer:
            self._user_indexer.update_from_iter(self._log.user_idx.unique())
        self._users = self._log.user_idx.unique()

    def _get_log_for_users(self, user_idxs: list):
        """
        Faster version of `__getitem__` for batched input. DataLoaders in torch >=2
        automatically use this method if it's implemented. In earlier versions of pytorch
        a custom sampler is required.

        :param user_idxs: list of user indexes. If None, all users are returned.
        """
        users_log = self._log
        if user_idxs:
            users_log = self._log[self._log["user_idx"].isin(user_idxs)]
        users_log = (
            users_log.groupby(["user_idx", "__iter"])
            .agg(
                item_idx=pd.NamedAgg(column="item_idx", aggfunc=list),
                response=pd.NamedAgg(column="response", aggfunc=list),
            )
            .rename(columns={"user_idx": "user_idxs"})
            .reset_index()
        )
        # convert to list of rows to match spark .collect() format:
        users_log = users_log.to_dict(orient="records")
        return users_log

    def apply_scoring(self, score_df):
        self.log["item_index"] = self.item_indexer.index_np(self.log("item_idx"))
        self.log["user_index"] = self.item_indexer.index_np(self.log("user_idx"))
        self.log = self.log.merge(score_df, on=["user_index", "item_index"], how="left")
        self.log["response_proba"] = self.log["score"]
        self.log.drop(columns=["score"], inplace=True)
