import pathlib
from abc import ABC
from typing import Tuple, Union, Optional

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.ml import Transformer, PipelineModel

from replay.session_handler import State
from sim4rec.modules.generator import GeneratorBase


# pylint: disable=too-many-instance-attributes
class Simulator(ABC):
    """
    Simulator for recommendation systems, which uses the users
    and items data passed to it, to simulate the users responses
    to recommended items
    """

    ITER_COLUMN = '__iter'
    DEFAULT_LOG_FILENAME = 'log.parquet'

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        user_gen : GeneratorBase,
        item_gen : GeneratorBase,
        data_dir : str,
        log_df : DataFrame = None,
        user_key_col : str = 'user_idx',
        item_key_col : str = 'item_idx',
        spark_session : SparkSession = None
    ):
        """
        :param user_gen: Users data generator instance
        :param item_gen: Items data generator instance
        :param log_df: The history log with user-item pairs with other
            necessary fields. During the simulation the results will be
            appended to this log on update_log() call, defaults to None
        :param user_key_col: User identifier column name, defaults
            to 'user_idx'
        :param item_key_col: Item identifier column name, defaults
            to 'item_idx'
        :param data_dir: Directory name to save simulator data
        :param spark_session: Spark session to use, defaults to None
        """

        self._spark = spark_session if spark_session is not None else State().session

        self._user_key_col = user_key_col
        self._item_key_col = item_key_col
        self._user_gen = user_gen
        self._item_gen = item_gen

        if data_dir is None:
            raise ValueError('Pass directory name as `data_dir` parameter')

        self._data_dir = data_dir
        pathlib.Path(self._data_dir).mkdir(parents=True, exist_ok=False)

        self._log_filename = self.DEFAULT_LOG_FILENAME

        self._log = None
        self._log_schema = None
        if log_df is not None:
            self.update_log(log_df, iteration='start')

    @property
    def log(self):
        """
        Returns log
        """
        return self._log

    @property
    def data_dir(self):
        """
        Returns directory with saved simulator data
        """
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value):
        self._data_dir = value

    @property
    def log_filename(self):
        """
        Returns name of log file
        """
        return self._log_filename

    @log_filename.setter
    def log_filename(self, value):
        self._log_filename = value

    def clear_log(
        self
    ) -> None:
        """
        Clears the log
        """

        self._log = None
        self._log_schema = None

    @staticmethod
    def _check_names_and_types(df1_schema, df2_schema):
        """
        Check if names of columns and their types are equal for two schema.
        `Nullable` parameter is not compared.

        """
        df1_schema_s = sorted(
            [(x.name, x.dataType) for x in df1_schema],
            key=lambda x: (x[0], x[1])
        )
        df2_schema_s = sorted(
            [(x.name, x.dataType) for x in df2_schema],
            key=lambda x: (x[0], x[1])
        )
        names_diff = set(df1_schema_s).symmetric_difference(set(df2_schema_s))

        if names_diff:
            raise ValueError(
                f'Columns of two dataframes are different.\nDifferences: \n'
                f'In the first dataframe:\n'
                f'{[name_type for name_type in df1_schema_s if name_type in names_diff]}\n'
                f'In the second dataframe:\n'
                f'{[name_type for name_type in df2_schema_s if name_type in names_diff]}'
            )

    def update_log(
        self,
        log : DataFrame,
        iteration : Union[int, str]
    ) -> None:
        """
        Appends the passed log to the existing one

        :param log: The log with user-item pairs with their respective
            necessary fields. If there was no log before this: remembers
            the log schema, to which the future logs will be compared.
            To reset current log and the schema see clear_log()
        :param iteration: Iteration label or index
        """

        if self._log_schema is None:
            self._log_schema = log.schema.fields
        else:
            self._check_names_and_types(self._log_schema, log.schema)

        write_path = str(
            pathlib.Path(self._data_dir)
            .joinpath(f'{self.log_filename}/{self.ITER_COLUMN}={iteration}')
        )
        log.write.parquet(write_path)

        read_path = str(pathlib.Path(self._data_dir).joinpath(f'{self.log_filename}'))
        self._log = self._spark.read.parquet(read_path)

    def sample_users(
        self,
        frac_users : float
    ) -> DataFrame:
        """
        Samples a fraction of random users

        :param frac_users: Fractions of users to sample from user generator
        :returns: Sampled users dataframe
        """

        return self._user_gen.sample(frac_users)

    def sample_items(
        self,
        frac_items : float
    ) -> DataFrame:
        """
        Samples a fraction of random items

        :param frac_items: Fractions of items to sample from item generator
        :returns: Sampled users dataframe
        """

        return self._item_gen.sample(frac_items)

    def get_log(
        self,
        user_df : DataFrame
    ) -> DataFrame:
        """
        Returns log for users listed in passed users' dataframe

        :param user_df: Dataframe with user identifiers to get log for
        :return: Users' history log. Will return None, if there is no log data
        """

        if self.log is not None:
            return self.log.join(
                user_df, on=self._user_key_col, how='leftsemi'
            )

        return None

    def get_user_items(
        self,
        user_df : DataFrame,
        selector : Transformer
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Froms candidate pairs to pass to the recommendation algorithm based
        on the provided users

        :param user_df: Users dataframe with features and identifiers
        :param selector: Transformer to use for creating user-item pairs
        :returns: Tuple of user-item pairs and log dataframes which will
            be used by recommendation algorithm. Will return None as a log,
            if there is no log data
        """

        log = self.get_log(user_df)
        pairs = selector.transform(user_df)

        return pairs, log

    def sample_responses(
        self,
        recs_df : DataFrame,
        action_models : PipelineModel,
        user_features : Optional[DataFrame] = None,
        item_features : Optional[DataFrame] = None
    ) -> DataFrame:
        """
        Simulates the actions users took on their recommended items

        :param recs_df: Dataframe with recommendations. Must contain
            user's and item's identifier columns. Other columns will
            be ignored
        :param user_features: Users dataframe with features and identifiers,
                              can be None
        :param item_features: Items dataframe with features and identifiers,
                              can be None
        :param action_models: Spark pipeline to evaluate responses
        :returns: DataFrame with user-item pairs and the respective actions
        """

        if user_features is not None:
            recs_df = recs_df.join(user_features, self._user_key_col, 'left')

        if item_features is not None:
            recs_df = recs_df.join(item_features, self._item_key_col, 'left')

        return action_models.transform(recs_df)
