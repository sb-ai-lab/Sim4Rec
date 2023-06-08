from pyspark.sql import functions as sf
from pyspark.sql import DataFrame
from pyspark.ml import Transformer, Estimator
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import keyword_only
from sim4rec.params import HasUserKeyColumn, HasItemKeyColumn, HasSeed, HasSeedSequence


class ItemSelectionEstimator(Estimator,
                             HasUserKeyColumn,
                             HasItemKeyColumn,
                             DefaultParamsReadable,
                             DefaultParamsWritable):
    @keyword_only
    def __init__(
        self,
        userKeyColumn : str = None,
        itemKeyColumn : str = None
    ):
        """
        Base class for item selection estimator
        """

        super().__init__()
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(
        self,
        userKeyColumn : str = None,
        itemKeyColumn : str = None
    ):
        return self._set(**self._input_kwargs)


class ItemSelectionTransformer(Transformer,
                               HasUserKeyColumn,
                               HasItemKeyColumn,
                               DefaultParamsReadable,
                               DefaultParamsWritable):
    @keyword_only
    def __init__(
        self,
        userKeyColumn : str = None,
        itemKeyColumn : str = None
    ):
        """
        Base class for item selection transformer. transform()
        will be used to create user-item pairs
        """
        super().__init__()
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(
        self,
        userKeyColumn : str = None,
        itemKeyColumn : str = None
    ):
        self._set(**self._input_kwargs)


class CrossJoinItemEstimator(ItemSelectionEstimator, HasSeed):
    def __init__(
        self,
        k : int,
        userKeyColumn : str = None,
        itemKeyColumn : str = None,
        seed : int = None
    ):
        """
        Assigns k items for every user from random items subsample

        :param k: Number of items for every user
        :param userKeyColumn: Users identifier column, defaults to None
        :param itemKeyColumn: Items identifier column, defaults to None
        :param seed: Random state seed, defaults to None
        """

        super().__init__(userKeyColumn=userKeyColumn,
                         itemKeyColumn=itemKeyColumn)

        self.setSeed(seed)

        self._k = k

    def _fit(
        self,
        df : DataFrame
    ):
        """
        Fits estimator with items dataframe

        :param df: Items dataframe
        :returns: CrossJoinItemTransformer instance
        """

        userKeyColumn = self.getUserKeyColumn()
        itemKeyColumn = self.getItemKeyColumn()
        seed = self.getSeed()
        
        if itemKeyColumn not in df.columns:
            raise ValueError(f'Dataframe has no {itemKeyColumn} column')

        return CrossJoinItemTransformer(
            item_df=df,
            k=self._k,
            userKeyColumn=userKeyColumn,
            itemKeyColumn=itemKeyColumn,
            seed=seed
        )


class CrossJoinItemTransformer(ItemSelectionTransformer, HasSeedSequence):
    def __init__(
        self,
        item_df : DataFrame,
        k : int,
        userKeyColumn : str = None,
        itemKeyColumn : str = None,
        seed : int = None
    ):
        """
        Assigns k items for every user from random items subsample
        """

        super().__init__(userKeyColumn=userKeyColumn,
                         itemKeyColumn=itemKeyColumn)

        self.initSeedSequence(seed)

        self._item_df = item_df
        self._k = k

    def _transform(
        self,
        df : DataFrame
    ):
        """
        Takes a users dataframe and assings defined number of items

        :param df: Users dataframe
        :returns: Users cross join on random items subsample
        """

        userKeyColumn = self.getUserKeyColumn()
        itemKeyColumn = self.getItemKeyColumn()
        seed = self.getNextSeed()

        if userKeyColumn not in df.columns:
            raise ValueError(f'Dataframe has no {userKeyColumn} column')

        random_items = self._item_df.orderBy(sf.rand(seed=seed))\
                                    .limit(self._k)

        return df.select(userKeyColumn)\
                 .crossJoin(random_items.select(itemKeyColumn))
