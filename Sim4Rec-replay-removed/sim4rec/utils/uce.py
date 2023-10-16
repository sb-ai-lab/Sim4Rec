# Utility class extensions
# pylint: disable=no-member,unused-argument
import pickle
import pyspark.sql.functions as sf
import pyspark.sql.types as st

from pyspark.sql import DataFrame
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


from pyspark.ml.param.shared import Params, Param, TypeConverters
from pyspark import keyword_only


class NotFittedError(Exception):
    # pylint: disable=missing-class-docstring
    pass


class EmptyDataFrameError(Exception):
    # pylint: disable=missing-class-docstring
    pass


# pylint: disable=too-many-ancestors
class VectorElementExtractor(Transformer,
                             HasInputCol, HasOutputCol,
                             DefaultParamsReadable, DefaultParamsWritable):
    """
    Extracts element at index from array column
    """

    index = Param(
        Params._dummy(),
        "index",
        "Array index to extract",
        typeConverter=TypeConverters.toInt
    )

    def setIndex(self, value):
        """
        Sets index to a certain value
        :param value: Value to set index of an element
        """
        return self._set(index=value)

    def getIndex(self):
        """
        Returns index of element
        """
        return self.getOrDefault(self.index)

    @keyword_only
    def __init__(
        self,
        inputCol : str = None,
        outputCol : str = None,
        index : int = None
    ):
        """
        :param inputCol: Input column with array
        :param outputCol: Output column name
        :param index: Index of an element within array
        """
        super().__init__()
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(
        self,
        inputCol : str = None,
        outputCol : str = None,
        index : int = None
    ):
        """
        Sets parameters for extractor
        """
        return self._set(**self._input_kwargs)

    def _transform(
        self,
        dataset : DataFrame
    ):
        index = self.getIndex()

        el_udf = sf.udf(
            lambda x : float(x[index]), st.DoubleType()
        )

        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()

        return dataset.withColumn(outputCol, el_udf(inputCol))


def save(obj : object, filename : str):
    """
    Saves an object to pickle dump
    :param obj: Instance
    :param filename: File name of a dump
    """

    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load(filename : str):
    """
    Loads a pickle dump from file
    :param filename: File name of a dump
    :return: Read instance
    """

    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj
