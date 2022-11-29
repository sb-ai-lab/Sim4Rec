from simulator.response import ActionModelTransformer
from pyspark.ml.param.shared import HasInputCol
from pyspark.sql import DataFrame
import pyspark.sql.functions as sf
from pyspark.ml.param.shared import Params, Param, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

class HasMultiplierValue(Params):
    multiplierValue = Param(
        Params._dummy(),
        "multiplierValue",
        "Multiplier value parameter",
        typeConverter=TypeConverters.toFloat
    )

    def __init__(self):
        super().__init__()

    def setMultiplierValue(self, value):
        return self._set(multiplierValue=value)

    def getMultiplierValue(self):
        return self.getOrDefault(self.multiplierValue)

class ModelCalibration(ActionModelTransformer,
                       HasInputCol,
                       HasMultiplierValue,
                       DefaultParamsReadable,
                       DefaultParamsWritable):
    def __init__(
        self,
        value : float = 0.0,
        inputCol : str = None,
        outputCol : str = None
    ):
        """
        Multiplies response function output by the chosen value.
        :param value: Multiplier value
        :param outputCol: Output column name
        """

        super().__init__(outputCol=outputCol)
        
        self._set(inputCol=inputCol)
        self._set(multiplierValue=value)

    def _transform(
        self,
        df : DataFrame
    ):
        value = self.getMultiplierValue()
        inputCol = self.getInputCol()
        outputColumn = self.getOutputCol()
        

        return df.withColumn(outputColumn, sf.lit(value)*sf.col(inputCol))
