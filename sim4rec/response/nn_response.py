import os
import pickle
import pyspark.sql.functions as sf

from .response import ActionModelEstimator, ActionModelTransformer
from .nn_utils.models import ResponseModel
from .nn_utils.embeddings import IndexEmbedding
from .nn_utils.datasets import (
    RecommendationData,
    # PandasRecommendationData,
)

from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
)

# move this to simulator core(?)
SIM_LOG_SCHEMA = StructType(
    [
        StructField("user_idx", IntegerType(), True),
        StructField("item_idx", IntegerType(), True),
        StructField("relevance", DoubleType(), True),
        StructField("response_proba", DoubleType(), True),
        StructField("response", IntegerType(), True),
        StructField("__iter", IntegerType(), True),
    ]
)
SIM_LOG_COLS = [field.name for field in SIM_LOG_SCHEMA.fields]


class NNResponseTransformer(ActionModelTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.hist_data = None
        for param, value in kwargs.items():
            setattr(self, param, value)

    @classmethod
    def load(cls, checkpoint_dir):
        with open(os.path.join(checkpoint_dir, "_params.pkl"), "rb") as f:
            params_dict = pickle.load(f)
        params_dict["backbone_response_model"] = ResponseModel.load(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "_item_indexer.pkl"), "rb") as f:
            params_dict["item_indexer"] = pickle.load(f)
        with open(os.path.join(checkpoint_dir, "_user_indexer.pkl"), "rb") as f:
            params_dict["user_indexer"] = pickle.load(f)
        return cls(**params_dict)

    def save(self, path):
        """Save model at given path."""
        os.makedirs(path)
        self.backbone_response_model.dump(path)
        with open(os.path.join(path, "_item_indexer.pkl"), "wb") as f:
            pickle.dump(self.item_indexer, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, "_user_indexer.pkl"), "wb") as f:
            pickle.dump(self.user_indexer, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, "_params.pkl"), "wb") as f:
            pickle.dump(
                {
                    "outputCol": self.outputCol,
                    "log_dir": self.log_dir,
                    "hist_data_dir": self.hist_data_dir,
                },
                f,
                pickle.HIGHEST_PROTOCOL,
            )

    def _transform(self, new_recs):
        """
        Predict responses for given dataframe with recommendations.

        :param dataframe: new recommendations.
        """

        def predict_udf(df):
            # if not do this, something unstable happens to the Method Resolution Order
            from .nn_utils.datasets import PandasRecommendationData

            dataset = PandasRecommendationData(
                log=df,
                item_indexer=self.item_indexer,
                user_indexer=self.user_indexer,
            )

            # replacing clicks in datset with predicted
            dataset = self.backbone_response_model.transform(dataset=dataset)

            return dataset._log[SIM_LOG_COLS]

        spark = new_recs.sql_ctx.sparkSession

        # read the historical data
        hist_data = spark.read.schema(SIM_LOG_SCHEMA).parquet(self.hist_data_dir)
        if not hist_data:
            print("Warning: the historical data is empty")
            hist_data = spark.createDataFrame([], schema=SIM_LOG_SCHEMA)
        # filter users whom we don't need
        hist_data = hist_data.join(new_recs, on="user_idx", how="semi")

        # read the updated simulator log
        simlog = spark.read.schema(SIM_LOG_SCHEMA).parquet(self.log_dir)
        if not simlog:
            print("Warning: the simulator log is empty")
            simlog = spark.createDataFrame([], schema=SIM_LOG_SCHEMA)
        # filter users whom we don't need
        simlog = simlog.join(new_recs, on="user_idx", how="semi")

        NEW_ITER_NO = 9999999

        # since all the historical records are older than simulated by design,
        # and new slates are newer than simulated, i can simply concat it
        combined_data = hist_data.unionByName(simlog).unionByName(
            new_recs.withColumn("response_proba", sf.lit(0.0))
            .withColumn("response", sf.lit(0.0))
            .withColumn(
                "__iter",
                sf.lit(
                    NEW_ITER_NO
                ),  # this is just a large number, TODO: add correct "__iter" field to sim4rec.sample_responses to avoid this constants
            )
        )

        # not very optimal way, it makes one worker to
        # operate with one user, discarding batched computations inside torch
        groupping_column = "user_idx"
        result_df = combined_data.groupby(groupping_column).applyInPandas(
            predict_udf, SIM_LOG_SCHEMA
        )
        filtered_df = result_df.filter(sf.col("__iter") == NEW_ITER_NO)
        return filtered_df.select(new_recs.columns + [self.outputCol])


class NNResponseEstimator(ActionModelEstimator):
    def __init__(
        self,
        log_dir: str,
        model_name: str,
        hist_data_dir=None,
        val_data_dir=None,
        outputCol: str = "response_proba",
        **kwargs,
    ):
        """
        :param log_dir: The directory containing simulation logs.
        :param model_name: Backbone model name.
        :param hist_data_dir: (Optional) Spark DataFrame with historical data.
        :param val_data_dir: (Optional) Spark DataFrame with validation data.
                            TODO: split automatically.
        :param outputCol: Output column for MLLib pipeline.

        """
        self.fit_params = kwargs
        self.outputCol = outputCol

        # sim log is not loaded immideately, because
        # it can be not created when the response model is initialized
        self.log_dir = log_dir
        self.hist_data_dir = hist_data_dir
        self.val_data_dir = val_data_dir

        # create new model
        self.item_indexer = self.user_indexer = None
        self.model_name = model_name
        self.backbone_response_model = None

    def _fit(self, train_data):
        """
        Fits the model on given data.

        :param DataFrame train_data: Data to train on
        """
        train_dataset = RecommendationData(
            log=train_data,
            item_indexer=self.item_indexer,
            user_indexer=self.user_indexer,
        )
        self.item_indexer = train_dataset._item_indexer
        self.user_indexer = train_dataset._user_indexer
        val_dataset = RecommendationData(
            log=train_data.sql_ctx.sparkSession.read.parquet(self.val_data_dir),
            item_indexer=self.item_indexer,
            user_indexer=self.user_indexer,
        )
        n_items = train_dataset.n_items
        backbone_response_model = ResponseModel(
            self.model_name, IndexEmbedding(n_items)
        )
        backbone_response_model.fit(
            train_dataset, val_data=val_dataset, **self.fit_params
        )
        return NNResponseTransformer(
            backbone_response_model=backbone_response_model,
            item_indexer=self.item_indexer,
            user_indexer=self.user_indexer,
            hist_data_dir=self.hist_data_dir,
            log_dir=self.log_dir,
            outputCol=self.outputCol,
        )
