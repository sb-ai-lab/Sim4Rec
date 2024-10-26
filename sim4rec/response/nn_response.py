import os
import pickle
import pyspark.sql.functions as sf

from .response import ActionModelEstimator, ActionModelTransformer
from .sim4rec_response_function.models import ResponseModel
from .sim4rec_response_function.embeddings import IndexEmbedding
from .sim4rec_response_function.datasets import (
    RecommendationData,
    PandasRecommendationData,
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
        for param, value in kwargs.items():
            print(param, value)
            setattr(self, param, value)

    @classmethod
    def load(cls, checkpoint_dir):
        with open(os.path.join(checkpoint_dir, "_params.pkl"), "rb") as f:
            params_dict = pickle.load(f)
        params_dict["backbone_response_model"] = ResponseModel.load(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "_item_indexer.pkl"), "rb") as f:
            params_dict["_item_indexer"] = pickle.load(f)
        with open(os.path.join(checkpoint_dir, "_user_indexer.pkl"), "rb") as f:
            params_dict["_user_indexer"] = pickle.load(f)
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
                {"outputCol": self.outputCol, "log_dir": self.log_dir},
                f,
                pickle.HIGHEST_PROTOCOL,
            )

    def _transform(self, new_recs):
        """
        Predict responses for given dataframe with recommendations.

        :param dataframe: new recommendations.

        """
        spark = new_recs.sql_ctx.sparkSession
        self.__simlog = spark.read.schema(SIM_LOG_SCHEMA).parquet(self.log_dir)
        if not self.__simlog:
            print("Warning: the simulator log is empty")
            self.__simlog = spark.createDataFrame([], schema=SIM_LOG_SCHEMA)

        def agg_func(df):
            return self._predict_udf(df)

        # TODO: add option to make bacthed predictions by ading
        # temorary "batch_id" column
        groupping_column = "user_idx"

        new_recs = new_recs.withColumn("__iter", sf.lit(999999999))
        new_recs = new_recs.withColumn("response", sf.lit(0.0))
        new_recs = new_recs.withColumn("response_proba", sf.lit(0.0))
        return (
            new_recs.groupby(groupping_column)
            .applyInPandas(agg_func, new_recs.schema)
            .show()
        )

    def _predict_udf(self, df):
        """
        Make predictions for given pandas DataFrame.
        :param df: pandas DataFrame.
        :return: pandas DataFrame with the same schema, but overwritten column 'respone_proba'.
        """

        # select only users whom we need
        # will this be fast enought, or better filter before?
        hist_data_selected_users = self.hist_data.join(
            self.__simlog, on="user_idx", how="inner"
        ).select(self.hist_data["*"])

        # assume that historical data interactions were BEFORE simulate
        previous_interactions = hist_data_selected_users.unionByName(self.__simlog)

        new_slates = PandasRecommendationData(
            df, item_indexer=self.item_indexer, user_indexer=self.user_indexer
        )

        # generating clicks
        predicted_clicks = self.backbone_response_model.transform(
            dataset=previous_interactions,
            new_slates=new_slates,
            method="autoregressive",
        )

        print(predicted_clicks)
        #### DEBUG I am here now
        # removing redundant columns
        predictions_clean = predicted_clicks.to_iteraction_table()[
            ["user_id", "item_id", "predicted_probs", "predicted_response"]
        ]
        predictions_clean["item_id"] = predictions_clean["item_id"].astype(int)
        predictions_clean["user_id"] = predictions_clean["user_id"].astype(int)
        predictions_clean.rename(
            columns={
                "user_id": "user_idx",
                "item_id": "item_idx",
                "predicted_probs": "response_proba",
                "predicted_response": "response",
            },
            inplace=True,
        )

        final = new_recs_data[["user_idx", "item_idx", "relevance"]].join(
            predictions_clean.set_index(["user_idx", "item_idx"]),
            on=["user_idx", "item_idx"],
            validate="one_to_one",
        )

        return final


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
        )
