# Simulator

Simulator is framework for training and evaluating recommendation algorithms on real or synthetic data. Framework is based on pyspark library to work with big data.
As a part of simulation process the framework incorporates data generators, response functions and other tools, that can provide flexible usage of simulator.

# Table of contents

* [Installation](#installation)
* [Quickstart](#quickstart)
* [Examples](#examples)
* [Build from sources](#build-from-sources)
* [Building documentation](#compile-documentation)
* [Running tests](#tests)

## Installation

```bash
pip install sim4rec
```

If the installation takes too long, try
```bash
pip install sim4rec --use-deprecated=legacy-resolver
```

To install dependencies with poetry run

```bash
pip install --upgrade pip wheel poetry lightfm==1.17
poetry install
```

## Quickstart

The following example shows how to use simulator to train model iteratively by refitting recommendation algorithm on the new upcoming history log

```python
import numpy as np
import pandas as pd

import pyspark.sql.types as st
from pyspark.ml import PipelineModel
from sim4rec.utils import pandas_to_spark
from sim4rec.modules import RealDataGenerator, Simulator
from sim4rec.response import NoiseResponse, BernoulliResponse

from ucb import UCB
from replay.metrics import NDCG

LOG_SCHEMA = st.StructType([
    st.StructField('user_idx', st.LongType(), True),
    st.StructField('item_idx', st.LongType(), True),
    st.StructField('relevance', st.DoubleType(), False),
    st.StructField('response', st.IntegerType(), False)
])

users_df = pd.DataFrame(
    data=np.random.normal(0, 1, size=(100, 15)),
    columns=[f'user_attr_{i}' for i in range(15)]
)
items_df = pd.DataFrame(
    data=np.random.normal(1, 1, size=(30, 10)),
    columns=[f'item_attr_{i}' for i in range(10)]
)
history_df = pandas_to_spark(pd.DataFrame({
    'user_idx' : [1, 10, 10, 50],
    'item_idx' : [4, 25, 26, 25],
    'relevance' : [1.0, 0.0, 1.0, 1.0],
    'response' : [1, 0, 1, 1]
}), schema=LOG_SCHEMA)

users_df['user_idx'] = np.arange(len(users_df))
items_df['item_idx'] = np.arange(len(items_df))

users_df = pandas_to_spark(users_df)
items_df = pandas_to_spark(items_df)

user_gen = RealDataGenerator(label='users_real')
item_gen = RealDataGenerator(label='items_real')
user_gen.fit(users_df)
item_gen.fit(items_df)
_ = user_gen.generate(100)
_ = item_gen.generate(30)

sim = Simulator(
    user_gen=user_gen,
    item_gen=item_gen,
    data_dir='test_simulator',
    user_key_col='user_idx',
    item_key_col='item_idx',
    log_df=history_df
)

noise_resp = NoiseResponse(mu=0.5, sigma=0.2, outputCol='__noise')
br = BernoulliResponse(inputCol='__noise', outputCol='response')
pipeline = PipelineModel(stages=[noise_resp, br])

model = UCB()
model.fit(log=history_df)

ndcg = NDCG()

train_ndcg = []
for i in range(10):
    users = sim.sample_users(0.1).cache()

    recs = model.predict(log=sim.log, k=5, users=users, items=items_df, filter_seen_items=True).cache()

    true_resp = sim.sample_responses(
        recs_df=recs,
        user_features=users,
        item_features=items_df,
        action_models=pipeline
    ).select('user_idx', 'item_idx', 'relevance', 'response').cache()

    sim.update_log(true_resp, iteration=i)

    train_ndcg.append(ndcg(recs, true_resp.filter(true_resp['response'] >= 1), 5))

    model.fit(sim.log.drop('relevance').withColumnRenamed('response', 'relevance'))

    users.unpersist()
    recs.unpersist()
    true_resp.unpersist()

print(train_ndcg)

```

## Examples

You can find useful examples in `notebooks` folder, which demonstrates how to use synthetic data generators, composite generators, evaluate scores of the generators, iteratively refit recommendation algorithm, use response functions and more.

## Build from sources

```bash
poetry build
pip install ./dist/sim4rec-0.0.1-py3-none-any.whl
```

## Compile documentation

```bash
cd docs
make clean && make html
```

## Tests

For tests the pytest python library is used and to run tests for all modules you can run the following command from repository root directory

```bash
pytest
```

## Licence
Sim4Rec is distributed under the [Apache License Version 2.0](https://github.com/sb-ai-lab/Sim4Rec/blob/main/LICENSE), 
nevertheless the SDV package, imported by the Sim4Rec for synthetic data generation,
is distributed under [Business Source License (BSL) 1.1](https://github.com/sdv-dev/SDV/blob/master/LICENSE).

Synthetic tabular data generation not a purpose of the Sit4Rec framework. 
The Sim4Rec offers an API and wrappers to run simulation with synthetic data, but the method of synthetic data generation is determined by the user. 
SDV package is imported for illustration purposes and may be replaced by another synthetic data generation solution.  

Thus, synthetic data generation functional and quality evaluation with SDV library, 
namely the `SDVDataGenerator` from [generator.py](sim4rec/modules/generator.py) and `evaluate_synthetic` from [evaluation.py](sim4rec/modules/evaluation.py) 
should be used for non-production purposes only according to the SDV License. 
