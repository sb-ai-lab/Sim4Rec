# Case studies for demo
Cases for the demo:
1. [Synthetic data generation](https://github.com/sb-ai-lab/Sim4Rec/blob/main/demo/synthetic_data_generation.ipynb)

Generation of synthetic users based on real data.

2. [Long-term RS performance evaluation](https://github.com/sb-ai-lab/Sim4Rec/blob/main/demo/rs_performance_evaluation.ipynb)

Simple simulation pipeline for long-term performance evaluation of recommender system. 

# Table of contents

* [Installation](#installation)
* [Synthetic data generation pipeline](#synthetic-data-generation-pipeline)
* [Long-term RS performance evaluation pipeline](#long-term-RS-performance-evaluation-pipeline)

## Installation

Install dependencies with poetry run

```bash
pip install --upgrade pip wheel poetry
poetry install
```

Please install `rs_datasets` to import MovieLens dataset for the synthetic_data_generation.ipynb
```bash
pip install rs_datasets
```

## Synthetic data generation pipeline
1. Fit non-negative ALS to real data containing user interactions
2. Obtain vector representations of real users
3. Fit CopulaGAN to non-negative ALS embeddings of real users
4. Generate synthetic user feature vectors with CopulaGAN
5. Evaluate the quality of the synthetic user profiles

## Long-term RS performance evaluation pipeline
1. Before running the simulation cycle:
 - Initialize and fit recommender model to historical data
 - Construct response function pipeline and fit it to items
 - Initialize simulator
2. Run the simulation cycle: 
 - Sample users using a simulator
 - Get recommendations for sampled users from the recommender system
 - Get responses to recommended items from the response function
 - Update user interaction history
 - Measure the quality of the recommender system
 - Refit the recommender model
3. After running the simulation cycle:
 - Get recommendations for all users from the recommender system
 - Get responses to recommended items from the response function
 - Measure the quality of the recommender system trained in the simulation cycle
 
