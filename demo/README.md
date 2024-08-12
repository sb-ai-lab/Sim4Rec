# Case studies for demo
These are the cases for the demo track RECSYS 2024 conference:
1. Synthetic data generation

Generation of synthetic users based on real data.
2. Long-term RS performance evaluation

Simple simulation pipeline for a long-term performance evaluation of recommender system. 

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

## Synthetic data generation pipeline
1. Fit non-negative ALS for users embeddings
2. Get users features
3. Generate users features with CopulaGAN
4. Evaluate generator

## Long-term RS performance evaluation pipeline
1. Choose users
2. Initialize and fit the recommender model
3. Initialize and fit the response function
4. Initialize simulator
5. Run simulation cycle: 
 - Choose users
 - Get recommendations from the recommender system
 - Get responses from the response function
 - Update the interaction history
 - Measure quality
 - Refit the model
4. Get final prediction 
5. Measure quality
