Modules
=======

.. automodule:: sim4rec.modules


Generators
__________

Generators serves for generating synthetic either real data for simulation process.
All of the generators are derived from ``GeneratorBase`` base class and to implement
your own generator you must inherit from it. Basicly, the generator fits from a provided
dataset (in case of real generator it just remembers it), than it creates a population to
sample from with a number of rows by calling the ``generate()`` method and samples from
this those population with ``sample()`` method. Note, that ``sample()`` takes the fraction
of population size.

If a user is interested in using multiple generators at ones (e.g. modelling multiple groups
of users or mixing results from different generating models) that it will be useful to look
at ``CompositeGenerator`` which can handle a list of generators has a proportion mixing parameter
which controls the weights of particular generators at sampling and generating time.

.. autoclass:: sim4rec.modules.GeneratorBase
    :members:

.. autoclass:: sim4rec.modules.RealDataGenerator
    :members:

.. autoclass:: sim4rec.modules.SDVDataGenerator
    :members:

.. autoclass:: sim4rec.modules.CompositeGenerator
    :members:


Embeddings
__________

Embeddings can be utilized in case of high dimensional data or high data sparsity and it should be
applied before performing the main simulator pipeline. Here the autoencoder estimator and transformer
are implemented in the chance of the existing spark methods are not enough for your propose. The usage
example can be found in `notebooks` directory.

.. autoclass:: sim4rec.modules.EncoderEstimator
    :members:

.. autoclass:: sim4rec.modules.EncoderTransformer
    :members:


Items selectors
_______________

Those spark transformers are used to assign items to given users while making the candidate
pairs for prediction by recommendation system. It is optional to use selector in your pipelines
if a recommendation algorithm can recommend any item with no restrictions. The one should
implement own selector in case of custom logic of items selection for a certain users. Selector
could implement some business rules (e.g. some items are not available for a user), be a simple
recommendation model, generating candidates, or create user-specific items (e.g. price offers)
online. To implement your custom selector, you can derive from a ``ItemSelectionEstimator`` base
class, to implement some pre-calculation logic, and ``ItemSelectionTransformer`` to perform pairs
creation. Both classes are inherited from spark's Estimator and Transformer classes and to define
fit() and transform() methods the one can just overwrite ``_fit()`` and ``_transform()``.

.. autoclass:: sim4rec.modules.ItemSelectionEstimator
    :members:

.. autoclass:: sim4rec.modules.ItemSelectionTransformer
    :members:

.. autoclass:: sim4rec.modules.CrossJoinItemEstimator
    :members:
    :private-members: _fit

.. autoclass:: sim4rec.modules.CrossJoinItemTransformer
    :members:
    :private-members: _transform


Simulator
_________

The simulator class provides a way to handle the simulation process by connecting different
parts of the module such as generatos and response pipelines and saving the results to a
given directory. The simulation process consists of the following steps:

- Sampling random real or synthetic users
- Creation of candidate user-item pairs for a recommendation algorithm
- Prediction by a recommendation system
- Evaluating respones on a given recommendations
- Updating the history log
- Metrics evaluation
- Refitting the recommendation model with a new data

Some of the steps can be skipped depending on the task your perform. For example you don't need
a second step if your algorithm dont use user-item pairs as an input or you don't need to refit
the model if you want just to evaluate it on some data. For more usage please refer to examples

.. autoclass:: sim4rec.modules.Simulator
    :members:


Evaluation
__________

.. autoclass:: sim4rec.modules.EvaluateMetrics
    :members:
    :special-members: __call__

.. autofunction:: sim4rec.modules.evaluate_synthetic

.. autofunction:: sim4rec.modules.ks_test

.. autofunction:: sim4rec.modules.kl_divergence
