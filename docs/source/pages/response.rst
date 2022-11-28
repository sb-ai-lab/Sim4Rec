Response functions
==================

.. automodule:: simulator.response


Response functions are used to model users behaviour on items to simulate any
kind of reaction that user perform. For example it can be binary classification
model that determines whether user clicked on a item, or rating, that user gave
to an item.


Base classes
____________

All of the existing response functions are made on a top of underlying base classes
``ActionModelEstimator`` and ``ActionModelTransformer`` which follows the logic of
spark's Estimator and Transformer classes. To implement custom response function
the one can inherit from base classes:

* ``ActionModelEstimator`` if any learning logic is necessary
* ``ActionModelTransformer`` for performing model infering

Base classes are inherited from spark's Estimator and Transformer and to define fit()
or transform() logic the one should overwrite ``_fit()`` and ``_transform()`` respectively.
Note, that those base classes are useful to implement your own response function, but are not
necessary, and to create a response pipeline any proper spark's estimators/transformers can be used

.. autoclass:: simulator.response.ActionModelEstimator
    :members:

.. autoclass:: simulator.response.ActionModelTransformer
    :members:


Response functions
__________________

.. autoclass:: simulator.response.ConstantResponse
    :members:

.. autoclass:: simulator.response.NoiseResponse
    :members:

.. autoclass:: simulator.response.CosineSimilatiry
    :members:

.. autoclass:: simulator.response.BernoulliResponse
    :members:

.. autoclass:: simulator.response.ParametricResponseFunction
    :members:
