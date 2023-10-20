Utils
=====

.. automodule:: sim4rec.utils


Dataframe convertation
______________________

.. autofunction:: sim4rec.utils.pandas_to_spark


Session
______________________

.. autofunction:: sim4rec.utils.session_handler.get_spark_session

.. autoclass:: sim4rec.utils.session_handler.State
    :members:

Exceptions
__________

.. autoclass:: sim4rec.utils.NotFittedError
    :members:

.. autoclass:: sim4rec.utils.EmptyDataFrameError
    :members:


Transformers
____________

.. autoclass:: sim4rec.utils.VectorElementExtractor
    :members:


File management
_______________

.. autofunction:: sim4rec.utils.save
.. autofunction:: sim4rec.utils.load
