Concepts in Machine Learning
============================

- :code:`batch size`: If we are using stochastic gradient descent (SGD) to update the parameters of the model, the batch size is the number of data points in each batch. We will update the parameters after calculating the loss function on each batch.

- :code:`epoch`: One epoch means that we have passed through the **entire training dataset** once.

- "Irrespective of whether the data is huge or not, cross validation is a must when building any model. If this takes more time than an end consumer is willing to wait, you may need to reset their expectations, or get faster hardware/software to build the model; but do not skip cross validation. Plotting learning curves and cross-validation are effective steps to help guide us so we recognize and correct mistakes earlier in the process. I've experienced instances when a simple train-test set does not reveal any problems until I run cross-fold validations and find a large variance in the performance of the algorithm on different folds." (from `this link <https://datascience.stackexchange.com/questions/13901/machine-learning-best-practices-for-big-dataset>`__)

- 