# notebooks

This directory contains Jupyter notebooks that demonstrate how to use the `iQual` package.


* [Model with Single Annotation](Model-sklearn-SingleAnnotation.ipynb) - This notebook demonstrates how to construct a model for a single annotation task.

* [Model with Spacy](Model-SpaCy.ipynb) - This notebook demonstrates how to construct a model for a single annotation task using the [SpaCy](https://spacy.io/) library.

* [Model with Sentence-Transformers](Model-SentenceTransformers.ipynb) - This notebook demonstrates how to construct a model for a single annotation task using the [Sentence-Transformers](https://www.sbert.net/) library.

* [Model with Saved Dictionary](Model-SavedDictionary.ipynb) - This notebook demonstrates how to construct a model for a single annotation task using vectors from a saved dictionary. This can be useful if you have a large number of annotations and want to save time by not having to recompute the vectors for each annotation.

* [Model with Multiple Vectorizers](Model-MultipleVectorizers.ipynb) - This notebook demonstrates how to construct a model for a single annotation task using multiple vectorizers. This can be useful if you want to compare the performance of different vectorizers.

* [Model with Multiple Classifiers](Model-MultipleClassifiers.ipynb) - This notebook demonstrates how to construct a model for a single annotation task using multiple classifiers. This can be useful if you want to compare the performance of different classifiers.

* [Model with Multiple Annotations](Model-MultipleAnnotations.ipynb) - This notebook demonstrates how to run the model fitting process for a multiple annotations.

* [Model with Bootstrap](Model-Bootstrap.ipynb) - This notebook demonstrates how to run the model fitting process with bootstrap resampling.

* [Bias Test](BiasTest.ipynb) - This notebook demonstrates how to run the bias tests on a model.

* [Interpretability Tests](InterpretabilityTest.ipynb) - This notebook demonstrates how to run the interpretability tests on a model.

* [Interpretability with increasing N<sub>h</sub>](Interpretability_Increasing_N_h.ipynb) - This notebook demonstrates how to interpretability increases with increasing N<sub>h</sub>.

* [Distribution of Regression Coefficients](RegressionCoefficientTest.ipynb) - This notebook demonstrates how to run the interpretability tests on a model and plot the distribution of regression coefficients.

* [Efficiency Tests](EfficiencyTest.ipynb) - This notebook demonstrates how to run the efficiency tests on a model.