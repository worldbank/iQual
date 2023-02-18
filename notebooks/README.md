# notebooks

This directory contains Jupyter notebooks that demonstrate how to use the `iQual` package.

## [Basic Usage](basic/)

The following notebooks demonstrate how to construct a basic model for a single annotation task:
* [Model with Scikit-learn](basic/Model-Sklearn.ipynb) - This notebook demonstrates how to construct a model using the [scikit-learn](https://scikit-learn.org/stable/) library.
* [Model with SpaCy](basic/Model-SpaCy.ipynb) - This notebook demonstrates how to construct a model using the [SpaCy](https://spacy.io/) library.
* [Model with Sentence-Transformers](basic/Model-SentenceTransformers.ipynb) - This notebook demonstrates how to construct a model using the [Sentence-Transformers](https://www.sbert.net/) library.
* [Model with Saved Dictionary](basic/Model-SavedDictionary.ipynb) - This notebook demonstrates how to construct a model using vectors from a saved dictionary. This can be useful if you have a large number of annotations and want to save time by not having to recompute the vectors for each annotation.

## [Advanced Usage](advanced/)

The following notebooks demonstrate how to construct more advanced models, including models for multiple annotation tasks, models with multiple vectorizers and classifiers, and models with bootstrap resampling.

* [Model with Multiple Vectorizers](advanced/Model-MultipleVectorizers.ipynb) - This notebook demonstrates how to construct a model for a single annotation task using multiple vectorizers. This can be useful if you want to compare the performance of different vectorizers.

* [Model with Multiple Classifiers](advanced/Model-MultipleClassifiers.ipynb) - This notebook demonstrates how to construct a model for a single annotation task using multiple classifiers. This can be useful if you want to compare the performance of different classifiers.

* [Model with Multiple Annotations](advanced/Model-MultipleAnnotations.ipynb) - This notebook demonstrates how to run the model fitting process for a multiple annotations.

* [Model with Bootstrap](advanced/Model-Bootstrap.ipynb) - This notebook demonstrates how to run the model fitting process with bootstrap resampling.

## [Interpretability](interpretability/) 

The following notebooks demonstrate how to measure the interpretability of a model.

* [Interpretability Tests](interpretability/InterpretabilityTest.ipynb) - This notebook demonstrates how to run the interpretability tests on a model.

* [Interpretability with increasing N[h]](interpretability/Interpretability_Increasing_N_h.ipynb) - This notebook demonstrates how to interpretability increases with increasing N<sub>h</sub>.

* [Distribution of Regression Coefficients](RegressionCoefficientTest.ipynb) - This notebook demonstrates how to run the interpretability tests on a model and plot the distribution of regression coefficients.

## [Efficiency](efficiency/)

The following notebooks demonstrate how to measure the efficiency of a model.

* [Efficiency Tests](efficiency/EfficiencyTest.ipynb) - This notebook demonstrates how to run the efficiency tests on a model.

## [Bias](bias/)

The following notebooks demonstrate how to measure the bias of a model.

* [Bias Test](bias/BiasTest.ipynb) - This notebook demonstrates how to run the bias tests on a model.

