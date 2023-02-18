# Notebooks

This directory contains Jupyter notebooks that demonstrate how to use the `iQual` package.

## [Basic Usage](basic/)

The following notebooks demonstrate how to construct a basic model for a single annotation task:
* [Model with Scikit-learn](basic/Model-Sklearn.ipynb) - This notebook demonstrates how to construct a model using the [scikit-learn](https://scikit-learn.org/stable/) library.
* [Model with SpaCy](basic/Model-SpaCy.ipynb) - This notebook demonstrates how to construct a model using the [SpaCy](https://spacy.io/) library.
* [Model with Sentence-Transformers](basic/Model-SentenceTransformers.ipynb) - This notebook demonstrates how to construct a model using the [Sentence-Transformers](https://www.sbert.net/) library.
* [Model with Saved Dictionary](basic/Model-SavedDictionary.ipynb) - This notebook demonstrates how to construct a model using vectors from a saved dictionary. This can be useful if you have a large number of annotations and want to save time by not having to recompute the vectors for each annotation.

## [Advanced Usage](advanced/)

The following notebooks demonstrate how to construct more advanced models, including models for multiple annotation tasks, models with multiple vectorizers and classifiers, and models with bootstrap resampling.

* [Model with Multiple Vectorizers](advanced/Model-MultipleVectorizers.ipynb) - This notebook demonstrates how to construct a model for a single annotation task using multiple vectorizers. This can be useful if you want to combine different types of vectorizers (e.g. pretrained-embedding models, count-based models)

* [Model with Multiple Classifiers](advanced/Model-MultipleClassifiers.ipynb) - This notebook demonstrates how to construct a model for a single annotation task using multiple classifiers. This can be useful if you want to combine different types of classifiers, and compare their performance on the same data.

* [Model with Multiple Annotations](advanced/Model-MultipleAnnotations.ipynb) - This notebook demonstrates how to run the model fitting process for a multiple annotations.

* [Model with Bootstrap](advanced/Model-Bootstrap.ipynb) - This notebook demonstrates how to run the model fitting process with bootstrap resampling.

## [Interpretability](interpretability/) 

The following notebooks demonstrate how to measure the interpretability of a model, to test whether interpretability improves with increasing N<sub>h</sub> (the number of human annotations), and how to plot the distribution of regression coefficients.

* [Interpretability Tests](interpretability/InterpretabilityTest.ipynb) - This notebook demonstrates how to run the interpretability tests on human and enhanced data to determine whether the enhanced data adds value by augmenting the human data.

* [Interpretability with increasing N[h]](interpretability/Interpretability_Increasing_N_h.ipynb) - This notebook demonstrates how interpretability of ML-assisted enhanced data increases with increasing N<sub>h</sub>. This notebooks takes a look at the effect of increasing N<sub>h</sub> while holding N = N<sub>h</sub> +N<sub>m</sub> fixed. Intuitively, this can be thought of as adding human annotations to some of the existing interviews that are currently machine annotated.

* [Distribution of Regression Coefficients](interpretability/RegressionCoefficientTest.ipynb) - This notebook demonstrates how to run the interpretability tests on a model and plot the distribution of regression coefficients. The sizes of both the human annotated (Nh) and machine annotated (Nm) samples are varied to evaluate how many documents should be annotated by humans to achieve a certain level of interpretability.

## [Efficiency](efficiency/)

* [Efficiency Tests](efficiency/EfficiencyTest.ipynb) - This notebook demonstrates how to run the efficiency tests on a model by accounting for two types of errors in machine annotations: idiosyncratic error (i.e. the prediction error) and model error (i.e. the sampling errors in the model).
For more information on the efficiency tests, refer to the **Efficiency** section in [A Method to Scale-Up Interpretative Qualitative Analysis, with an Application to Aspirations in Cox’s Bazaar, Bangladesh (English). (Policy Research Working Paper No. WPS 10046) ](http://documents.worldbank.org/curated/en/099759305162210822/IDU0a357362e00b6004c580966006b1c2f2e3996)

## [Bias](bias/)

* [Bias Test](bias/BiasTest.ipynb) - This notebook demonstrates how to explicity run bias tests on a model using cross-validated predictions across 25 bootstrap samples.

