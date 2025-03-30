# iQual

[![MIT license](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/worldbank/iQual/blob/main/LICENSE.md)
[![PyPI version](https://img.shields.io/pypi/v/iQual?color=blue&label=pypi%20package&kill_cache=1)](https://badge.fury.io/py/iQual)
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=orange&label=docs&message=iQual)](https://worldbank.github.io/iQual/)

This repository contains the code and resources necessary to implement the techniques described in the paper [A Method to Scale-Up Interpretative Qualitative Analysis, with an Application to Aspirations in Cox's Bazaar, Bangladesh](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099759305162210822/idu0a357362e00b6004c580966006b1c2f2e3996). The `iQual` package is designed for qualitative analysis of open-ended interviews and aims to extend a small set of interpretative human-codes to a much larger set of documents using natural language processing. The package provides a method for assessing the robustness and reliability of this approach. The `iQual` package has been applied to analyze 2,200 open-ended interviews on parent's aspirations for children from Rohingya refugees and their Bangladeshi hosts in Cox's Bazaar, Bangladesh.

With `iQual`, researchers can efficiently analyze large amounts of qualitative data while maintaining the nuance and accuracy of human interpretation.

---

## Installation
- To install `iQual` using pip, use the following command:
```sh
pip install -U iQual
```

---

## Getting Started

For a quick introduction to using iQual, check out our [Getting Started notebook](notebooks/Getting%20Started.ipynb). This tutorial provides:
- A complete overview of the basic workflow
- Step-by-step examples using real-world data
- Clear explanations of key concepts
- Code you can run immediately

This notebook is perfect for new users who want to understand iQual's core functionality without diving into the technical details.

---

## Model Training

The [Model Training](notebooks/Model%20Training.ipynb) notebook demonstrates advanced training techniques including cross-validation and hyperparameter optimization using the same politeness dataset from [Getting Started](notebooks/Getting%20Started.ipynb).

---

## Features

`iQual` is a package designed for qualitative analysis of open-ended interviews. It allows researchers to efficiently analyze large amounts of qualitative data while maintaining the nuance and accuracy of human interpretation.

- Customizable pipelines using [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline) pipelines

- Text-vectorization using:
    - Any of the [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text) text feature extraction method.
    - Any [sentence-transformers](https://sbert.net/) compatible model.
    - Any [spaCy](https://spacy.io/) model with a `doc.vector` attribute.

- Classification using any [scikit-learn](https://scikit-learn.org/stable/modules/) classification method

- Feature Transformation:
    - Dimensionality reduction using any [scikit-learn](https://scikit-learn.org/stable/modules/) `decomposition` method, or UMAP using [umap-learn](https://umap-learn.readthedocs.io/en/latest/).
    - Feature scaling using any [scikit-learn](https://scikit-learn.org/stable/modules/) `preprocessing` method.
    
- Model selection and performance evaluation using [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) methods.
- Tests for bias and interpretability, with [statsmodels](https://www.statsmodels.org/stable/index.html).

---

## Basic Usage

The following code demonstrates the basic usage of the `iQual` package:

```python
from iqual import iqualnlp     # Import `iqualnlp` from the `iqual` package

iqual_model = iqualnlp.Model() # Initiate the model class

# Add text features (using TF-IDF vectorization by default)
iqual_model.add_text_features('question', 'answer')

# Add a classifier (Logistic Regression by default)
iqual_model.add_classifier()

# Add a threshold layer for improved performance on imbalanced data
iqual_model.add_threshold()

# Compile the model
iqual_model.compile()

# Fit the model to your data
iqual_model.fit(X_train, y_train)

# Make predictions
y_pred = iqual_model.predict(X_test)
```

For a more detailed introduction, check out our [Getting Started notebook](notebooks/Getting%20Started.ipynb).

---

## Notebooks

The [notebooks](https://github.com/worldbank/iQual/tree/main/notebooks) folder contains detailed examples on using `iQual`:

* [Getting Started](https://github.com/worldbank/iQual/tree/main/notebooks/Getting%20Started.ipynb)
A complete introduction to iQual with a self-contained example for new users.

* [Basic Modelling](https://github.com/worldbank/iQual/tree/main/notebooks/basic)
These notebooks demonstrate the basic usage of the package, the pipeline construction, and the vectorization and classification options.

* [Advanced Modelling](https://github.com/worldbank/iQual/tree/main/notebooks/advanced) 
These notebooks demonstrate advanced pipeline construction, mixing and matching of feature extraction and classification methods, and model selection.

* [Interpretability](https://github.com/worldbank/iQual/tree/main/notebooks/interpretability)
These notebooks demonstrate the interpretability and related tests for measurement and comparison of interpretability across human and enhanced (machine + human) codes.

* [Bias](https://github.com/worldbank/iQual/tree/main/notebooks/bias) and [Efficiency](https://github.com/worldbank/iQual/tree/main/notebooks/efficiency)
These notebooks demonstrate the bias and efficiency tests for determining the value and validity of enhanced codes.

---

## Citation & Authors
If you use this package, please cite the following paper:

[A Method to Scale-Up Interpretative Qualitative Analysis, with an Application to Aspirations in Cox’s Bazaar, Bangladesh](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099759305162210822/idu0a357362e00b6004c580966006b1c2f2e3996)

```
Ashwin,Julian; Rao,Vijayendra; Biradavolu,Monica Rao; Chhabra,Aditya; Haque,Arshia; Khan,Afsana Iffat; Krishnan,Nandini.
A Method to Scale-Up Interpretative Qualitative Analysis, with an Application to Aspirations in Cox’s Bazaar, Bangladesh (English). (Policy Research Working Paper No. WPS 10046)
Paper is funded by the Knowledge for Change Program (KCP) Washington, D.C. : World Bank Group.
http://documents.worldbank.org/curated/en/099759305162210822/IDU0a357362e00b6004c580966006b1c2f2e3996
```

---

## Maintainers
### Please contact the following people for any queries regarding the package:

- [Aditya Karan Chhabra](mailto:aditya0chhabra@gmail.com)
- [Julian Ashwin](mailto:julianashwin@googlemail.com)
