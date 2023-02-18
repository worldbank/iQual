# iQual

[![MIT license](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/worldbank/iQual/blob/main/LICENSE.md)
[![PyPI version](https://badge.fury.io/py/iQual.svg)](https://badge.fury.io/py/iQual)

This repository contains the code and resources necessary to implement the techniques described in the paper [A Method to Scale-Up Interpretative Qualitative Analysis, with an Application to Aspirations in Cox’s Bazaar, Bangladesh](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099759305162210822/idu0a357362e00b6004c580966006b1c2f2e3996). The `iQual` package is designed for qualitative analysis of open-ended interviews and aims to extend a small set of interpretative human-codes to a much larger set of documents using natural language processing. The package provides a method for assessing the robustness and reliability of this approach. The `iQual` package has been applied to analyze 2,200 open-ended interviews on parent’s aspirations for children from Rohingya refugees and their Bangladeshi hosts in Cox’s Bazaar, Bangladesh. It draws on work in anthropology and philosophy to expand conceptions of aspirations in economics to distinguish between material goals, moral and religious values, and navigational capacity—the ability to achieve goals and aspirations, showing that they have very different correlates. 

With `iQual`, researchers can efficiently analyze large amounts of qualitative data while maintaining the nuance and accuracy of human interpretation.

---

## Installation
- To install `iQual` using pip, use the following command:
```sh
pip install iQual
```
- Alternatively, you can install `iQual` from source. To do so, use the following commands:
```sh
git clone https://github.com/worldbank/iQual.git
cd iQual
pip install -e .
```
---

## Dependencies

`iQual` requires Python 3.7+ and the following dependencies:

> - [pandas](https://pandas.pydata.org/) 
> - [scikit-learn](https://scikit-learn.org/stable/)
> - [sentence-transformers](https://sbert.net/)
> - [spaCy](https://spacy.io/)
> - [numpy](https://numpy.org/)
> - [umap-learn](https://umap-learn.readthedocs.io/en/latest/)
> - [scipy](https://www.scipy.org/)
> - [statsmodels](https://www.statsmodels.org/stable/index.html) 
> - [simplejson](https://simplejson.readthedocs.io/en/latest/)
> - [num2words](https://pypi.org/project/num2words/)
> - [matplotlib](https://matplotlib.org/)
> - [seaborn](https://seaborn.pydata.org/)

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
- Model performance evaluation using [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) metrics.
- Tests for bias and interpretability, with [statsmodels](https://www.statsmodels.org/stable/index.html).

---

## Basic Usage

The following code demonstrates the basic usage of the `iQual` package. It shows how to construct a pipeline, fit it to the data, and use it to classify new data.

Import the `iqual` package and initiate the model class.

```python
from iqual import iqualnlp     # Import `iqualnlp` from the `iqual` package

iqual_model = iqualnlp.Model() # Initiate the model class
```

Add text features to the model. The `add_text_features` method takes the following arguments:
- `question_col`: The name of the column containing the question text.
- `answer_col`: The name of the column containing the answer text.
- `model`: Name of a scikit-learn, spaCy, sentence-transformers, or a precomputed vector (picklized dictionary) model. The default is `TfidfVectorizer`.
- `env`: The environment or package which is being used. The default is `scikit-learn`. Available options are `scikit-learn`, `spacy`, `sentence-transformers`, and `saved-dict`.
- `**kwargs`: Additional keyword arguments to pass to the model.

```python
# Use a scikit-learn feature extraction method
iqual_model.add_text_features(question_col,answer_col,model='TfidfVectorizer',env='scikit-learn') 

# OR - Use a sentence-transformers model
iqual_model.add_text_features(question_col,answer_col,model='all-mpnet-base-v2',env='sentence-transformers') 

# OR - Use a spaCy model
iqual_model.add_text_features(question_col,answer_col,model='en_core_web_lg',env='spacy') 

# OR - Use a precomputed vector (picklized dictionary)
iqual_model.add_text_features(question_col,answer_col,model='qa_precomputed.pkl',env='saved-dict') 
```

***(OPTIONAL)*** Add a feature transformation layer. The `add_feature_transformer` method takes the following arguments:
- `name`: The name of the feature transformation layer.
- `transformation`: The type of transformation. Available options are `FeatureScaler` and `DimensionalityReduction`.

To add a **feature scaling** layer, use the following code:
```python
iqual_model.add_feature_transformer(name='StandardScaler', transformation="FeatureScaler") # or any other scikit-learn scaler
```
To add a **dimensionality reduction** layer, use the following code:
```python
iqual_model.add_feature_transformer(name='UMAP', transformation="DimensionalityReduction") # supports UMAP or any other scikit-learn decomposition method
```
Add a classifier layer. The `add_classifier` method takes the following arguments:
- `name`: The name of the classifier layer. The default is `LogisticRegression`. 
- `**kwargs`: Additional keyword arguments to pass to the classifier.

```python
iqual_model.add_classifier(name = "LogisticRegression") #  Add a classifier layer from scikit-learn
```
***(OPTIONAL)*** Add a threshold layer for the classifier using `add_threshold` 
```python
iqual_model.add_threshold() # Add a threshold layer for the classifier, recommended for imbalanced data
```
Compile the model with `compile`.
```python
iqual_model.compile() # Compile the model
```
Fit the model to the data using `fit`. The `fit` method takes the following arguments:
- `X_train`: The training data. (pandas dataframe)
- `y_train`: The training labels. (pandas series)

```python
iqual_model.fit(X_train,y_train) # Fit the model to the data
```

Predict the labels for new data using `predict`. The `predict` method takes the following arguments:
- `X_test`: The test data. (pandas dataframe)

```python
y_pred = iqual_model.predict(X_test) # Predict the labels for new data
```

For examples on cross-validation fitting, model selection & performance evaluation, bias, interpretability and measurement tests, refer to the [notebooks](notebooks) folder.

---

## Notebooks

The [notebooks](notebooks) folder contains detailed examples on using `iQual`. The notebooks are organized into the following categories:

* [Basic Modelling](notebooks/basic)
These notebooks demonstrates the basic usage of the package, the pipeline construction, and the vectorization and classification options.

* [Advanced Modelling](notebooks/advanced) 
These notebooks demonstrate advanced pipeline construction, mixing and matching of feature extraction and classification methods, and model selection.

* [Interpretability](notebooks/interpretability)
These notebooks demonstrate the interpretability and related tests for measurement and comparison of interpretability across human and enhanced (machine + human) codes.

* [Bias](notebooks/bias) and [Efficiency](notebooks/efficiency)
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