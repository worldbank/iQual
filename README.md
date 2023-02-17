# iQual

This repository contains the code and resources necessary to implement the techniques described in the paper [A Method to Scale-Up Interpretative Qualitative Analysis, with an Application to Aspirations in Cox’s Bazaar, Bangladesh](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099759305162210822/idu0a357362e00b6004c580966006b1c2f2e3996). The `iQual` package is designed for qualitative analysis of open-ended interviews and aims to extend a small set of interpretative human-codes to a much larger set of documents using natural language processing. The package provides a method for assessing the robustness and reliability of this approach. The `iQual` package has been applied to analyze 2,200 open-ended interviews on parent’s aspirations for children from Rohingya refugees and their Bangladeshi hosts in Cox’s Bazaar, Bangladesh. It draws on work in anthropology and philosophy to expand conceptions of aspirations in economics to distinguish between material goals, moral and religious values, and navigational capacity—the ability to achieve goals and aspirations, showing that they have very different correlates. 

With `iQual`, researchers can efficiently analyze large amounts of qualitative data while maintaining the nuance and accuracy of human interpretation.


### `NOTE`: This package is still under development.

## Installation
- To install iQual using pip, use the following command:
```sh
pip install -U iQual
```
- Alternatively, you can install iQual from source. To do so, use the following commands:
```sh
git clone https://github.com/worldbank/iQual.git
cd iQual
pip install -e .
```

### Dependencies

iQual requires Python 3.7+ and the following dependencies:
- [scikit-learn](https://scikit-learn.org/stable/) 0.22.1+
- [sentence-transformers](https://sbert.net/) 0.3.6+
- [spaCy](https://spacy.io/) 2.2.4+
- [pandas](https://pandas.pydata.org/) 1.0.3+
- [numpy](https://numpy.org/) 1.18.1+
- [scipy](https://www.scipy.org/) 1.4.1+
- [matplotlib](https://matplotlib.org/) 3.1.3+
- [seaborn](https://seaborn.pydata.org/) 0.10.0+

### Features

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
- Model selection using [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) methods.
- Model performance evaluation using [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) metrics.
- Tests for bias and interpretability, with [statsmodels](https://www.statsmodels.org/stable/index.html).

### Notebooks
Please refer to the [notebooks](notebooks) folder for a detailed description of the iQual package. These notebooks provide examples of how to use the package and demonstrate its capabilities.

### Citation & Authors
If you use this package, please cite the following paper:

```bibtex
Ashwin,Julian; Rao,Vijayendra; Biradavolu,Monica Rao; Chhabra,Aditya; Haque,Arshia; Khan,Afsana Iffat; Krishnan,Nandini.
A Method to Scale-Up Interpretative Qualitative Analysis, with an Application to Aspirations in Cox’s Bazaar, Bangladesh (English). (Policy Research Working Paper No. WPS 10046)
Paper is funded by the Knowledge for Change Program (KCP) Washington, D.C. : World Bank Group.
http://documents.worldbank.org/curated/en/099759305162210822/IDU0a357362e00b6004c580966006b1c2f2e3996
```