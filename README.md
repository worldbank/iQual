# iQual


This repository contains the code and resources necessary to implement the techniques described in the paper [A Method to Scale-Up Interpretative Qualitative Analysis, with an Application to Aspirations in Cox’s Bazaar, Bangladesh](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099759305162210822/idu0a357362e00b6004c580966006b1c2f2e3996). The `iQual` package is designed for qualitative analysis of open-ended interviews and aims to extend a small set of interpretative human-codes to a much larger set of documents using natural language processing. The package provides a method for assessing the robustness and reliability of this approach. The `iQual` package has been applied to analyze 2,200 open-ended interviews on parent’s aspirations for children from Rohingya refugees and their Bangladeshi hosts in Cox’s Bazaar, Bangladesh. It draws on work in anthropology and philosophy to expand conceptions of aspirations in economics to distinguish between material goals, moral and religious values, and navigational capacity—the ability to achieve goals and aspirations, showing that they have very different correlates. 

With `iQual`, researchers can efficiently analyze large amounts of qualitative data while maintaining the nuance and accuracy of human interpretation.


## `NOTE`: This package is still under development.



> ## Installation
- ### With pip
```python
pip install -U iQual
```

- ### From source

```bash
git clone https://github.com/worldbank/iqual.git```
cd iqual
pip install -e .
```

> ### Dependencies
>
> - Python 3.7+
> - [scikit-learn](https://scikit-learn.org/stable/) 0.22.1+
> - [sentence-transformers](https://sbert.net/) 0.3.6+
> - [spaCy](https://spacy.io/) 2.2.4+
> - [pandas](https://pandas.pydata.org/) 1.0.3+
> - [numpy](https://numpy.org/) 1.18.1+
> - [scipy](https://www.scipy.org/) 1.4.1+
> - [matplotlib](https://matplotlib.org/) 3.1.3+
> - [seaborn](https://seaborn.pydata.org/) 0.10.0+


### Notebooks
> For a detailed description of the package, please see the [notebooks](notebooks) folder.




