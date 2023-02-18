# Interpretability

The following notebooks demonstrate how to measure the interpretability of a model, to test whether interpretability improves with increasing N<sub>h</sub> (the number of human annotations), and how to plot the distribution of regression coefficients.

* [Interpretability Tests](InterpretabilityTest.ipynb) - This notebook demonstrates how to run the interpretability tests on human and enhanced data to determine whether the enhanced data adds value by augmenting the human data.

* [Interpretability with increasing N[h]](Interpretability_Increasing_N_h.ipynb) - This notebook demonstrates how interpretability of ML-assisted enhanced data increases with increasing N<sub>h</sub>. This notebooks takes a look at the effect of increasing N<sub>h</sub> while holding N = N<sub>h</sub> +N<sub>m</sub> fixed. Intuitively, this can be thought of as adding human annotations to some of the existing interviews that are currently machine annotated.

* [Distribution of Regression Coefficients](RegressionCoefficientTest.ipynb) - This notebook demonstrates how to run the interpretability tests on a model and plot the distribution of regression coefficients. The sizes of both the human annotated (Nh) and machine annotated (Nm) samples are varied to evaluate how many documents should be annotated by humans to achieve a certain level of interpretability.
