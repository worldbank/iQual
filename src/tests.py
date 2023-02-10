import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

class Bias:
    
    models = smf.__all__ # List of statsapiformula models

    def __init__(
            self,
            enhanced_dataframe,
            annotation_vars=[],
            categorical_regressors=[],
            continuous_regressors=[],
            model='ols',
            act_suffix='_act',
            pred_suffix='_pred',
            error_suffix='_error',
            ):
        """
        Bias test

        Args:
            enhanced_dataframe (pandas dataframe): enhanced dataframe with actual and predicted values
            annotation_vars (list): list of annotation variables
            categorical_regressors (list): list of categorical regressors
            continuous_regressors (list): list of continuous regressors
            model (str): statsmodels model to use. Default is 'ols'
            act_suffix (str): suffix for actual values. This suffix must be present in the enhanced dataframe. Default is '_act'
            pred_suffix (str): suffix for predicted values. This suffix must be present in the enhanced dataframe. Default is '_pred'
            error_suffix (str): suffix for error values. This suffix will be added to the enhanced dataframe. Default is '_error'

        Returns:
            bias_test (Bias): Bias test object
        """
        # TODO: Add support for statsmodels - parent class
        if model not in self.models:
            raise ValueError(f"Model must be one of {', '.join(self.models)}")
        self.model     = model
        self.model_obj = getattr(smf,model)
        self.data      = enhanced_dataframe.copy()
        self.annotation_vars = annotation_vars
        self.categorical_regressors = categorical_regressors
        self.continuous_regressors  = continuous_regressors
        self.model_fits   = {}
        self.act_suffix   = act_suffix
        self.pred_suffix  = pred_suffix
        self.error_suffix = error_suffix
        self._calc_error()

    def _calc_error(self):
        """
        Calculate the error between the actual and predicted values and add it to the dataframe        
        """
        for var in self.annotation_vars:
            self.data[f"{var}{self.error_suffix}"] = self.data[f"{var}{self.act_suffix}"] - self.data[f"{var}{self.pred_suffix}"]
    
    def _format_formula(self,annotation_var):
        """
        Format the formula for the model

        Bias test formula:
            annotation_error ~ C(categorical_regressors) + continuous_regressors

        Args:
            annotation_var (str): annotation variable
        Returns:
            formula (str): statsmodels formula string
        """
        formula = f"{annotation_var}{self.error_suffix} ~ " # Dependent variable
        formula += " + ".join([f"C({var})" for var in self.categorical_regressors]) # Categorical regressors
        formula += " + " + " + ".join([f"{var}" for var in self.continuous_regressors]) # Continuous regressors
        return formula
    
    def fit(self,formula):
        """
        Fit a statsmodels model to the data using the formula
        Args:
            formula (str): statsmodels formula string to fit the model
        Returns:
            model_fit (statsmodels model): fitted model
        """
        model_fit = self.model_obj(formula,data=self.data).fit()
        return model_fit
    
    def fit_all(self):
        """
        Fit all the models.
        All fitted models are stored in self.model_fits
        """
        for var in self.annotation_vars:
            formula = self._format_formula(var)
            self.model_fits[var] = self.fit(formula)

    def get_model_fits(self):
        """
        Get all the model fits
        Returns:
            model_fits (dict): dictionary of model fits
        """
        return self.model_fits
    
    def get_model_fit(self,annotation_var):
        """
        Get the model fit for a particular annotation variable
        Args:
            annotation_var (str): annotation variable
        Returns:
            model_fit (statsmodels model): fitted model
        """
        return self.model_fits[annotation_var]
    
    def get_model_fit_summary(self,annotation_var):
        """
        Get the model fit summary for a particular annotation variable
        Args:
            annotation_var (str): annotation variable
        Returns:
            model_fit_summary (str): model fit summary
        """
        return self.model_fits[annotation_var].summary()
    
    def get_model_fit_summary_all(self):
        """
        Get the model fit summary for all annotation variables
        Returns:
            model_fit_summary_all (str): model fit summary for all annotation variables
        """
        model_fit_summary_all = ""
        for var in self.annotation_vars:
            model_fit_summary_all += f"{var}: {self.model_fits[var].summary()}"
        return model_fit_summary_all
            

"""
- model error (i.e. the sampling errors in the model)
- idiosyncratic error (i.e. the prediction error) 

- Model Error:

- To approximate the model error, we bootstrap the model by sampling the interviews with replacement B times. This gives us an empirical distribution over the predictions based on the sampled distribution.
- The variance of the machine annotations, taking model error into account, can then be approximated by the variance across all of these bootstrap samples.


 This can be calculated either - 
 - in the training set only, 
 - or also in the out-of-sample predictions
 - but we find that the estimates are virtually identical in each

("bootstrap_run", "annotation", "annotation_name","fstat_annot", "pval_annot", "fstat_enh", "pval_enh")


"""

### Interpretability

"""
Assuming text-based variables should be related to household characteristics, if our enhanced sample has improved the
interpretability of our analysis it should give stronger evidence of a relationship between the annotations and household
characteristics.12 We can therefore compare F statistics for regression of annotations on household characteristics
in the human and enhanced samples. If the enhanced sample increases this F statistic relative to the human sample it
suggests that the larger sample leads to more interpretable results in spite of the greater measurement error.
"""


class Interpretability:
    
    models = smf.__all__ # List of statsapiformula models

    def __init__(
            self,
            dataframe,
            annotation_vars=[],
            categorical_regressors=[],
            continuous_regressors=[],
            model='ols',
            ):
        """
        Interpretability class

        Args:
            dataframe (pandas dataframe): dataframe containing the data
            annotation_vars (list): list of annotation variables
            categorical_regressors (list): list of categorical regressors
            continuous_regressors (list): list of continuous regressors
            model (str): statsmodels model to use. Default is 'ols'

        Returns:
            Interpretability object
        """
        # TODO: Add support for statsmodels - parent class
        if model not in self.models:
            raise ValueError(f"Model must be one of {','.join(self.models)}")

        self.dataset = dataframe.copy()
        self.annotation_vars = annotation_vars
        self.categorical_regressors = categorical_regressors
        self.continuous_regressors  = continuous_regressors
        self.model     = model
        self.model_obj = getattr(smf,model)
        self.model_fits = {}

    def _format_formula(self,annotation_var):
        """
        Format the formula for the model.

        Interpretability test formula:
            annotation_var ~ C(categorical_regressors) + continuous_regressors

        Args:
            annotation_var (str): annotation variable
        Returns:
            formula (str): statsmodels formula string
        """

        formula = f"{annotation_var} ~ "
        formula += " + ".join([f"C({var})" for var in self.categorical_regressors])
        formula += " + " + " + ".join([f"{var}" for var in self.continuous_regressors])
        return formula
    
    def fit(self,formula):
        """
        Fit a statsmodels model to the data using the formula.

        Args:
            formula (str): statsmodels formula string to fit the model
        Returns:
            model_fit (statsmodels model): fitted model
        """
        model_fit = self.model_obj(formula,data=self.dataset).fit()
        return model_fit
    
    def fit_all(self):
        """
        Fit all the models. All fitted models are stored in self.model_fits
        """
        for var in self.annotation_vars:
            formula = self._format_formula(var)
            self.model_fits[var] = self.fit(formula)
        return self

    def get_fstats(self,annotation_vars=[]):
        """
        Get the F statistics for the annotation variables
        Args:
            annotation_vars (list): list of annotation variables

        Returns:
            fstats (dict): dictionary of F statistics
        """
        fstats = {}
        for var in annotation_vars:
            fstats[var] = self.model_fits[var].fvalue
        return fstats
    
    def get_pvals(self,annotation_vars=[]):
        """
        Get the p-values for the annotation variables.

        Args:
            annotation_vars (list): list of annotation variables
        Returns:
            pvals (dict): dictionary of p-values
        """
        pvals = {}
        for var in annotation_vars:
            pvals[var] = self.model_fits[var].f_pvalue
        return pvals
    
    def get_fstats_all(self):
        """
        Get the F statistics for all annotation variables
        Returns:
            fstats (dict): dictionary of F statistics
        """
        return self.get_fstats(self.annotation_vars)
    
    def get_pvals_all(self):
        """
        Get the p-values for all annotation variables
        Returns:
            pvals (dict): dictionary of p-values
        """
        return self.get_pvals(self.annotation_vars)
    

    def get_results(self):
        """
        Get the results for all annotation variables

        Returns:
            results (pandas dataframe): dataframe containing the results
        """
        results = []
        fstats = self.get_fstats_all()
        pvals  = self.get_pvals_all()
        for var in self.annotation_vars:
            results.append(dict(
                annotation=var,
                fstat=fstats[var],
                log_fstat=np.log(fstats[var]),
                pval=pvals[var],
                ))
        return pd.DataFrame(results)
