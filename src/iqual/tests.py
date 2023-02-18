import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

class Bias:
    # TODO: Add support for statsmodels - parent class

    models = smf.__all__ # List of statsapiformula models

    def __init__(
            self,
            enhanced_dataframe,
            annotation_vars=[],
            categorical_regressors=[],
            numerical_regressors=[],
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
            numerical_regressors (list): list of continuous or binary regressors
            model (str): statsmodels model to use. Default is 'ols'
            act_suffix (str): suffix for actual values. This suffix must be present in the enhanced dataframe. Default is '_act'
            pred_suffix (str): suffix for predicted values. This suffix must be present in the enhanced dataframe. Default is '_pred'
            error_suffix (str): suffix for error values. This suffix will be added to the enhanced dataframe. Default is '_error'

        Returns:
            bias_test (Bias): Bias test object
        """
        
        if model not in self.models:
            raise ValueError(f"Model must be one of {', '.join(self.models)}")
        self.model     = model
        self.model_obj = getattr(smf,model)
        self.data      = enhanced_dataframe.copy()
        self.annotation_vars = annotation_vars
        self.categorical_regressors = categorical_regressors
        self.numerical_regressors  = numerical_regressors
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
            annotation_error ~ C(categorical_regressors) + numerical_regressors

        Args:
            annotation_var (str): annotation variable
        Returns:
            formula (str): statsmodels formula string
        """
        formula = f"{annotation_var}{self.error_suffix} ~ " # Dependent variable
        formula += " + ".join([f"C({var})" for var in self.categorical_regressors]) # Categorical regressors
        formula += " + " + " + ".join([f"{var}" for var in self.numerical_regressors]) # Continuous regressors
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
        return self
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
            
class Interpretability:
    
    models = smf.__all__ # List of statsapiformula models

    def __init__(
            self,
            dataframe,
            annotation_vars=[],
            categorical_regressors=[],
            numerical_regressors=[],
            model='ols',
            ):
        """
        Interpretability class

        Args:
            dataframe (pandas dataframe): dataframe containing the data
            annotation_vars (list): list of annotation variables
            categorical_regressors (list): list of categorical regressors
            numerical_regressors (list): list of continuous or binary regressors
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
        self.numerical_regressors  = numerical_regressors
        self.model        = model
        self.model_obj    = getattr(smf,model)
        self.model_fits   = {}
        self.model_params = {}

    def _format_formula(self,annotation_var):
        """
        Format the formula for the model.

        Interpretability test formula:
            annotation_var ~ C(categorical_regressors) + numerical_regressors

        Args:
            annotation_var (str): annotation variable
        Returns:
            formula (str): statsmodels formula string
        """

        formula = f"{annotation_var} ~ "
        formula += " + ".join([f"C({var})" for var in self.categorical_regressors])
        formula += " + " + " + ".join([f"{var}" for var in self.numerical_regressors])
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
            fitted_model = self.fit(formula)
            self.model_fits[var] = fitted_model
            self.model_params[var] = dict(fitted_model.params)            
        return self
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
    
    def get_model_params(self,annotation_var):
        """
        Get the model parameters for a particular annotation variable
        Args:
            annotation_var (str): annotation variable
        Returns:
            model_params (dict): model parameters
        """
        return self.model_params[annotation_var]

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

    def get_model_std_errors_all(self):
        """
        Get the model standard errors for all annotation variables
        Returns:
            model_std_errors_all (pandas dataframe): dataframe containing the model standard errors
        """
        model_std_errors_all = {}
        for var in self.annotation_vars:
            model_std_errors_all[var] = self.model_fits[var].bse
        return model_std_errors_all

    def get_model_std_error(self, annotation_var, regressor):
        """
        Get the model standard error for a particular annotation variable and regressor
        Args:
            annotation_var (str): annotation variable
            regressor (str): regressor
        Returns:
            model_std_error (float): model standard error
        """
        return self.model_fits[annotation_var].bse[regressor]

    def get_model_coefficient(self, annotation_var, regressor):
        """
        Get the model coefficient for a particular annotation variable and regressor
        Args:
            annotation_var (str): annotation variable
            regressor (str): regressor
        Returns:
            model_coefficient (float): model coefficient
        """
        return self.model_fits[annotation_var].params[regressor]
    
    def get_model_coefficients_all(self):
        """
        Get the model coefficients for all annotation variables
        Returns:
            model_coefficients_all (pandas dataframe): dataframe containing the model coefficients
        """
        model_coefficients_all = {}
        for var in self.annotation_vars:
            model_coefficients_all[var] = self.model_fits[var].params
        return model_coefficients_all
    
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


class Efficiency:
    """
    Class to compute the efficiency of a model
    """

    def __init__(self,
                 train_df,
                 test_df,
                 machine_df,
                 annotation_vars,
                 uid_var='uid',
                 act_suffix='_act',
                 pred_suffix='_pred'):
        """
        Args:
            train_df (pandas dataframe): dataframe containing the training data
            test_df (pandas dataframe): dataframe containing the `test data` OR `cross-validation predictions`
            machine_df (pandas dataframe): dataframe containing the unannotated data
            annotation_vars (list): list of annotation variables
            uid_var (str): user id variable
            act_suffix (str): suffix for the actual values
            pred_suffix (str): suffix for the predicted values

        """
        self.train_df = train_df
        self.test_df = test_df
        self.human_df = pd.concat([train_df, test_df])

        self.machine_df = machine_df

        self.annotation_vars = annotation_vars
        self.act_suffix = act_suffix
        self.pred_suffix = pred_suffix

        self.n_m = self.machine_df[uid_var].nunique()
        self.n_h = self.human_df[uid_var].nunique()
        self.n_sum = self.n_h + self.n_m

    def get_variance_machine(self, var):
        """
        Get the variance of the machine data
        Returns:
            sig2_y (float): variance of the machine data
        """
        return self.machine_df[var+self.pred_suffix].var()

    def get_variance_human(self, var):
        """
        Get the variance of the Human data
        Returns:
            sig2_h (float): variance of the Human data
        """
        return self.human_df[var+self.act_suffix].var()

    def get_variance_residual(self, var):
        """
        Get the variance of the residuals
        Returns:
            sig2_eps (float): variance of the residuals
        """
        return (self.test_df[var+self.act_suffix] - self.test_df[var+self.pred_suffix]).var()

    def get_mean_human(self, var):
        """
        Get the mean of the Human data
        Returns:
            mu_h (float): mean of the Human data
        """
        return self.human_df[var+self.act_suffix].mean()

    def get_mean_machine(self, var):
        """
        Get the mean of the Machine data
        Returns:
            mu_m (float): mean of the Machine data
        """
        return self.machine_df[var+self.pred_suffix].mean()

    def get_std_error_human(self, sig2_h):
        """
        Get the standard error of the Human data
        Returns:
            se_h (float): standard error of the annotated data
        """
        return np.sqrt(sig2_h/self.n_h)

    def get_std_error_machine(self, sig2_m):
        """
        Get the standard error of the Machine data
        Returns:
            se_m (float): standard error of the Machine data
        """
        return np.sqrt(sig2_m/self.n_m)

    def get_std_error_enhanced(self, sig2_h, sig2_m):
        """
        Get the standard error of the Enhanced Data
        Returns:
            se_enh (float): standard error of the Enhanced Data
        """
        return np.sqrt(((self.n_h * sig2_h) + (self.n_m * sig2_m))/(self.n_sum**2))

    def get_results(self, var):
        """
        Get the results for a particular annotation variable
        """
        mu_h = self.get_mean_human(var)
        mu_m = self.get_mean_machine(var)

        sig2_h = self.get_variance_human(var)
        sig2_y = self.get_variance_machine(var)
        sig2_eps = self.get_variance_residual(var)

        sig2_m = sig2_y + sig2_eps

        se_h = self.get_std_error_human(sig2_h)
        se_m = self.get_std_error_machine(sig2_m)
        se_enh = self.get_std_error_enhanced(sig2_h, sig2_m)

        return dict(
            annotation=var,
            sig2_m=sig2_m,
            sig2_y=sig2_y,
            sig2_h=sig2_h,
            sig2_eps=sig2_eps,
            mu_h=mu_h,
            mu_m=mu_m,
            se_h=se_h,
            se_m=se_m,
            se_enh=se_enh,
        )

    def get_results_all(self):
        """
        Get the results for all annotation variables
        """
        results = []
        for var in self.annotation_vars:
            results.append(self.get_results(var))
        return pd.DataFrame(results)