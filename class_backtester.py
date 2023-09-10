from __future__ import print_function

import platform
import time
import tqdm 
from tqdm import tqdm
tqdm.pandas(desc='My bar!')

# database access
import pandas_datareader as web
import quandl as quandl
import wrds as wrds

# storage and operations
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import joblib

# Visualization Libraries 
import matplotlib.pyplot as plt 
import seaborn as sns

# Statistical Analysis Libraries 
from statsmodels.regression.rolling import RollingOLS
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats.mstats import winsorize
from scipy.optimize import minimize
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer, MinMaxScaler, QuantileTransformer, RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin, clone # How to create our own scaler 
import statsmodels.api as sm
import linearmodels as lm 
from itertools import product, combinations
# import torch
import xgboost as xgb

# Others 
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count

plt.rcParams.update({'font.size':20})

import warnings 
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

def MVP(wvec,*args):
    
    """
    Calculates the minimum variance portfolio (MVP) given a weight vector and covariance matrix.

    Parameters:
    -----------
    wvec : np.ndarray
        Weight vector for portfolio.
    *args : tuple
        Additional arguments passed as a tuple. Expects a single argument, the covariance matrix.

    Returns:
    --------
    float
        The variance of the portfolio given the weight vector and covariance matrix.

    Notes:
    ------
    The MVP is the portfolio with the lowest variance possible given the set of assets in the portfolio.
    This function takes in a weight vector and a covariance matrix (provided as a single argument in a tuple)
    and returns the variance of the resulting portfolio.
    """
    cov = args[0]
    var = wvec@cov@wvec
    return var

def MSR(wvec,*args):
    """
    Calculates the maximum Sharpe ratio (MSR) given a weight vector, covariance matrix, and expected returns.

    Parameters:
    -----------
    wvec : np.ndarray
        Weight vector for portfolio.
    *args : tuple
        Additional arguments passed as a tuple. Expects two arguments, the covariance matrix and the expected returns.

    Returns:
    --------
    float
        The negative maximum Sharpe ratio of the portfolio given the weight vector, covariance matrix, and expected returns.

    Notes:
    ------
    The MSR is the portfolio with the highest Sharpe ratio possible given the set of assets in the portfolio.
    This function takes in a weight vector, a covariance matrix, and expected returns (provided as a tuple)
    and returns the negative of the maximum Sharpe ratio of the resulting portfolio.
    """

    cov = args[0]
    mu  = args[1]
    sr = mu@wvec/(wvec@cov@wvec)
    return -sr 
 
def LS(wvec, *args):
    
    """
    Computes the negative Sharpe ratio of a long-short portfolio given a weight vector and the covariance matrix and expected returns of the assets.
    
    Parameters:
    -----------
    wvec: array-like
        The weight vector representing the long-short portfolio.
        
    args: tuple
        A tuple containing the following positional arguments:
        cov: numpy.ndarray
            The covariance matrix of the assets.
        mu: numpy.ndarray
            The expected returns of the assets.
        expret: numpy.ndarray
            The expected returns of the assets.
    
    Returns:
    --------
    float
        The negative Sharpe ratio of the portfolio.
    """

    # unpack the arguments
    cov = args[0]
    mu = args[1]
    expret = args[1]
    num_stocks = len(mu)
    
    # calculate portfolio weights for long and short positions
    long_wvec = np.zeros(num_stocks)
    short_wvec = np.zeros(num_stocks)
    long_wvec[wvec > 0] = wvec[wvec > 0] / np.sum(wvec[wvec > 0]) # scale by sum of positive weights
    short_wvec[wvec < 0] = -wvec[wvec < 0] / np.sum(wvec[wvec < 0]) # scale by sum of negative weights
    
    # calculate delta-neutral portfolio weights
    delta_wvec = long_wvec - short_wvec
    delta_wvec /= np.sum(np.abs(delta_wvec)) # scale by sum of absolute weights
    
    # calculate portfolio return
    port_return = expret @ delta_wvec
    
    # calculate portfolio standard deviation
    port_std = np.sqrt(delta_wvec @ cov @ delta_wvec)
    
    # calculate Sharpe ratio
    sr = port_return / port_std
    
    return -sr

def calculate_drawdown(df):
    '''
    This function calculates maximum drawdown for a dataframe using the date range in the dataframe.

    Arguments:
        df: pd.DataFrame
            MUST CONTAIN THESE COLUMNS: `date`, `return`
            The `date` column will serve as the time window in which drawdown is calculated.
            The `return` column is the grouped sum of the weights * returns by date. This represents the returns
            for all permnos for that date.

    Returns:
        Float value which is maximum drawdown over the time period on dates within the dataframe.
    '''
    
    drawdown_previous = 0
    date_list = np.sort(df['date'].unique())
    returns = df.set_index('date')['return'].values
    for start_date in range(len(date_list)):
        for end_date in range(start_date + 1, len(date_list)):
            drawdown_current = returns[end_date] - returns[start_date]
            if drawdown_current < drawdown_previous:
                drawdown_previous = drawdown_current
            else:
                continue
            
    return drawdown_previous

def optimize(cov, mu, fun, cons, bounds, permno_list):
    """
    Optimize portfolio weights based on a given objective function and constraints.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix of asset returns.
    mu : np.ndarray
        Vector of expected returns for each asset.
    fun : callable
        Objective function to be minimized.
    cons : list
        List of constraint dictionaries. Each dictionary represents a constraint
        and has the keys 'type' (string) and 'fun' (callable).
    bounds : list
        List of tuples specifying the bounds for each asset weight. Each tuple
        contains two elements, the lower and upper bounds for the weight.

    Returns
    -------
    res : OptimizeResult
        An object containing the optimization results, including the optimized
        portfolio weights.

    """
    N = len(mu)

    res = minimize(fun,
                    np.ones(N)/N,
                    args = (cov, mu), 
                    constraints = cons, 
                    bounds = bounds, 
                    method = 'SLSQP',
                    options = {'ftol': 1e-8, 'disp': False})
    
    weights_dict = {}
    for k, v in zip(permno_list, res.x):
        weights_dict[k] = v
    
    return weights_dict

class Winsorize(BaseEstimator, TransformerMixin):
    feature_names = None
    level_winsorize = None
    # add another additional parameter, just for fun, while we are at it

    def __init__(self, feature_names = None, level_winsorize = 0.025):
        # print('\n>>>>>>>init() called.\n')
        self.feature_names = feature_names
        self.level_winsorize = level_winsorize
        self.abs_level_winsorize = {}

    def fit(self, X, y = None):
        # print('>>>>>>>fit() called.')
        X_ = X.copy() # creating a copy to avoid changes to original dataset
        self.feature_names = X_.columns if self.feature_names is None else self.feature_names
        for ft in self.feature_names:
            temp = winsorize(X_.loc[:,ft], limits=self.level_winsorize, nan_policy='omit').data
            self.abs_level_winsorize.update({ft:(np.nanmin(temp), np.nanmax(temp))})
            pass
        # print(f'level:{self.level_winsorize}\n')
        # print(f'abs levels: {self.abs_level_winsorize}\n')
        return self

    def transform(self, X, y = None):
        # print('\n>>>>>>>transform() called.\n')
        X_ = X.copy() # creating a copy to avoid changes to original dataset
        for ft in self.feature_names:
            if ft in X_.columns:
                lims = self.abs_level_winsorize[ft]
                tail_left = X_[ft]<=lims[0]
                tail_right = X_[ft]>=lims[1]
                X_.loc[tail_left, ft] = lims[0]
                X_.loc[tail_right, ft] = lims[1]
            else:
                # print(f'Warning: {ft} not in the list of variables/ skipping')
                continue
            pass
        return X_



class Backtester:
    
    def __init__(self,
                 df,
                 params,
                 optimise,  
                 preprocess_features,
                 modeling_features,
                 rolling_frw,
                 look_back_prm,
                 configurations,
                 col_to_pred,
                 days_avoid_bias
                 ):
        
        self.df = df
        self.params = params
        self.optimise = optimise
        self.preprocess_features = preprocess_features
        self.modeling_features = modeling_features
        self.col_to_pred = col_to_pred
        self.rolling_frq = rolling_frw
        self.look_back_prm = look_back_prm
        self.days_avoid_bias = days_avoid_bias
        self.configurations = configurations
        
        self.optimal_weights = {}
        self.returns_strategy = {}
        self.realized_betas = {}
        self.dict_all_predictions = {}
        self.dict_feature_importance = {}

        self.dict_feature_importance["random_forest"] = {}
        self.dict_feature_importance["xgboost"] = {}
        self.beta_mktrf = None


        for key in configurations.keys():
            self.optimal_weights[key] = self.df[['permno', 'date']].copy().set_index(['permno', 'date'])
            self.optimal_weights[key]['weight'] = 0
            self.returns_strategy[key] = self.df[['date']].copy().drop_duplicates().set_index(['date'])
            self.returns_strategy[key]['return'] = None
            self.returns_strategy[key].index = pd.to_datetime(self.returns_strategy[key].index)

            self.realized_betas[key] = self.df[['date']].copy().drop_duplicates().set_index(['date'])
            self.realized_betas[key]['return'] = None
            self.realized_betas[key].index = pd.to_datetime(self.returns_strategy[key].index)

            self.dict_all_predictions[key] = pd.DataFrame()


    def make_prediction(self, df_out_sample, model):
        
        df_pred = df_out_sample.copy()
        df_pred[self.col_to_pred + '_pred'] = model.predict(df_out_sample[self.modeling_features])
        return df_pred
    
    def get_n_days_rolling(self):
        if self.rolling_frq == '1M':
            return 30
        elif self.rolling_frq == '1W':
            return 7
        elif self.rolling_frq == '1D':
            return 1
        else:
            print('rolling_frq not supported: ' + self.rolling_frq)

    
    def alpha_estimation(self, df_r, alpha_estimation_method, alpha: None, l1_ratio: None):
        
        if alpha_estimation_method == "Lasso": 
            model = Lasso(alpha=alpha, max_iter=int(1e5))
            
        elif alpha_estimation_method == "RollingOLS":
            model = LinearRegression(normalize=True)
            
        elif alpha_estimation_method == "Ridge":
            model = Ridge(alpha=alpha)
            
        elif alpha_estimation_method == "ElasticNet":
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000)
            
        elif alpha_estimation_method == "xgboost":
            if self.params:
                model = xgb.XGBRegressor(**self.params)
            else: 
                model = xgb.XGBRegressor(random_state = 42, n_jobs = -1)
            
        elif alpha_estimation_method == "random_forest": 
            model = RandomForestRegressor(random_state=0 , n_jobs=-1, max_features=int(1))
            
        # Fit the model
        model.fit(df_r[self.modeling_features], df_r[self.col_to_pred])
        
        return model
    
    def prepare_inputs_for_optimisation(self, df_r, df_out_sample, cfg):   
        #take the list of names that are present in both insample and out of sample datatset
        #we need that to have exactly the same stock that we make prediction for in the out of sample
        permnos = list(set(df_r['permno'].unique()).intersection(df_out_sample['permno'].unique()))
        df_r = df_r[df_r['permno'].isin(permnos)]
        df_out_sample = df_out_sample[df_out_sample['permno'].isin(permnos)]
        #rearrange the forward returns to a matrix

        ret = pd.pivot_table(df_r, values='fret1d', index=['date'], columns=['permno'])
        ret = ret.fillna(0.0)

        #rearrange the predicted returns to a matrix 
        #we create here the dataframe of expected returnds
        expret = pd.pivot_table(df_out_sample, 
                                values='fret1d_pred', index=['date'], columns=['permno'])
        expret = expret.fillna(0.0)

        # calculate covariance between returns using insample data
        # in practive we use a risk model here (barrs, axioma)
        r1d = pd.pivot_table(df_r, values='ret', index=['date'], columns=['permno'])
        r1d = r1d.fillna(0.0)
        cov = r1d.cov().values

        #we have predicted returns for each day, take the last day in our period
        expret = expret.iloc[0].values  # changed from -1 to 0

        # Here we need to change to col_to_pred because otherwise we are using the predicted returns and not the 
        # actual returns
        #rearrange realised returns to simplify strategy return calculation
        realised_ret = pd.pivot_table(df_out_sample, values='fret1d', index=['date'], columns=['permno'])
        realised_ret.index = pd.to_datetime(realised_ret.index)
        realised_ret = realised_ret.fillna(0)

        # ---- Get dictionary of beta values from df_out_sample ----
        cons_betas = ['cons_beta_mktrf', 'cons_beta_smb', 'cons_beta_hml', 'cons_beta_mom']
        date_to_lookup = pd.pivot_table(df_out_sample, values = 'cons_beta_mktrf', index = ['date'], columns = ['permno']).index[0]
        
        cons_betas_dict = {}                        
        for beta in cons_betas:
                beta_df = pd.pivot_table(df_out_sample, values = beta, index = ['date'], columns = ['permno'])
                cons_betas_dict[f'{beta}'] = beta_df.iloc[0, :].values.flatten()
        
        # ---- Get dictionary of portfolio beta values of the benchmark using the first date from the out of sample data ----
        bench_lookup_df = self.df.loc[self.df['permno'].isin(permnos)]
        bench_lookup_df = bench_lookup_df.loc[bench_lookup_df['date'] == date_to_lookup][['date',
                                                                                          'permno',
                                                                                          'beta_mktrf_bench', 
                                                                                          'beta_smb_bench', 
                                                                                          'beta_hml_bench', 
                                                                                          'beta_mom_bench', 
                                                                                          'vw']]
        beta_bench_dict = {'beta_mktrf_bench': np.array(bench_lookup_df['beta_mktrf_bench']),
                           'beta_smb_bench': np.array(bench_lookup_df['beta_smb_bench']),
                           'beta_hml_bench': np.array(bench_lookup_df['beta_hml_bench']),
                           'beta_mom_bench': np.array(bench_lookup_df['beta_mom_bench'])}

        # ---- Constraint Functions for beta_mktrf ----
        def beta_mktrf_constraint_upper(wvec, beta_mktrf, beta_mktrf_bench):
                a = 1.05*beta_mktrf_bench - wvec@beta_mktrf
                # print(a)
                return 1.05*beta_mktrf_bench - wvec@beta_mktrf

        def beta_mktrf_constraint_lower(wvec, beta_mktrf, beta_mktrf_bench):
                a = wvec@beta_mktrf - 0.95*beta_mktrf_bench
                # print(a)
                return wvec@beta_mktrf - 0.95*beta_mktrf_bench
        
        # ---- Constraint Functions for beta_smb ----
        def beta_smb_constraint_upper(wvec, beta_smb, beta_smb_bench):
                return 1.05*beta_smb_bench - wvec@beta_smb
                
        def beta_smb_constraint_lower(wvec, beta_smb, beta_smb_bench):
                return wvec@beta_smb - 0.95*beta_smb_bench
        
        # ---- Constraint Functions for beta_hml ----
        def beta_hml_constraint_upper(wvec, beta_hml, beta_hml_bench):
                return 1.05*beta_hml_bench - wvec@beta_hml
        
        def beta_hml_constraint_lower(wvec, beta_hml, beta_hml_bench):
                return wvec@beta_hml - 0.95*beta_hml_bench
        
        # ---- Constraint Functions for beta_mom ----
        def beta_mom_constraint_upper(wvec, beta_mom, beta_mom_bench):
                return 1.05*beta_mom_bench - wvec@beta_mom
        
        def beta_mom_constraint_lower(wvec, beta_mom, beta_mom_bench):
                return wvec@beta_mom - 0.95*beta_mom_bench
        
        # ---- Constraint Function for Drawdown ----
        # Benchmark drawdown value using the first date from the out of sample data ----
        bench_drawdown = self.df.copy()
        bench_drawdown['ret_vw'] = bench_drawdown['ret'] * bench_drawdown['vw']
        bench_drawdown['return'] = bench_drawdown.groupby(by = ['date'])['ret_vw'].transform(sum)
        bench_drawdown = bench_drawdown.groupby(by = ['date']).first().reset_index()[['date', 'return']]
        
        reference_day = df_out_sample['date'].min()
        current_realized_ret = realised_ret.iloc[0, :]
        
        def drawdown_constraint(wvec, bench_drawdown, reference_day, current_realized_ret):
            reference_day = reference_day
            first_day = reference_day - pd.Timedelta(days = 29)
            
            # Get returns for current date, dt, using wvec and realized returns
            returns_for_dt = wvec@current_realized_ret
            new_row = {'date': reference_day,
                       'return': returns_for_dt}
            
            # Get historical returns from previously calculated weights and realized returns
            historical_returns = self.returns_strategy[cfg].reset_index()
            historical_returns = historical_returns.loc[(historical_returns['date'] >= first_day) & (historical_returns['date'] < reference_day)]
            
            # Ensure that on the first 30 days of prediction for dt, the portfolio and benchmark drawdown are both set to 0.
            # This is because there is not a full month of historical weights/returns that exist yet to compare to the benchmark portfolio.
            if historical_returns['return'].isna().sum() > 1:
                benchmark_drawdown = 0
                portfolio_drawdown = 0
                
                return portfolio_drawdown - (0.99*benchmark_drawdown)
            else:
                # Subset benchmark drawdown df according to first day and reference day, then calculate benchmark drawdown.
                bench_drawdown = bench_drawdown.loc[(bench_drawdown['date'] >= first_day) & (bench_drawdown['date'] <= reference_day)]
                bench_drawdown = bench_drawdown[['date', 'return']]
                bench_drawdown['return'] = bench_drawdown['return'].values.cumsum() # Drawdown should be calculated with cumulative returns for both benchmark and our portfolio.
                benchmark_drawdown = calculate_drawdown(df = bench_drawdown)

                # Create new dataframe which includes historical returns + current returns from dt, then calculate portfolio drawdown
                historical_and_current_returns = historical_returns.append(new_row, ignore_index = True)
                historical_and_current_returns = historical_and_current_returns.dropna()
                historical_and_current_returns['return'] = historical_and_current_returns['return'].values.cumsum() # Drawdown should be calculated with cumulative returns for both benchmark and our portfolio.
                portfolio_drawdown = calculate_drawdown(df = historical_and_current_returns)

                return portfolio_drawdown - (0.99*benchmark_drawdown)
            
        # define constraints
        cons = [
            {'type': 'eq', 'fun' : lambda wvec: wvec.sum()-1},
            {'type': 'ineq', 'fun': beta_mktrf_constraint_upper, 'args': (cons_betas_dict['cons_beta_mktrf'], beta_bench_dict['beta_mktrf_bench'], )},
            {'type': 'ineq', 'fun': beta_mktrf_constraint_lower, 'args': (cons_betas_dict['cons_beta_mktrf'], beta_bench_dict['beta_mktrf_bench'], )},
            {'type': 'ineq', 'fun': beta_smb_constraint_upper, 'args': (cons_betas_dict['cons_beta_smb'], beta_bench_dict['beta_smb_bench'], )},
            {'type': 'ineq', 'fun': beta_smb_constraint_lower, 'args': (cons_betas_dict['cons_beta_smb'], beta_bench_dict['beta_smb_bench'], )},
            {'type': 'ineq', 'fun': beta_hml_constraint_upper, 'args': (cons_betas_dict['cons_beta_hml'], beta_bench_dict['beta_hml_bench'], )},
            {'type': 'ineq', 'fun': beta_hml_constraint_lower, 'args': (cons_betas_dict['cons_beta_hml'], beta_bench_dict['beta_hml_bench'], )},
            {'type': 'ineq', 'fun': beta_mom_constraint_upper, 'args': (cons_betas_dict['cons_beta_mom'], beta_bench_dict['beta_mom_bench'], )},
            {'type': 'ineq', 'fun': beta_mom_constraint_lower, 'args': (cons_betas_dict['cons_beta_mom'], beta_bench_dict['beta_mom_bench'], )},
            {'type': 'ineq', 'fun': drawdown_constraint, 'args': (bench_drawdown, reference_day, current_realized_ret)}
        ]
        
        # ---- Weight Deviation of 10% per Asset Constraint ----
        bench_asset_weights = np.array([bench_lookup_df.loc[bench_lookup_df['permno'] == i]['vw'].item() for i in permnos])
        bounds = [[0.9*bench_asset_weights[i], 1.1*bench_asset_weights[i]] for i in range(len(permnos))]
        
        return cov, expret, realised_ret, cons, bounds, permnos

    def run_backtest(self):
    
        for dt in tqdm(pd.date_range(start=self.df['date'].min() + pd.Timedelta(days=self.look_back_prm),
                                end=self.df['date'].max() - pd.Timedelta(days=self.get_n_days_rolling()+1), 
                                freq=self.rolling_frq)):
            
        
            for cfg_name, cfg in self.configurations.items():
                #############################################
                ### ASSIGN CONFIGURATION PARAMETERS #########
                parsing_pipe = cfg.get('parsing',None)
                #############################################

                df = self.df 
                df['constant'] = 1


                # step1: restrict dataset to insample
                df_r = df.loc[np.logical_and(
                    df['date'] >= dt - pd.Timedelta(days=self.look_back_prm), 
                    df['date'] < dt - pd.Timedelta(days=self.days_avoid_bias)), :].copy()
           

                # step2: prepare data - filter out outliers, normalise, winsorise, cox-box transform
                # ADDED THE PARSING PIPE
                # if the pipeline in the configuration exists, preprocess the data, otherwise skip preprocessing
                if parsing_pipe is not None: 
                    df_r[self.preprocess_features] = parsing_pipe.fit_transform(df_r[self.preprocess_features])
    
                
                # step3: run alpha estimation method
                try: 
                    alpha = cfg['alpha']
                except:
                    alpha = None   
                
                try:   
                    l1_ratio = cfg['l1_ratio']

                except:
                    l1_ratio = None
                
                model = self.alpha_estimation(df_r, cfg['alpha_estimation_method'], alpha, l1_ratio)

                # step4: set out of sample period 
                df_out_sample = df.loc[np.logical_and(
                    df['date'] >= dt, 
                    df['date'] < dt + pd.Timedelta(days=self.get_n_days_rolling())), :].copy()
                
                if df_out_sample.empty:
                    continue
                
                # step5: prepare out of data (only factors!) - filter out outlier, normalise, winsorise, cox-box trasnform
                if parsing_pipe is not None:
                    df_out_sample[self.preprocess_features] = parsing_pipe.transform(df_out_sample[self.preprocess_features])

                # step6: make prediction for out of sample
                df_pred = self.make_prediction(df_out_sample, model)
                
                if self.optimise == True:
                    # step7: estimate covarinace and prepare inputs for optimisation
                    cov, expret, realised_ret, cons, bounds, permnos = \
                        self.prepare_inputs_for_optimisation(df_r, df_pred, cfg = cfg_name)
            
                    #step8: solve optimisation problem
                    weights_dict = optimize(cov, expret, cfg['opt_function'], cons, bounds, permno_list = permnos)
                    for d in df_out_sample['date'].unique():
                        # Get number of unique permnos for day `d`
                        p = self.optimal_weights[cfg_name].loc[(permnos, d), :].reset_index()
                        p = list(p['permno'].unique())
                        p = [int(i) for i in p]
                        new_dict = {k: weights_dict[k] for k in p}

                        # Sort permnos so that we can append into optimal_weights to ensure order is preserved
                        index_map = {v: i for i, v in enumerate(p)}
                        sorted(new_dict.items(), key = lambda pair: index_map[pair[0]])
                        
                        self.optimal_weights[cfg_name].loc[(permnos, d), :] = np.array(list(new_dict.values()))

                        if d in realised_ret.index:
                            # Ensure that realized returns also have the same unique permnos for day `d` using `p` as defined above
                            realised_ret_copy = realised_ret.copy()[p]
                            self.returns_strategy[cfg_name].loc[d] = np.array(list(new_dict.values()))@realised_ret_copy.loc[d]
                            
                # save all predictions of one specific configuration to a dictionary
                self.dict_all_predictions[cfg_name] = pd.concat([self.dict_all_predictions[cfg_name], df_pred])
        
                # save the feature importance if the model is random forest or xgboost
                if cfg['alpha_estimation_method'] in ['xgboost', 'random_forest']:
                    key = str(list(df_out_sample['date'].unique())[0])
                    # saving the feature importance for insample data in a dictionary
                    self.dict_feature_importance[cfg['alpha_estimation_method']][key] = dict(zip(self.modeling_features, list(model.feature_importances_)))    