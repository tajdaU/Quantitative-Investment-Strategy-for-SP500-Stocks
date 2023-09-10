import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import quantstats as qs

# Helper Functions 
def select_permno(df, permno): 
    permno_selection_df = df[df["permno"] == permno]
    return permno_selection_df


def plot_variable(df, X_label: str, y_label:str, title: str):
    
    df = df.sort_values(by = "date", ascending = True)
    plt.figure(figsize=(20,5))
    plt.plot(df[X_label], df[y_label])
    plt.legend([y_label])
    plt.title(title)
    plt.show()
    
def plot_hist(df1, df2, column_name):
    # creating axes to draw plots 
    fig, ax = plt.subplots(1, 2) 

    # plotting the original data(non-normal) and  
    # fitted data (normal) 
    sns.distplot(df1[column_name], hist = False, kde = True, 
                kde_kws = {'shade': True, 'linewidth': 2},  
                label = "Non-Normal", color ="green", ax = ax[0]) 

    sns.distplot(df2[column_name], hist = False, kde = True, 
                kde_kws = {'shade': True, 'linewidth': 2},  
                label = "Normal", color ="green", ax = ax[1]) 

    # adding legends to the subplots 
    plt.legend(loc = "upper right") 

    # rescaling the subplots 
    fig.set_figheight(5) 
    fig.set_figwidth(10) 


def perform_trading_analysis(df, trading_strat, benchmark_df): 
    
    """
    Performs trading analysis on a given strategy's returns data and outputs
    key performance metrics and plots.

    Parameters:
    -----------
    dict : dict
        A dictionary containing returns data for different trading strategies.
    trading_strategy : str
        The name of the trading strategy to analyze.

    Returns:
    --------
        This function prints the Sharpe ratio, annual return, and average daily
        return (in basis points) for the specified trading strategy to the console.
        It also generates several plots of the returns data using qs.reports.full().
    """
    
    df = df.set_index(pd.to_datetime(df.index))
    df = df[[ trading_strat]]
    df = df.sort_index(ascending = True)
    df = df[[trading_strat]].fillna(0.0)
    print('Sharpe ratio = ' + str(df[trading_strat].mean()/df[trading_strat].std()*np.sqrt(252)))
    print('Average daily return (bps)= ' + str(df[trading_strat].mean()* 1e4))
    print("-------------")
    
    qs.reports.full(df[trading_strat], benchmark = benchmark_df)


def get_metrics(dict_): 
    
    results = pd.DataFrame()
    for run_name in dict_.keys():
        info = pd.DataFrame(dict_[run_name].sort_values(by = "date", ascending = True).fillna(0.0))    
        results["date"] = pd.to_datetime(info.index)
        results[run_name] = info["return"].values
    results = results.set_index(results["date"]).drop(columns = "date")
    
    #results = results[(results.index > "2002-01-01") & (results.index < "2015-01-01")]
    print(results.mean()/results.std()*np.sqrt(252))
    results.cumsum().plot()
    
    return results

def calculate_performance_metrics(returns_df, config_name: str,  risk_free_rate=0.0):
    
    returns_df = returns_df.set_index(pd.to_datetime(returns_df.index))
    
    # Calculate daily returns
    daily_returns = returns_df[config_name]

    # Calculate cumulative returns
    cumulative_return = (1 + daily_returns).cumprod() 
    cumulative_return = cumulative_return.iloc[-1] - 1
    cumulative_return = np.round(cumulative_return*100,2)

    # Calculate Sharpe ratio
    sharpe_ratio = qs.stats.sharpe(returns_df[config_name])

    # Calculate smart sortino ratio 
    smart_sortino_ratio = qs.stats.smart_sortino(returns_df[config_name])

    # Calculate max drawdown
    max_drawdown=qs.stats.max_drawdown(returns_df[config_name])
    
    # Calculate Calmar 
    calmar = qs.stats.calmar(returns_df[config_name])
    
    # average loss
    avg_loss = qs.stats.avg_loss(returns_df[config_name])
    
    
    # volatility calculation 
    
    volatility = qs.stats.volatility(returns_df[config_name])
    
    
    # win loss ratio 
    win_loss_ratio =  qs.stats.win_loss_ratio(returns_df[config_name])
    
    
    performance_data = {
        'model_&_strategy': config_name,
        "max_drawdown": max_drawdown*100,
        "smart_sortino_ratio": smart_sortino_ratio,
        "cum_returns" :  cumulative_return,
        "sharpe_ratio": sharpe_ratio,
        "calmar": calmar,
        "avg_loss":avg_loss,
        "vol_annualized": volatility,
        "win_loss_ratio":win_loss_ratio
        }
    
    results_df = pd.DataFrame(performance_data , index=[0])
    return results_df

def save_results(dict_name, trading_strategy): 
    '''
    Save results after Backtester.run_backtest() has been performed.
    
    Arguments:
        dict_name: dict()
            Dictionary containing dictionaries of returns strategy, optimal weights, predictions, etc. for each configuration.
            ex) backtester_msr_linear
        
        trading_strategy: str
            Name of trading strategy & type of models run.
            ex) mvp_linear
            
    Returns:
        total_returns_df (pd.DataFrame): A dataframe containing realized returns for each period of time.
        total_optimal_weights_df (pd.DataFrame): A dataframe containing optimal weights for each period of time.
        total_predictions_df (pd.DataFrame): A dataframe containing predictions made for each period of time.

    Example Use Case:
        save_results(backtester_msr_linear, 'msr_linear')
    '''
    total_returns_df = pd.DataFrame()
    total_optimal_weights_df = pd.DataFrame()
    total_predictions_df = pd.DataFrame()
    
    for key in list(dict_name.returns_strategy.keys()):

        # Saving Realized returns
        returns_df = pd.DataFrame.from_dict(dict_name.returns_strategy[key])        
        total_returns_df[key] = returns_df["return"].values
        
        # Saving optimal weights 
        optimal_weights = dict_name.optimal_weights[key]
        total_optimal_weights_df[key] = optimal_weights["weight"].values
            
        # Saving the predictions made for that period of time 
        predictions_df = dict_name.dict_all_predictions[key]
        total_predictions_df["fret1d"] = predictions_df["fret1d"].values
        total_predictions_df[key] = predictions_df["fret1d_pred"].values
        
    # Saving all values
    total_returns_df = total_returns_df.set_index(returns_df.index)
    total_returns_df.to_csv(f"{trading_strategy}_returns_df.csv")
    
    total_optimal_weights_df = total_optimal_weights_df.set_index(optimal_weights.index)
    total_optimal_weights_df.to_csv(f"{trading_strategy}_optimal_weights_df.csv")
    
    total_predictions_df = total_predictions_df.set_index([predictions_df.date, predictions_df.permno])
    total_predictions_df.to_csv(f"{trading_strategy}_predictions_df.csv")

    return total_returns_df, total_optimal_weights_df, total_predictions_df