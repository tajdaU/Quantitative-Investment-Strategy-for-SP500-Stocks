# Quantitative Investment Strategy for S&P500 Stocks 

## Overview

This repository presents a comprehensive quantitative investment strategy designed to outperform a market-cap-weighted portfolio of 50 S&P500 stocks over the period from January 2016 to December 2022. The strategy is crafted with specific constraints, ensuring a disciplined and risk-controlled approach.

## Key Constraints

- **Factor Exposure Control:** The strategy maintains a maximum factor exposure deviation of 0.05 for each considered factor.
- **Portfolio Weight Management:** Asset weights are controlled with a maximum deviation of 10% from the benchmark (BNCH) weight for each asset.
- **Risk Management:** The strategy enforces a maximum drawdown relative to BNCH of 1% per month.

## Modeling Approach

To achieve these objectives, the strategy employs a combination of machine learning models, including Extreme Gradient Boosting Regressor, Lasso, Ridge, and ElasticNet regression. These models are selected, trained, and optimized on historical data from January 2000 to December 2015.

## Performance Metrics

The performance of both the models and the trading strategy is rigorously evaluated using a range of key performance metrics, including:

- **Mean Absolute Error**
- **Sharpe Ratio**
- **Maximum Drawdown**
- **Smart Sortino Ratio (Sortino Ratio)**
- **Sharpe-Calmar Ratio**
- **Win Loss Ratio**

## Out-of-Sample Testing

Once a model and strategy are selected, trained, and optimized on the in-sample period (January 2000 to December 2015), the same model and strategy are applied without any changes or adjustments to the out-of-sample period from January 2016 to December 2021.
