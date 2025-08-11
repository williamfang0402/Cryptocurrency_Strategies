import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Importing the data for cryptocurrencies with Yahoo Finance
univ = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD']
crypto_px = yf.download(univ, start="2024-01-01")
crypto_px['Close'][univ].plot(figsize=(10,6))

plt.title('Cryptocurrencies Price History')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
# plt.show()

monthly_returns = crypto_px.resample('ME').ffill().pct_change()
print(monthly_returns)

# UMD strategy: Buy stocks with high returns in the past 12 months, and short those with poor returns
# while ignoring the most recent month’s return.
def get_umd_rank(monthly_returns):
    formation_period_returns = monthly_returns.shift(2).rolling(window=11).apply(lambda x: (1+x).prod()-1)
    ranks = formation_period_returns.rank(axis=1, ascending=False, method='first')
    return ranks

def form_portfolio(ranks, num_quantiles=5):
    num_assets = ranks.shape[1]
    top_quantile = num_assets / num_quantiles
    bottom_quantile = num_assets - top_quantile

    long_portfolio = (ranks <= top_quantile).astype(int)
    short_portfolio = (ranks > bottom_quantile).astype(int) * -1

    long_weights = long_portfolio.div(long_portfolio.sum(axis=1), axis=0).fillna(0)
    short_weights = short_portfolio.div(short_portfolio.abs().sum(axis=1), axis=0).fillna(0)

    portfolio_weights = long_weights + short_weights

    return portfolio_weights

# Backtesting
umd_ranks = get_umd_rank(monthly_returns)
portfolio_weights = form_portfolio(umd_ranks)

strategy_returns = (portfolio_weights.shift(1) * monthly_returns).sum(axis=1)

transaction_cost_per_trade = 0.002  # assuming 20 bps
turnover = portfolio_weights.diff().abs().sum(axis=1) / 2
total_transaction_cost = turnover * transaction_cost_per_trade
adjusted_strategy_ret = strategy_returns - total_transaction_cost

# Performance Evaluation
cum_returns = (1 + adjusted_strategy_ret).cumprod()

annualized_return = np.power(cum_returns.iloc[-1], (12 / len(cum_returns))) - 1
annualized_volatility = adjusted_strategy_ret.std() * np.sqrt(12)
sharpe_ratio = annualized_return / annualized_volatility

rolling_max = cum_returns.cummax()
drawdown = cum_returns / rolling_max - 1
max_drawdown = drawdown.min()

print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")