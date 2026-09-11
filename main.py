from utils.data_loader import download_data
from agents.momentum_agent import MomentumAgent
from backtester.backtest import run_backtest

df = download_data()

df.to_csv("data/PFE.csv")

agent = MomentumAgent()

results = run_backtest(df, agent)

print("Backtest Results:")
print("Sharpe Ratio:", results["Sharpe Ratio"])
print("Max Drawdown:", results["Max Drawdown"])
print("Alpha:", results["Alpha"])
print("Beta:", results["Beta"])