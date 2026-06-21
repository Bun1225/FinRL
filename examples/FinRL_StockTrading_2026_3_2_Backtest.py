"""
Stock NeurIPS2018 Part 3. Backtest

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introducing how to use the agents we trained to do backtest, and compare with baselines such as
Mean Variance Optimization and DJIA index.

このファイルは元のファイル(FinRL_StockTrading_2026_3_Backtest.py)の一部を改変して、ポートフォリオの配分を小数点表記でCSVに保存し、積み上げ面グラフを描画する機能を追加したものです。
"""

from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, TRADE_START_DATE, TRADE_END_DATE
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


# %% 🔥 [安全化修正] ポートフォリオウェイト（小数点表記CSV & グラフ）保存用関数
def save_portfolio_weights_and_plot(env_instance, df_actions, agent_name):
    """FinRLの環境内部ログからウェイトを抽出し、小数点表記CSVと積み上げ面グラフを保存します。"""
    os.makedirs("results", exist_ok=True)

    dates = df_actions.index
    # 【修正箇所】環境の内部プロパティを一切使わず、df_actionsの列から銘柄リストを安全に取得
    ticker_list = list(df_actions.columns)
    stock_dim = len(ticker_list)
    state_memory = env_instance.state_memory[1:]

    portfolio_data = []
    for i, date in enumerate(dates):
        if i >= len(state_memory):
            break
        current_state = state_memory[i]

        cash = current_state[0]
        prices = current_state[1 : 1 + stock_dim]
        shares = current_state[1 + stock_dim : 1 + 2 * stock_dim]

        row_data = {"date": date, "Cash": cash}
        for j, ticker in enumerate(ticker_list):
            row_data[ticker] = shares[j] * prices[j]
        portfolio_data.append(row_data)

    df_portfolio_value = pd.DataFrame(portfolio_data)
    asset_cols = ["Cash"] + ticker_list

    # 小数点表記 (0.0 〜 1.0) の割合を算出
    df_weights = df_portfolio_value[asset_cols].div(
        df_portfolio_value[asset_cols].sum(axis=1), axis=0
    )
    df_weights["date"] = pd.to_datetime(df_portfolio_value["date"])
    df_weights.set_index("date", inplace=True)

    # 💾 小数点表記CSVの保存
    csv_path = f"results/portfolio_weights_{agent_name}.csv"
    df_weights.to_csv(csv_path)
    print(f"💾 Saved weights CSV for {agent_name} to {csv_path}")

    # 📉 積み上げ面グラフの描画・保存
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="white")
    colors = sns.color_palette("husl", len(asset_cols))

    plt.stackplot(
        df_weights.index,
        [df_weights[col] * 100 for col in asset_cols],
        labels=asset_cols,
        colors=colors,
        alpha=0.85,
    )
    plt.title(
        f"Portfolio Weight Transitions - {agent_name.upper()}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Allocation Percentage (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.xlim(df_weights.index.min(), df_weights.index.max())
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=11)
    plt.tight_layout()

    plot_path = f"results/portfolio_weight_transitions_{agent_name}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✨ Saved weights Plot for {agent_name} to {plot_path}")


# %% Part 1. Load data

train = pd.read_csv("train_data_JP_HIGH_LOW_VOL.csv")
trade = pd.read_csv("trade_data_JP_HIGH_LOW_VOL.csv")

train = train.set_index(train.columns[0])
train.index.names = [""]
trade = trade.set_index(trade.columns[0])
trade.index.names = [""]

# %% Part 2. Load trained agents

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

# %% Part 3. Backtesting - DRL agents

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_trade_gym = StockTradingEnv(
    df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
)

# --- A2C ---
if if_using_a2c:
    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
        model=trained_a2c, environment=e_trade_gym
    )
    save_portfolio_weights_and_plot(e_trade_gym, df_actions_a2c, "a2c")
else:
    df_account_value_a2c, df_actions_a2c = (None, None)

# --- DDPG ---
if if_using_ddpg:
    df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
        model=trained_ddpg, environment=e_trade_gym
    )
    save_portfolio_weights_and_plot(e_trade_gym, df_actions_ddpg, "ddpg")
else:
    df_account_value_ddpg, df_actions_ddpg = (None, None)

# --- PPO ---
if if_using_ppo:
    df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
        model=trained_ppo, environment=e_trade_gym
    )
    save_portfolio_weights_and_plot(e_trade_gym, df_actions_ppo, "ppo")
else:
    df_account_value_ppo, df_actions_ppo = (None, None)

# --- TD3 ---
if if_using_td3:
    df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
        model=trained_td3, environment=e_trade_gym
    )
    save_portfolio_weights_and_plot(e_trade_gym, df_actions_td3, "td3")
else:
    df_account_value_td3, df_actions_td3 = (None, None)

# --- SAC ---
if if_using_sac:
    df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
        model=trained_sac, environment=e_trade_gym
    )
    save_portfolio_weights_and_plot(e_trade_gym, df_actions_sac, "sac")
else:
    df_account_value_sac, df_actions_sac = (None, None)


# %% Part 4. Mean Variance Optimization baseline


def process_df_for_mvo(df):
    return df.pivot(index="date", columns="tic", values="close")


def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):
        for i in range(Rows - 1):
            StockReturn[i, j] = (
                (StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]
            ) * 100
    return StockReturn


StockData = process_df_for_mvo(train)
TradeData = process_df_for_mvo(trade)

arStockPrices = np.asarray(StockData)
[Rows, Cols] = arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

meanReturns = np.mean(arReturns, axis=0)
covReturns = np.cov(arReturns, rowvar=False)

np.set_printoptions(precision=3, suppress=True)
print("Mean returns of assets in portfolio\n", meanReturns)

from pypfopt.efficient_frontier import EfficientFrontier

ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
raw_weights_mean = ef_mean.max_sharpe()
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array(
    [1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))]
)

LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
Initial_Portfolio = np.multiply(mvo_weights, LastPrice)

Portfolio_Assets = TradeData @ Initial_Portfolio
MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

# %% Part 5. DJIA index baseline

import yfinance as yf

# df_dji = yf.download("^DJI", start=TRADE_START_DATE, end=TRADE_END_DATE)
# df_dji = df_dji[["Close"]].reset_index()
# df_dji.columns = ["date", "close"]
# df_dji["date"] = df_dji["date"].astype(str)
# fst_day = df_dji["close"].iloc[0]
# dji = pd.merge(
#     df_dji["date"],
#     df_dji["close"].div(fst_day).mul(1000000),
#     how="outer",
#     left_index=True,
#     right_index=True,
# ).set_index("date")

# %% Part 6. Compare results

df_result_a2c = (
    df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
    if if_using_a2c
    else None
)
df_result_ddpg = (
    df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
    if if_using_ddpg
    else None
)
df_result_ppo = (
    df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
    if if_using_ppo
    else None
)
df_result_td3 = (
    df_account_value_td3.set_index(df_account_value_td3.columns[0])
    if if_using_td3
    else None
)
df_result_sac = (
    df_account_value_sac.set_index(df_account_value_sac.columns[0])
    if if_using_sac
    else None
)

result = pd.DataFrame(
    {
        "a2c": df_result_a2c["account_value"] if if_using_a2c else None,
        "ddpg": df_result_ddpg["account_value"] if if_using_ddpg else None,
        "ppo": df_result_ppo["account_value"] if if_using_ppo else None,
        "td3": df_result_td3["account_value"] if if_using_td3 else None,
        "sac": df_result_sac["account_value"] if if_using_sac else None,
        "mvo": MVO_result["Mean Var"],
        # "dji": dji["close"],
    }
)

print("\n=== Backtest Results ===")
print(result)

# %% Part 7. Plot

plt.rcParams["figure.figsize"] = (10, 7)
plt.figure()
result.plot()
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to backtest_result.png")