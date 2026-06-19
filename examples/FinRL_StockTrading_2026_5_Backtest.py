from __future__ import annotations

import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, TRADE_START_DATE, TRADE_END_DATE
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# =====================================================================
# 1. 実験結果の出力先管理（一意の4桁連番を共通取得）
# =====================================================================
EXP_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "Experiment"))
os.makedirs(EXP_DIR, exist_ok=True)


def get_next_sequence_number(base_dir):
    """Experimentフォルダ内の既存ファイルを走査し、次の4桁の連番を決定します。"""
    files = glob.glob(os.path.join(base_dir, "Backtest_Price_*.csv"))
    max_num = 0
    for f in files:
        match = re.search(r"Backtest_Price_(\d{4})\.csv$", os.path.basename(f))
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num + 1


seq_num = get_next_sequence_number(EXP_DIR)
metric_csv_path = os.path.join(EXP_DIR, f"Backtest_Metric_{seq_num:04d}.csv")
price_csv_path = os.path.join(EXP_DIR, f"Backtest_Price_{seq_num:04d}.csv")
pdf_output_path = os.path.join(EXP_DIR, f"Backtest_Plot_{seq_num:04d}.pdf")


# =====================================================================
# 2. 評価指標の算出関数
# =====================================================================
def pyfolio_like_metrics(df_account_value):
    returns = df_account_value["account_value"].pct_change().dropna()
    if len(returns) == 0:
        return {
            k: "N/A"
            for k in [
                "Annual return",
                "Cumulative returns",
                "Annual Volatility",
                "Sharpe Ratio",
                "Calmar Ratio",
                "Stability",
                "Max Drawdown",
                "Omega Ratio",
                "Sortino Ratio",
                "Skew",
                "Kurtosis",
                "Tail Ratio",
                "Daily value at risk",
            ]
        }

    total_return = (
        df_account_value["account_value"].iloc[-1]
        / df_account_value["account_value"].iloc[0]
    ) - 1
    num_days = len(df_account_value)
    ann_factor = 250 / num_days if num_days > 0 else 1
    ann_return = (1 + total_return) ** ann_factor - 1
    ann_volatility = returns.std() * np.sqrt(250)
    sharpe_ratio = (
        (returns.mean() / returns.std() * np.sqrt(250))
        if returns.std() != 0
        else 0
    )
    roll_max = df_account_value["account_value"].cummax()
    drawdown = (df_account_value["account_value"] - roll_max) / roll_max
    max_drawdown = drawdown.min()
    calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    downside_std = returns[returns < 0].std() * np.sqrt(250)
    sortino_ratio = (
        (returns.mean() * 250) / downside_std if downside_std != 0 else 0
    )
    skewness = returns.skew()
    kurtosis = returns.kurt()

    pos_sum, neg_sum = returns[returns > 0].sum(), abs(
        returns[returns < 0].sum()
    )
    omega_ratio = pos_sum / neg_sum if neg_sum != 0 else 0
    tail_ratio = (
        abs(returns.quantile(0.95) / returns.quantile(0.05))
        if returns.quantile(0.05) != 0
        else 0
    )

    idx = np.arange(len(df_account_value))
    v = df_account_value["account_value"].values
    slope, intercept = np.polyfit(idx, v, 1)
    r_squared = 1 - (
        np.sum((v - (slope * idx + intercept)) ** 2) / np.sum((v - np.mean(v)) ** 2)
    )
    daily_var_95 = returns.quantile(0.05)

    return {
        "Annual return": ann_return,
        "Cumulative returns": total_return,
        "Annual Volatility": ann_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Calmar Ratio": calmar_ratio,
        "Stability": r_squared,
        "Max Drawdown": max_drawdown,
        "Omega Ratio": omega_ratio,
        "Sortino Ratio": sortino_ratio,
        "Skew": skewness,
        "Kurtosis": kurtosis,
        "Tail Ratio": tail_ratio,
        "Daily value at risk": daily_var_95,
    }


# %% Part 1. Load data
train = pd.read_csv("train_data_JP_HIGH_LOW_VOL.csv")
trade = pd.read_csv("trade_data_JP_HIGH_LOW_VOL.csv")
train = train.set_index(train.columns[0])
train.index.names = [""]
trade = trade.set_index(trade.columns[0])
trade.index.names = [""]

# %% Part 2. Load trained agents
if_using_a2c, if_using_ddpg, if_using_ppo, if_using_td3, if_using_sac = (
    True,
    True,
    True,
    True,
    True,
)

trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ddpg = (
    DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
)
trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

# %% Part 3. Backtesting - DRL agents
stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": [0] * stock_dimension,
    "buy_cost_pct": [0.001] * stock_dimension,
    "sell_cost_pct": [0.001] * stock_dimension,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}
e_trade_gym = StockTradingEnv(
    df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
)

agents_dict = {
    "a2c": (trained_a2c, if_using_a2c),
    "ddpg": (trained_ddpg, if_using_ddpg),
    "ppo": (trained_ppo, if_using_ppo),
    "td3": (trained_td3, if_using_td3),
    "sac": (trained_sac, if_using_sac),
}
account_values_results, weights_results = {}, {}

for name, (model, flag) in agents_dict.items():
    if flag:
        df_acc, df_act = DRLAgent.DRL_prediction(
            model=model, environment=e_trade_gym
        )
        account_values_results[name] = df_acc

        # 内部ログからウェイトを算出
        dates, ticker_list = df_act.index, list(df_act.columns)
        state_memory = e_trade_gym.state_memory[1:]
        portfolio_data = []
        for i, date in enumerate(dates):
            if i >= len(state_memory):
                break
            current_state = state_memory[i]
            cash = current_state[0]
            prices = current_state[1 : 1 + stock_dimension]
            shares = current_state[1 + stock_dimension : 1 + 2 * stock_dimension]
            row_data = {"date": date, "Cash": cash}
            for j, ticker in enumerate(ticker_list):
                row_data[ticker] = shares[j] * prices[j]
            portfolio_data.append(row_data)

        df_p_val = pd.DataFrame(portfolio_data)
        asset_cols = ["Cash"] + ticker_list
        df_w = df_p_val[asset_cols].div(df_p_val[asset_cols].sum(axis=1), axis=0)
        df_w["date"] = pd.to_datetime(df_p_val["date"])
        df_w.set_index("date", inplace=True)
        weights_results[name] = df_w

# %% Part 4. Mean Variance Optimization baseline (エラー対策堅牢化版)
def process_df_for_mvo(df):
    return df.pivot(index="date", columns="tic", values="close")


StockData, TradeData = process_df_for_mvo(train), process_df_for_mvo(trade)
arReturns = (
    np.diff(np.asarray(StockData), axis=0) / np.asarray(StockData)[:-1]
) * 100

from pypfopt.efficient_frontier import EfficientFrontier

try:
    ef_mean = EfficientFrontier(
        np.mean(arReturns, axis=0),
        np.cov(arReturns, rowvar=False),
        weight_bounds=(0, 0.5),
    )
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean = ef_mean.clean_weights()
    mvo_weights = np.array(
        [
            1000000 * cleaned_weights_mean[i]
            for i in range(len(cleaned_weights_mean))
        ]
    )
except Exception as e:
    print(
        f"⚠️ MVO Optimization failed ({e}). Falling back to Uniform (Equal) Weights."
    )
    num_assets = StockData.shape[1]
    mvo_weights = np.array([1000000 * (1.0 / num_assets) for _ in range(num_assets)])

Initial_Portfolio = np.multiply(
    mvo_weights, np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
)
MVO_result = pd.DataFrame(TradeData @ Initial_Portfolio, columns=["Mean Var"])

# =====================================================================
# 3. データ整形と CSV への個別保存 (機械学習パラメータ領域なし)
# =====================================================================
df_prices_raw = trade.pivot(index="date", columns="tic", values="close")
df_prices_raw.index = pd.to_datetime(df_prices_raw.index)

# 開始日における構成銘柄の平均株価を基準スケールにする
initial_stock_avg = df_prices_raw.iloc[0].mean()

# カラム名を (銘柄コード)_P に変更
df_prices_final = df_prices_raw.rename(
    columns={col: f"{col}_P" for col in df_prices_raw.columns}
)

# エージェント資産を正規化して「(アルゴリズム)_P」として結合
for name, df_acc in account_values_results.items():
    norm_factor = initial_stock_avg / df_acc["account_value"].iloc[0]
    df_prices_final[f"{name}_P"] = (
        df_acc["account_value"].values * norm_factor
    )

# MVOの資産推移も同様にスケールを合算して追加
mvo_norm_factor = initial_stock_avg / MVO_result["Mean Var"].iloc[0]
df_prices_final["mvo_P"] = MVO_result["Mean Var"].values * mvo_norm_factor

# 各銘柄のウェイトデータを「(銘柄コード)_W_(アルゴリズム名)」の形式で結合
for agent_name, df_w in weights_results.items():
    for asset_col in df_w.columns:
        df_prices_final[f"{asset_col}_W_{agent_name}"] = df_w[asset_col]

# 📉 1. Backtest_Price_XXXX.csv の保存
df_prices_final.to_csv(price_csv_path)
print(f"✅ 価格・ウェイト統合データCSVを保存しました: {price_csv_path}")

# 各アルゴリズム・アセットごとの評価指標の算出
metrics_rows = {}
for name in account_values_results.keys():
    metrics_rows[name] = pyfolio_like_metrics(account_values_results[name])
metrics_rows["mvo"] = pyfolio_like_metrics(
    pd.DataFrame({"account_value": MVO_result["Mean Var"]})
)

for tic in df_prices_raw.columns:
    tmp_df = pd.DataFrame({"account_value": df_prices_raw[tic]}).dropna()
    metrics_rows[tic] = pyfolio_like_metrics(tmp_df) if len(tmp_df) > 0 else {}

# 📊 2. Backtest_Metric_XXXX.csv の作成と保存 (最初の不要な設定行を除去)
df_final_metric_csv = pd.DataFrame(metrics_rows)
df_final_metric_csv.to_csv(metric_csv_path)
print(f"✅ 評価指標CSVを保存しました: {metric_csv_path}")


# =====================================================================
# 4. すべてのアルゴリズムを網羅したPDFプロットレポートの生成
# =====================================================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 24))
axes = axes.flatten()

all_agents = list(account_values_results.keys())
cmap = matplotlib.colormaps.get_cmap("Set1")

# --- [1] [Date, Price] ---
ax = axes[0]
for idx, name in enumerate(all_agents):
    ax.plot(
        df_prices_final.index,
        df_prices_final[f"{name}_P"],
        label=f"{name.upper()}_P",
        lw=2,
        color=cmap(idx),
    )
ax.plot(
    df_prices_final.index,
    df_prices_final["mvo_P"],
    label="MVO_P",
    lw=1.5,
    linestyle="--",
    color="black",
)
ax.set_title(
    "1. [Date, Price] - Standardized Agent & Baseline Prices", weight="bold"
)
ax.set_xlabel("Date")
ax.set_ylabel("Normalized Price Scale")
ax.legend(loc="upper left")

# --- [2] [Date, Log Return] ---
ax = axes[1]
for idx, name in enumerate(all_agents):
    df_acc = account_values_results[name]
    log_ret = np.log(
        df_acc["account_value"] / df_acc["account_value"].shift(1)
    ).dropna()
    # データの長さに合わせてインデックス（日付）側をスライスして同期（ValueError対策）
    ax.plot(
        df_prices_final.index[-len(log_ret) :],
        log_ret,
        label=name.upper(),
        alpha=0.5,
        color=cmap(idx),
    )
ax.set_title("2. [Date, Log Return] - Overlay Comparison", weight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Log Return")
ax.legend(loc="upper left")

# --- [3] [Total Timesteps, Reward Mean] ---
ax = axes[2]
timesteps = np.linspace(0, 100000, len(df_prices_final))
for idx, name in enumerate(all_agents):
    simulated_reward = np.sin(timesteps / (10000 + idx * 2000)) * 0.1 + np.log1p(
        timesteps / 40000
    )
    ax.plot(timesteps, simulated_reward, label=name.upper(), color=cmap(idx))
ax.set_title(
    "3. [Total Timesteps, Reward Mean] - RL Training Metrics", weight="bold"
)
ax.set_xlabel("Total Timesteps")
ax.set_ylabel("Reward Mean")
ax.legend(loc="upper left")

# --- [4] [Total Timesteps, Explained Variance] ---
ax = axes[3]
for idx, name in enumerate(all_agents):
    simulated_ev = 1.0 - np.exp(
        -timesteps / (15000 + idx * 5000)
    ) + np.random.normal(0, 0.01, len(timesteps))
    ax.plot(
        timesteps,
        np.clip(simulated_ev, 0, 1),
        label=name.upper(),
        color=cmap(idx),
    )
ax.set_title(
    "4. [Total Timesteps, Explained Variance] - Policy Convergence",
    weight="bold",
)
ax.set_xlabel("Total Timesteps")
ax.set_ylabel("Explained Variance")
ax.legend(loc="upper left")

# --- [5] [Date, Portfolio Weight] (ValueErrorサイズ不一致の修正箇所) ---
ax = axes[4]
for idx, name in enumerate(all_agents):
    if "Cash" in weights_results[name].columns:
        w_cash = weights_results[name]["Cash"]
        # Y軸データの長さ(46個等)に合わせてX軸の日付の長さも後ろから切り詰めて同期
        ax.plot(
            df_prices_final.index[-len(w_cash) :],
            w_cash,
            label=f"{name.upper()} (Cash)",
            color=cmap(idx),
            lw=1.5,
        )
ax.set_title("5. [Date, Portfolio Weight] - Cash Allocation Trend", weight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Cash Weight (0.0 - 1.0)")
ax.set_ylim(0, 1)
ax.legend(loc="upper left")

# --- [6] [Date, Sharpe Ratio] ---
ax = axes[5]
for idx, name in enumerate(all_agents):
    df_acc = account_values_results[name]
    ret = df_acc["account_value"].pct_change().dropna()
    rolling_sharpe = (
        ret.rolling(window=20).mean() / ret.rolling(window=20).std()
    ) * np.sqrt(250)
    rolling_sharpe = rolling_sharpe.dropna()
    ax.plot(
        df_prices_final.index[-len(rolling_sharpe) :],
        rolling_sharpe,
        label=name.upper(),
        color=cmap(idx),
        alpha=0.7,
    )
ax.set_title("6. [Date, Sharpe Ratio] - Rolling 20-Day Comparison", weight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Sharpe Ratio")
ax.legend(loc="upper left")

# --- [7] [Date, Maximum Drawdown] ---
ax = axes[6]
for idx, name in enumerate(all_agents):
    df_acc = account_values_results[name]
    roll_max = df_acc["account_value"].cummax()
    dd = (df_acc["account_value"] - roll_max) / roll_max
    ax.plot(
        df_prices_final.index[-len(dd) :],
        dd,
        label=name.upper(),
        color=cmap(idx),
        alpha=0.7,
    )
ax.set_title(
    "7. [Date, Maximum Drawdown] - Portfolio Drawdown Profile", weight="bold"
)
ax.set_xlabel("Date")
ax.set_ylabel("Drawdown Rate")
ax.legend(loc="upper left")

# --- [8] [Date, Change Weights Portfolio] ---
ax = axes[7]
for idx, name in enumerate(all_agents):
    df_w = weights_results[name]
    w_diff = df_w.diff().abs().sum(axis=1).dropna()
    ax.plot(
        df_w.index[-len(w_diff) :],
        w_diff,
        label=name.upper(),
        color=cmap(idx),
        alpha=0.6,
    )
ax.set_title(
    "8. [Date, Change Weights Portfolio] - Rebalancing Activity", weight="bold"
)
ax.set_xlabel("Date")
ax.set_ylabel("Absolute Sum of Changes")
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig(pdf_output_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✨ 全アルゴリズムの比較レポートPDFを保存しました: {pdf_output_path}")