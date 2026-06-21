from __future__ import annotations

import glob
import os
import re

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

# =====================================================================
# 1. 出力フォルダ・ファイルの階層管理 (Backtest_XXXX フォルダ自動生成)
# =====================================================================
BASE_EXP_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "Experiment"))
os.makedirs(BASE_EXP_DIR, exist_ok=True)


def get_next_sequence_number(base_dir):
    """Experimentフォルダ内の既存フォルダを走査し、次の4桁の連番を決定します。"""
    folders = glob.glob(os.path.join(base_dir, "Backtest_*"))
    max_num = 0
    for f in folders:
        if os.path.isdir(f):
            match = re.search(r"Backtest_(\d{4})$", os.path.basename(f))
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
    return max_num + 1


seq_num = get_next_sequence_number(BASE_EXP_DIR)
SUB_EXP_DIR = os.path.join(BASE_EXP_DIR, f"Backtest_{seq_num:04d}")
os.makedirs(SUB_EXP_DIR, exist_ok=True)

# CSVの保存先
metric_csv_path = os.path.join(SUB_EXP_DIR, f"Backtest_Metric_{seq_num:04d}.csv")
price_csv_path = os.path.join(SUB_EXP_DIR, f"Backtest_Price_{seq_num:04d}.csv")


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

        # 内部ログから資産ベースのポートフォリオウェイトを算出
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

# %% Part 4. Mean Variance Optimization baseline
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
# 3. データ整形と CSV への個別保存
# =====================================================================
df_prices_raw = trade.pivot(index="date", columns="tic", values="close")
df_prices_raw.index = pd.to_datetime(df_prices_raw.index)

initial_stock_avg = df_prices_raw.iloc[0].mean()
df_prices_final = df_prices_raw.rename(
    columns={col: f"{col}_P" for col in df_prices_raw.columns}
)

for name, df_acc in account_values_results.items():
    norm_factor = initial_stock_avg / df_acc["account_value"].iloc[0]
    df_prices_final[f"{name}_P"] = (
        df_acc["account_value"].values * norm_factor
    )

mvo_norm_factor = initial_stock_avg / MVO_result["Mean Var"].iloc[0]
df_prices_final["mvo_P"] = MVO_result["Mean Var"].values * mvo_norm_factor

for agent_name, df_w in weights_results.items():
    for asset_col in df_w.columns:
        df_prices_final[f"{asset_col}_W_{agent_name}"] = df_w[asset_col]

df_prices_final.to_csv(price_csv_path)
print(f"✅ 価格・ウェイト統合データCSVを保存しました: {price_csv_path}")

metrics_rows = {}
for name in account_values_results.keys():
    metrics_rows[name] = pyfolio_like_metrics(account_values_results[name])
metrics_rows["mvo"] = pyfolio_like_metrics(
    pd.DataFrame({"account_value": MVO_result["Mean Var"]})
)

for tic in df_prices_raw.columns:
    tmp_df = pd.DataFrame({"account_value": df_prices_raw[tic]}).dropna()
    metrics_rows[tic] = pyfolio_like_metrics(tmp_df) if len(tmp_df) > 0 else {}

df_final_metric_csv = pd.DataFrame(metrics_rows)
df_final_metric_csv.to_csv(metric_csv_path)
print(f"✅ 評価指標CSVを保存しました: {metric_csv_path}")


# =====================================================================
# 4. プロットごとに個別のPDFファイルとして分離出力 (修正要件適用)
# =====================================================================
sns.set_theme(style="whitegrid")
all_agents = list(account_values_results.keys())
cmap = matplotlib.colormaps.get_cmap("Set1")


def save_single_plot(filename_suffix, draw_func, width=10):
    """個別のPDFプロットファイルを共通フォーマットで出力する関数"""
    fig, ax = plt.subplots(figsize=(width, 6))
    draw_func(ax)
    plt.tight_layout()
    pdf_path = os.path.join(
        SUB_EXP_DIR, f"Backtest_plot_{seq_num:04d}_{filename_suffix}.pdf"
    )
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   -> プロット保存完了: {pdf_path}")


print("📉 個別PDFプロットレポートの生成を開始します...")

# --- [1] Price ---
def draw_price(ax):
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
    ax.set_title("Standardized Agent & Baseline Prices", weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price Scale")
    ax.legend(loc="upper left")


save_single_plot("Price", draw_price)


# --- [2] LogReturn ---
def draw_log_return(ax):
    for idx, name in enumerate(all_agents):
        df_acc = account_values_results[name]
        log_ret = np.log(
            df_acc["account_value"] / df_acc["account_value"].shift(1)
        ).dropna()
        ax.plot(
            df_prices_final.index[-len(log_ret) :],
            log_ret,
            label=name.upper(),
            alpha=0.5,
            color=cmap(idx),
        )
    ax.set_title("Log Return Overlay Comparison", weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.legend(loc="upper left")


save_single_plot("LogReturn", draw_log_return)


# --- [3] SharpeRatio ---
def draw_sharpe(ax):
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
    ax.set_title("Rolling 20-Day Sharpe Ratio Comparison", weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left")


save_single_plot("SharpeRatio", draw_sharpe)


# --- [4] MaximumDrawdown (★再出力・追加) ---
def draw_max_drawdown(ax):
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
    ax.set_title("Portfolio Drawdown Profile", weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown Rate")
    ax.legend(loc="lower left")


save_single_plot("MaximumDrawdown", draw_max_drawdown)


# --- [5] ChangeWeightsPortfolio ---
def draw_change_weights(ax):
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
    ax.set_title("Rebalancing Activity", weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Absolute Sum of Weight Changes")
    ax.legend(loc="upper left")


save_single_plot("ChangeWeightsPortfolio", draw_change_weights)


# --- [6] Weight (各アルゴリズムごとに個別の積み上げ面グラフ計5枚を生成) ---
for name in all_agents:
    if name in weights_results:

        def make_draw_weight_func(agent_name):
            def _draw(ax_w):
                df_w_target = weights_results[agent_name]
                ax_w.stackplot(
                    df_w_target.index,
                    [df_w_target[col] for col in df_w_target.columns],
                    labels=df_w_target.columns,
                    alpha=0.8,
                )
                ax_w.set_title(
                    f"Portfolio Allocation Changes Over Time ({agent_name.upper()})",
                    weight="bold",
                )
                ax_w.set_ylabel("Weight Ratio (0.0 - 1.0)")
                ax_w.set_ylim(0, 1)
                ax_w.legend(
                    loc="lower left", bbox_to_anchor=(1, 0), ncol=1, fontsize=9
                )

            return _draw

        # アルゴリズム名をファイル名に明示して個別に保存
        save_single_plot(f"Weight_{name.upper()}", make_draw_weight_func(name), width=12)

print(
    f"✨ すべてのデータおよび個別PDFプロットは次のフォルダへ格納されました: {SUB_EXP_DIR}"
)