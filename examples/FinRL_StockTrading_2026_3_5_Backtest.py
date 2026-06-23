from __future__ import annotations

from datetime import datetime
import os
import re
import sys
import copy

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
# 1. 引数の検証と対象Experiment/Backtestフォルダ・ファイルの決定
# =====================================================================
if len(sys.argv) < 2:
    print(
        "❌ エラー: 管理番号(4桁)を指定する引数が不足しています。\n使用例: python FinRL_StockTrading_2026_4_Backtest.py 0001"
    )
    sys.exit(1)

seq_str = sys.argv[1]
if not (seq_str.isdigit() and len(seq_str) == 4):
    print("❌ エラー: 引数は4桁の数字で指定してください。(例: 0001)")
    sys.exit(1)

# ベースとなる親Experiment_XXXXフォルダのパス特定と存在チェック
BASE_EXP_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "Experiment"))
SUB_EXP_DIR = os.path.join(BASE_EXP_DIR, f"Experiment_{seq_str}")

train_csv_path = os.path.join(SUB_EXP_DIR, f"train_data_{seq_str}.csv")
trade_csv_path = os.path.join(SUB_EXP_DIR, f"trade_data_{seq_str}.csv")
readme_md_path = os.path.join(SUB_EXP_DIR, f"README_{seq_str}.md")

if not os.path.exists(SUB_EXP_DIR):
    print(f"❌ エラー: 指定された親フォルダ '{SUB_EXP_DIR}' が存在しません。先にdata.pyを実行してください。")
    sys.exit(1)

if not (os.path.exists(train_csv_path) and os.path.exists(trade_csv_path)):
    print(f"❌ エラー: フォルダ内に train_data_{seq_str}.csv または trade_data_{seq_str}.csv が見つかりません。")
    sys.exit(1)

# ★【新規追加要件】Experiment_XXXX 内に成果物格納用の Backtest_XXXX フォルダを自動作成
BACKTEST_OUT_DIR = os.path.join(SUB_EXP_DIR, f"Backtest_{seq_str}")
os.makedirs(BACKTEST_OUT_DIR, exist_ok=True)

# CSVの保存先（Backtest_XXXX フォルダ内へ変更）
metric_csv_path = os.path.join(BACKTEST_OUT_DIR, f"Backtest_Metric_{seq_str}.csv")
price_csv_path = os.path.join(BACKTEST_OUT_DIR, f"Backtest_Price_{seq_str}.csv")

# モデルの読み込み先 (Train.pyが出力したフォルダ)
CUSTOM_TRAINED_MODEL_DIR = os.path.join(SUB_EXP_DIR, "trained_models")


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
print(f"📂 データファイルをロード中...\n  - {train_csv_path}\n  - {trade_csv_path}")
train = pd.read_csv(train_csv_path)
trade = pd.read_csv(trade_csv_path)
train = train.set_index(train.columns[0])
train.index.names = [""]
trade = trade.set_index(trade.columns[0])
trade.index.names = [""]

# %% Part 2. Load trained agents
if_using_a2c = if_using_ddpg = if_using_ppo = if_using_td3 = if_using_sac = True

print(f"🤖 学習済みエージェントをロード中: {CUSTOM_TRAINED_MODEL_DIR}")
# trained_a2c = A2C.load(os.path.join(CUSTOM_TRAINED_MODEL_DIR, "agent_a2c")) if if_using_a2c else None
# trained_ddpg = DDPG.load(os.path.join(CUSTOM_TRAINED_MODEL_DIR, "agent_ddpg")) if if_using_ddpg else None
trained_ppo = PPO.load(os.path.join(CUSTOM_TRAINED_MODEL_DIR, "agent_ppo")) if if_using_ppo else None
# trained_td3 = TD3.load(os.path.join(CUSTOM_TRAINED_MODEL_DIR, "agent_td3")) if if_using_td3 else None
# trained_sac = SAC.load(os.path.join(CUSTOM_TRAINED_MODEL_DIR, "agent_sac")) if if_using_sac else None

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
    # "a2c": (trained_a2c, if_using_a2c),
    # "ddpg": (trained_ddpg, if_using_ddpg),
    "ppo": (trained_ppo, if_using_ppo),
    # "td3": (trained_td3, if_using_td3),
    # "sac": (trained_sac, if_using_sac),
}
account_values_results = {}
weights_results = {}

# =====================================================================
# 【確証版】バックテスト実行と、エージェント名に完全紐づいたウェイト計算
# =====================================================================
for name, (model, flag) in agents_dict.items():
    if flag:
        print(f"▶️ {name.upper()} の予測（バックテスト）を実行中...")
        # df_acc: 資産額推移, df_act: エージェント固有の売買行動ログ(確定値)
        df_acc, df_act = DRLAgent.DRL_prediction(
            model=model, environment=e_trade_gym
        )
        account_values_results[name] = df_acc

        # 環境の state_memory は使わず、このエージェントが出力した df_act から直接計算する
        dates = df_act.index
        ticker_list = list(df_act.columns)
        
        # テストデータの価格情報を参照
        df_prices = trade.pivot(index="date", columns="tic", values="close")
        df_prices.index = pd.to_datetime(df_prices.index)
        
        # エージェントごとに初期状態を完全に独立させてシミュレート
        current_cash = 1000000
        current_shares = {tic: 0 for tic in ticker_list}
        portfolio_data = []
        
        for date in dates:
            p_date = pd.to_datetime(date)
            if p_date not in df_prices.index:
                continue
                
            day_prices = df_prices.loc[p_date]
            
            # 当日の売買前の保有額を記録
            row_data = {"date": date}
            for tic in ticker_list:
                # 株数 × 終値 ＝ 銘柄の評価額
                row_data[tic] = current_shares[tic] * day_prices[tic]
            row_data["Cash"] = current_cash
            portfolio_data.append(row_data)
            
            # 当日の夜にエージェントのアクション（注文）を実行し、株数と現金を更新
            if date in df_act.index:
                actions = df_act.loc[date]
                for tic in ticker_list:
                    action = actions[tic]
                    price = day_prices[tic]
                    
                    if action > 0:  # 買い注文
                        buy_cost = action * price * 1.001  # 手数料0.1%
                        current_cash -= buy_cost
                        current_shares[tic] += action
                    elif action < 0:  # 売り注文
                        sell_gain = abs(action) * price * 0.999  # 手数料0.1%
                        current_cash += sell_gain
                        current_shares[tic] += action

        # 出来上がったデータを比率（ウェイト）に変換
        df_p_val = pd.DataFrame(portfolio_data)
        asset_cols = ["Cash"] + ticker_list
        
        # 金額ベースのポートフォリオから比率(0.0~1.0)を算出
        df_w = df_p_val[asset_cols].div(df_p_val[asset_cols].sum(axis=1), axis=0)
        df_w["date"] = pd.to_datetime(df_p_val["date"])
        df_w.set_index("date", inplace=True)
        
        # 💡 これにより、現在ループしているエージェント(name)の純粋なデータのみが100%確実に格納されます
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
        weight_bounds=(0, 1),
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
# 3. データ整形と CSV への個別保存 (Backtest_XXXXフォルダ内)
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
# 4. 実験管理用 README.md ファイルへの統計量集計・追記 (Step3)
# =====================================================================
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

target_metrics_map = {
    "Annual return": "Annual return",
    "Cumulative returns": "Cumulative returns",
    "Annual Volatility": "annual volatility",
    "Sharpe Ratio": "sharpe ratio",
    "Calmar Ratio": "calmar ratio",
    "Stability": "stability",
    "Max Drawdown": "max drawdown",
    "Omega Ratio": "omega ratio",
    "Sortino Ratio": "sortino ratio",
    "Skew": "skew",
    "Kurtosis": "kurtosis",
    "Tail Ratio": "tail ratio",
    "Daily value at risk": "daily value at risk"
}

table_lines = [
    "| Metric | 平均値 | 最小値 | 第1四分位数 (25%) | 第2四分位数 (50%) | 第3四分位数 (75%) | 最大値 |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
]

for csv_idx, display_name in target_metrics_map.items():
    if csv_idx in df_final_metric_csv.index:
        row_data = pd.to_numeric(df_final_metric_csv.loc[csv_idx], errors='coerce').dropna()
        if not row_data.empty:
            mean_val = row_data.mean()
            min_val = row_data.min()
            q1 = row_data.quantile(0.25)
            q2 = row_data.quantile(0.50)
            q3 = row_data.quantile(0.75)
            max_val = row_data.max()
            
            table_lines.append(
                f"| {display_name} | {mean_val:.6f} | {min_val:.6f} | {q1:.6f} | {q2:.6f} | {q3:.6f} | {max_val:.6f} |"
            )
        else:
            table_lines.append(f"| {display_name} | N/A | N/A | N/A | N/A | N/A | N/A |")
    else:
        table_lines.append(f"| {display_name} | N/A | N/A | N/A | N/A | N/A | N/A |")

metrics_table_str = "\n".join(table_lines)

backtest_readme_append = f"""
## 📊 Step 3: バックテスト・評価フェーズ

バックテスト実行結果のサマリーおよび各種評価指標の統計量レポートです。

- **実行日時 (秒まで)**: {current_time_str}
- **バックテスト成果物格納先**: `Experiment_{seq_str}/Backtest_{seq_str}/`

### 📈 主要評価指標の統計サマリー（表形式）
{metrics_table_str}

---
"""

if os.path.exists(readme_md_path):
    with open(readme_md_path, "a", encoding="utf-8") as f:
        f.write(backtest_readme_append)
    print(f"📝 バックテストフェーズの評価統計量を README に追記しました: {readme_md_path}")
else:
    with open(readme_md_path, "w", encoding="utf-8") as f:
        f.write(backtest_readme_append)
    print(f"⚠️ READMEが見つからなかったため、新規に作成して保存しました: {readme_md_path}")


# =====================================================================
# 5. プロットごとに個別のPDFファイルとして分離出力 (Backtest_XXXXフォルダ内)
# =====================================================================
sns.set_theme(style="whitegrid")
all_agents = list(account_values_results.keys())
cmap = matplotlib.colormaps.get_cmap("Set1")


# 💡 修正箇所：プロットの比率を 11:7 に固定（不要な引数を廃止）
def save_single_plot(filename_suffix, draw_func):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    draw_func(ax)
    plt.tight_layout()
    # 保存先を BACKTEST_OUT_DIR (Backtest_XXXX) に指定
    pdf_path = os.path.join(
        BACKTEST_OUT_DIR, f"Backtest_plot_{seq_str}_{filename_suffix}.pdf"
    )
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   -> プロット保存完了: {pdf_path}")


print("\n📉 個別PDFプロットレポートの生成を開始します...")

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
    ax.set_title("Standardized Agent & Baseline Prices", weight="bold", fontsize=20)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Normalized Price Scale", fontsize=16)
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
    ax.set_title("Log Return Overlay Comparison", weight="bold", fontsize=20)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Log Return", fontsize=16)
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
        
        # 💡 【追加箇所】データが空（0件）の場合はエラーになるので描画をスキップする
        if len(rolling_sharpe) == 0:
            continue
            
        ax.plot(
            df_prices_final.index[-len(rolling_sharpe) :],
            rolling_sharpe,
            label=name.upper(),
            color=cmap(idx),
            alpha=0.7,
        )
    ax.set_title("Rolling 20-Day Sharpe Ratio Comparison", weight="bold", fontsize=20)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Sharpe Ratio", fontsize=16)
    ax.legend(loc="upper left")

save_single_plot("SharpeRatio", draw_sharpe)

# --- [4] MaximumDrawdown ---
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
    ax.set_title("Portfolio Drawdown Profile", weight="bold", fontsize=20)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Drawdown Rate", fontsize=16)
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
    ax.set_title("Rebalancing Activity", weight="bold", fontsize=20)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Absolute Sum of Weight Changes", fontsize=16)
    ax.legend(loc="upper left")

save_single_plot("ChangeWeightsPortfolio", draw_change_weights)

# --- [6] Weight ---
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
                    fontsize=20
                )
                ax_w.set_xlabel("Date", fontsize=16)
                ax_w.set_ylabel("Weight Ratio (0.0 - 1.0)", fontsize=16)
                ax_w.set_ylim(0, 1)
                ax_w.legend(
                    loc="upper right", fontsize=16
                )
            return _draw

        # 💡 修正箇所：個別の width 引数の指定を外して 11:7 に統一
        save_single_plot(f"Weight_{name.upper()}", make_draw_weight_func(name))

print(f"✨ すべてのバックテスト成果物は指定フォルダへ格納されました: {BACKTEST_OUT_DIR}")