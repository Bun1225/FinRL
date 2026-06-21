"""
FinRL StockTrading 2026 - Compare 5 Agents Backtest (個別独立プロット出力版)

各Experimentフォルダ内にすでに生成済みのバックテスト結果CSVを直接読み込み、
エージェントごと・指標ごとに完全に独立したファイルとして比較プロットを瞬時に出力します。
"""

from __future__ import annotations

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier

# %% Part 1. 引数の検証とフォルダ設定 (XXXX < YYYY を保証)
if len(sys.argv) < 3:
    print("❌ エラー: 比較する2つの管理番号(4桁)を指定してください。\n使用例: python ~.py 0007 0011")
    sys.exit(1)

arg1, arg2 = sys.argv[1], sys.argv[2]
if not (arg1.isdigit() and len(arg1) == 4 and arg2.isdigit() and len(arg2) == 4):
    print("❌ エラー: 引数は4桁の数字で指定してください。")
    sys.exit(1)

seq_list = sorted([arg1, arg2])
seq_xxxx, seq_yyyy = seq_list[0], seq_list[1]

BASE_EXP_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "Experiment"))
EXP_XXXX_DIR = os.path.join(BASE_EXP_DIR, f"Experiment_{seq_xxxx}")
EXP_YYYY_DIR = os.path.join(BASE_EXP_DIR, f"Experiment_{seq_yyyy}")

# 出力先フォルダの作成
COMPARE_BASE_DIR = os.path.join(BASE_EXP_DIR, "Compare_Backtest")
COMPARE_OUT_DIR = os.path.join(COMPARE_BASE_DIR, f"Compare_Backtest_{seq_xxxx}_{seq_yyyy}")
os.makedirs(COMPARE_OUT_DIR, exist_ok=True)

print(f"📂 既存CSVを用いた比較バックテストを開始します: Experiment_{seq_xxxx} vs Experiment_{seq_yyyy}")

# %% Part 2. Load data (MVO計算用の株価データをXXXX側から取得)
train = pd.read_csv(os.path.join(EXP_XXXX_DIR, f"train_data_{seq_xxxx}.csv"))
trade = pd.read_csv(os.path.join(EXP_XXXX_DIR, f"trade_data_{seq_xxxx}.csv"))
train = train.set_index(train.columns[0])
trade = trade.set_index(trade.columns[0])

# %% Part 3. 既存のバックテスト結果CSVファイルの直接読み込み
csv_xxxx_path = os.path.join(EXP_XXXX_DIR, f"Backtest_{seq_xxxx}", f"Backtest_Price_{seq_xxxx}.csv")
csv_yyyy_path = os.path.join(EXP_YYYY_DIR, f"Backtest_{seq_yyyy}", f"Backtest_Price_{seq_yyyy}.csv")

if not (os.path.exists(csv_xxxx_path) and os.path.exists(csv_yyyy_path)):
    print(f"❌ エラー: 対象のバックテストCSVが見つかりません。\n・{csv_xxxx_path}\n・{csv_yyyy_path}\n先に単体バックテストを実行してください。")
    sys.exit(1)

# インデックスを日時に変換して読み込み
df_price_xxxx = pd.read_csv(csv_xxxx_path, index_col=0, parse_dates=True)
df_price_yyyy = pd.read_csv(csv_yyyy_path, index_col=0, parse_dates=True)

# %% Part 4. Mean Variance Optimization (MVO) ベースラインの計算
print("📊 MVOベースラインを算出中...")
StockData = train.pivot(index="date", columns="tic", values="close")
TradeData = trade.pivot(index="date", columns="tic", values="close")
arReturns = (np.diff(np.asarray(StockData), axis=0) / np.asarray(StockData)[:-1]) * 100

try:
    ef_mean = EfficientFrontier(np.mean(arReturns, axis=0), np.cov(arReturns, rowvar=False), weight_bounds=(0, 0.5))
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean = ef_mean.clean_weights()
    mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))])
except Exception:
    mvo_weights = np.array([1000000 * (1.0 / StockData.shape[1]) for _ in range(StockData.shape[1])])

Initial_Portfolio = np.multiply(mvo_weights, np.array([1 / p for p in StockData.tail(1).to_numpy()[0]]))
MVO_result = pd.DataFrame(TradeData @ Initial_Portfolio, columns=["Mean Var"])
MVO_result.index = df_price_xxxx.index  # 日付インデックスを同期

# 基準化（3_5_Backtest.pyのロジックに合わせる）
df_prices_raw = trade.pivot(index="date", columns="tic", values="close")
initial_stock_avg = df_prices_raw.iloc[0].mean()
mvo_norm_factor = initial_stock_avg / MVO_result["Mean Var"].iloc[0]
mvo_standardized = MVO_result["Mean Var"].values * mvo_norm_factor

# %% Part 5. エージェントごとに個別・指標別に完全独立したプロットを出力
sns.set_theme(style="whitegrid")

# 評価対象の5つのエージェント名
agents = ["a2c", "ddpg", "ppo", "td3", "sac"]

print("\n🎨 エージェントごと・指標ごとに独立したPDFプロットを出力中...")

# プロット保存用のヘルパー関数
def save_plot(filename, draw_func):
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    draw_func(ax)
    plt.tight_layout()
    pdf_path = os.path.join(COMPARE_OUT_DIR, filename)
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   -> 保存完了: {pdf_path}")

for agent in agents:
    col_name = f"{agent}_P"
    
    # CSV内に該当エージェントのカラム（例: a2c_P）が存在するかチェック
    if col_name not in df_price_xxxx.columns and col_name not in df_price_yyyy.columns:
        continue
        
    agent_upper = agent.upper()
    
    # -----------------------------------------------------------------
    # 1. Price Plot (資産推移)
    # -----------------------------------------------------------------
    def draw_price(ax):
        if col_name in df_price_xxxx.columns:
            ax.plot(df_price_xxxx.index, df_price_xxxx[col_name], label=f"Experiment_{seq_xxxx}", lw=2, color="crimson")
        if col_name in df_price_yyyy.columns:
            ax.plot(df_price_yyyy.index, df_price_yyyy[col_name], label=f"Experiment_{seq_yyyy}", lw=2, color="royalblue")
        ax.plot(df_price_xxxx.index, mvo_standardized, label="MVO", lw=1.5, linestyle="--", color="black")
        ax.set_title(f"{agent_upper} - Portfolio Value Over Time", weight="bold", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Normalized Price Scale", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)

    save_plot(f"Compare_{agent_upper}_Price.pdf", draw_price)
    
    # -----------------------------------------------------------------
    # 2. Log Return Plot (対数収益率オーバーレイ)
    # -----------------------------------------------------------------
    def draw_log(ax):
        if col_name in df_price_xxxx.columns:
            log_x = np.log(df_price_xxxx[col_name] / df_price_xxxx[col_name].shift(1)).dropna()
            ax.plot(log_x.index, log_x, label=f"Experiment_{seq_xxxx}", alpha=0.5, color="crimson")
        if col_name in df_price_yyyy.columns:
            log_y = np.log(df_price_yyyy[col_name] / df_price_yyyy[col_name].shift(1)).dropna()
            ax.plot(log_y.index, log_y, label=f"Experiment_{seq_yyyy}", alpha=0.5, color="royalblue")
        log_m = np.log(mvo_standardized / pd.Series(mvo_standardized).shift(1)).dropna()
        log_m.index = df_price_xxxx.index[1:]
        ax.plot(log_m.index, log_m, label="MVO", alpha=0.3, linestyle=":", color="black")
        ax.set_title(f"{agent_upper} - Log Return Overlay Comparison", weight="bold", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Log Return", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)

    save_plot(f"Compare_{agent_upper}_LogReturn.pdf", draw_log)

    # -----------------------------------------------------------------
    # 3. Sharpe Ratio Plot (ローリング・シャープレシオ)
    # -----------------------------------------------------------------
    def draw_sharpe(ax):
        if col_name in df_price_xxxx.columns:
            ret_x = df_price_xxxx[col_name].pct_change().dropna()
            rolling_x = (ret_x.rolling(window=20).mean() / ret_x.rolling(window=20).std()) * np.sqrt(250)
            ax.plot(rolling_x.index, rolling_x, label=f"Experiment_{seq_xxxx}", color="crimson", alpha=0.7)
        if col_name in df_price_yyyy.columns:
            ret_y = df_price_yyyy[col_name].pct_change().dropna()
            rolling_y = (ret_y.rolling(window=20).mean() / ret_y.rolling(window=20).std()) * np.sqrt(250)
            ax.plot(rolling_y.index, rolling_y, label=f"Experiment_{seq_yyyy}", color="royalblue", alpha=0.7)
            
        ret_mvo = pd.Series(mvo_standardized).pct_change().dropna()
        rolling_mvo = (ret_mvo.rolling(window=20).mean() / ret_mvo.rolling(window=20).std()) * np.sqrt(250)
        rolling_mvo.index = df_price_xxxx.index[1:]
        ax.plot(rolling_mvo.index, rolling_mvo, label="MVO", color="black", alpha=0.5, linestyle="--")
        
        ax.set_title(f"{agent_upper} - Rolling 20-Day Sharpe Ratio (Annualized)", weight="bold", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sharpe Ratio", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)

    save_plot(f"Compare_{agent_upper}_SharpeRatio.pdf", draw_sharpe)

    # -----------------------------------------------------------------
    # 4. Change Weights Plot (リバランスアクティビティ)
    # -----------------------------------------------------------------
    # 💡 内包されたウェイトカラム（例: Cash_W_ppo などの集合）から変化量をその場で復元計算します
    def compute_weight_activity(df, ag_name):
        w_cols = [c for c in df.columns if c.endswith(f"_W_{ag_name}")]
        if not w_cols:
            return pd.Series(dtype=float)
        return df[w_cols].diff().abs().sum(axis=1).dropna()

    def draw_weight_diff(ax):
        w_diff_x = compute_weight_activity(df_price_xxxx, agent)
        w_diff_y = compute_weight_activity(df_price_yyyy, agent)
        
        if not w_diff_x.empty:
            ax.plot(w_diff_x.index, w_diff_x, label=f"Experiment_{seq_xxxx}", color="crimson", alpha=0.6)
        if not w_diff_y.empty:
            ax.plot(w_diff_y.index, w_diff_y, label=f"Experiment_{seq_yyyy}", color="royalblue", alpha=0.6)
            
        mvo_diff = pd.Series(0.0, index=df_price_xxxx.index[1:])
        ax.plot(mvo_diff.index, mvo_diff, label="MVO", color="black", alpha=0.5, linestyle="--")
        ax.set_title(f"{agent_upper} - Rebalancing Activity Comparison", weight="bold", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Absolute Sum of Weight Changes", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)

    save_plot(f"Compare_{agent_upper}_ChangeWeightsPortfolio.pdf", draw_weight_diff)

print(f"\n✨ すべてのプロットが個別のPDFファイルとして正常に出力されました！")
print(f"   出力先フォルダ: {COMPARE_OUT_DIR}")