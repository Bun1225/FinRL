"""
Stock NeurIPS2018 Part 1. Data (Dual-Argument Version)

2つの引数（第1引数: Train用銘柄リスト、第2引数: Trade用銘柄リスト）を動的に切り替え、
親ディレクトリの「Experiment/Experiment_XXXX」内に連番管理されたCSVデータと、
詳細な実験条件を記録する「README_XXXX.md」を自動的に生成するスクリプトです。
"""

from __future__ import annotations

from datetime import datetime
import glob
import itertools
import os
import re
import sys

import pandas as pd
import yfinance as yf

from finrl import config_tickers
from finrl.config import INDICATORS
from finrl.config import TRADE_END_DATE
from finrl.config import TRADE_START_DATE
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# =====================================================================
# 1. 引数処理 & config_tickers 存在チェック (2引数対応)
# =====================================================================
if len(sys.argv) < 3:
    print(
        "❌ エラー: 引数が不足しています。Train用とTrade用の2つの銘柄リストを指定してください。\n"
        "使用例: python FinRL_StockTrading_2026_1_2_data.py JP_HIGH_VOL JP_LOW_VOL"
    )
    sys.exit(1)

train_list_name = sys.argv[1]
trade_list_name = sys.argv[2]

# Train用銘柄リストの存在チェック
if not hasattr(config_tickers, train_list_name):
    print(f"❌ エラー: Train用に指定された '{train_list_name}' は config_tickers.py 内に存在しません。")
    sys.exit(1)

# Trade用銘柄リストの存在チェック
if not hasattr(config_tickers, trade_list_name):
    print(f"❌ エラー: Trade用に指定された '{trade_list_name}' は config_tickers.py 内に存在しません。")
    sys.exit(1)

# それぞれの銘柄リストを取得
train_tickers = getattr(config_tickers, train_list_name)
trade_tickers = getattr(config_tickers, trade_list_name)

# ダウンロード用に全銘柄の重複を排除したユニークなリストを作成
all_tickers = list(set(train_tickers + trade_tickers))

print(f"✅ Train用銘柄リスト '{train_list_name}' をロードしました（銘柄数: {len(train_tickers)}）")
print(f"✅ Trade用銘柄リスト '{trade_list_name}' をロードしました（銘柄数: {len(trade_tickers)}）")
print(f"🔄 合計ユニーク銘柄数: {len(all_tickers)}")


# =====================================================================
# 2. 出力フォルダ・ファイルの階層管理 (Experiment_XXXX フォルダ自動生成)
# =====================================================================
BASE_EXP_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "Experiment"))
os.makedirs(BASE_EXP_DIR, exist_ok=True)


def get_next_sequence_number(base_dir):
    """Experimentフォルダ内の既存フォルダを走査し、次の4桁の連番を決定します。"""
    folders = glob.glob(os.path.join(base_dir, "Experiment_*"))
    max_num = 0
    for f in folders:
        if os.path.isdir(f):
            match = re.search(r"Experiment_(\d{4})$", os.path.basename(f))
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
    return max_num + 1


seq_num = get_next_sequence_number(BASE_EXP_DIR)
# 専用のサブフォルダパス (Experiment_XXXX) を定義
SUB_EXP_DIR = os.path.join(BASE_EXP_DIR, f"Experiment_{seq_num:04d}")
os.makedirs(SUB_EXP_DIR, exist_ok=True)

# 各種ファイルパスの構築
train_csv_path = os.path.join(SUB_EXP_DIR, f"train_data_{seq_num:04d}.csv")
trade_csv_path = os.path.join(SUB_EXP_DIR, f"trade_data_{seq_num:04d}.csv")
readme_md_path = os.path.join(SUB_EXP_DIR, f"README_{seq_num:04d}.md")


# =====================================================================
# 3. データ取得 & 前処理 (統合された銘柄リストを使用)
# =====================================================================
print("\n=== Data Downloading ===")
df_raw = YahooDownloader(
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    ticker_list=all_tickers,
).fetch_data()

print("\n=== Feature Engineering ===")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False,
)

processed = fe.preprocess_data(df_raw)

list_ticker = processed["tic"].unique().tolist()
list_date = list(
    pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
)
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"])
processed_full = processed_full.fillna(0)


# =====================================================================
# 4. データ分割、銘柄フィルタリング & 連番ファイル名で保存
# =====================================================================
# 期間でベースデータを分割
raw_train_split = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
raw_trade_split = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)

# 各引数の銘柄リストに存在する銘柄のみを抽出（フィルタリング処理）
train = raw_train_split[raw_train_split["tic"].isin(train_tickers)].copy()
trade = raw_trade_split[raw_trade_split["tic"].isin(trade_tickers)].copy()

# 実際に処理され保存された最終的な銘柄コードのリストを取得
actual_train_tickers = train["tic"].unique().tolist()
actual_trade_tickers = trade["tic"].unique().tolist()

print(f"\nTrain data length: {len(train)} (銘柄数: {len(actual_train_tickers)})")
print(f"Trade data length: {len(trade)} (銘柄数: {len(actual_trade_tickers)})")

train.to_csv(train_csv_path)
trade.to_csv(trade_csv_path)
print(f"📂 Saved train data to: {train_csv_path}")
print(f"📂 Saved trade data to: {trade_csv_path}")


# =====================================================================
# 5. 実験管理用 README.md ファイルの生成 (Step1)
# =====================================================================
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

readme_content = f"""# Experiment_{seq_num:04d} - 実験管理レポート

## 📝 Step 1: データ収集・前処理フェーズ

後から実験条件を振り返るためのログシートです。

- **実行日時 (秒まで)**: {current_time_str}
- **Train用 銘柄リスト名 (第1引数)**: `{train_list_name}`
- **Trade用 銘柄リスト名 (第2引数)**: `{trade_list_name}`
- **Train対象 銘柄一覧**: {actual_train_tickers}
- **Trade対象 銘柄一覧**: {actual_trade_tickers}
- **Train検証 期間**: `{TRAIN_START_DATE}` 〜 `{TRAIN_END_DATE}`
- **Trade検証 期間**: `{TRADE_START_DATE}` 〜 `{TRADE_END_DATE}`

---
"""

with open(readme_md_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"📝 Experimental metadata successfully saved to: {readme_md_path}")
print(f"✨ 全ての処理が正常に終了し、フォルダ 'Experiment_{seq_num:04d}' へ格納されました。")