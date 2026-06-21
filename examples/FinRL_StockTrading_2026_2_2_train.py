"""
Stock NeurIPS2018 Part 2. Train (Optimized Version)

ターミナル引数「XXXX (4桁の連番)」を受け取り、対応する「Experiment/Experiment_XXXX」フォルダから
列車データ（train_data_XXXX.csv）をロードして学習を行います。
また、学習ログ（progress.csv）や学習済みモデル、さらには実験条件メタデータを同フォルダ内の
「README_XXXX.md」へ追記（Step 2）する機能を持っています。
"""

from __future__ import annotations

from datetime import datetime
import os
import shutil
import sys

import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# =====================================================================
# 1. 引数の検証と対象Experimentフォルダ・ファイルの決定
# =====================================================================
if len(sys.argv) < 2:
    print(
        "❌ エラー: 管理番号(4桁)を指定する引数が不足しています。\n使用例: python FinRL_StockTrading_2026_2_2_train.py 0001"
    )
    sys.exit(1)

seq_str = sys.argv[1]
if not (seq_str.isdigit() and len(seq_str) == 4):
    print("❌ エラー: 引数は4桁の数字で指定してください。(例: 0001)")
    sys.exit(1)

# 実験用ベースディレクトリと各種パスの特定
BASE_EXP_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "Experiment"))
SUB_EXP_DIR = os.path.join(BASE_EXP_DIR, f"Experiment_{seq_str}")

train_csv_path = os.path.join(SUB_EXP_DIR, f"train_data_{seq_str}.csv")
readme_md_path = os.path.join(SUB_EXP_DIR, f"README_{seq_str}.md")

# ファイルおよびフォルダの存在チェック
if not os.path.exists(SUB_EXP_DIR):
    print(f"❌ エラー: 指定されたフォルダ '{SUB_EXP_DIR}' が存在しません。先にdata.pyを実行してください。")
    sys.exit(1)

if not os.path.exists(train_csv_path):
    print(f"❌ エラー: 指定されたデータファイル '{train_csv_path}' が見つかりません。")
    sys.exit(1)

# 出力先をExperiment_XXXX内にリダイレクトするためのパス定義
CUSTOM_RESULTS_DIR = os.path.join(SUB_EXP_DIR, "results")
CUSTOM_TRAINED_MODEL_DIR = os.path.join(SUB_EXP_DIR, "trained_models")

os.makedirs(CUSTOM_RESULTS_DIR, exist_ok=True)
os.makedirs(CUSTOM_TRAINED_MODEL_DIR, exist_ok=True)


# =====================================================================
# 2. 環境の構築 (Environment Setup)
# =====================================================================
print(f"📂 データファイルをロード中: {train_csv_path}")
train = pd.read_csv(train_csv_path)
train = train.set_index(train.columns[0])
train.index.names = [""]

stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"📊 Stock Dimension: {stock_dimension}, State Space: {state_space}")

# 定数パラメータの定義
INITIAL_AMOUNT = 1000000
HMAX_SHARES = 10
TRANSACTION_COST_PCT = 0.001
REWARD_SCALING = 1e-4

buy_cost_list = sell_cost_list = [TRANSACTION_COST_PCT] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": HMAX_SHARES,
    "initial_amount": INITIAL_AMOUNT,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": REWARD_SCALING,
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()


# =====================================================================
# 3. エージェントのハイパーパラメータ定義
# =====================================================================
# ハイパーパラメータをREADMEに記録するために明示的に辞書で一元管理します
A2C_PARAMS = {"n_steps": 128, "ent_coef": 0.08, "learning_rate": 0.00005}
DDPG_PARAMS = {"batch_size": 256, "buffer_size": 1000000, "learning_rate": 0.0005}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.05,
    "learning_rate": 0.00005,
    "batch_size": 128,
}
TD3_PARAMS = {"batch_size": 256, "buffer_size": 1000000, "learning_rate": 0.00005, "policy_delay": 3}
SAC_PARAMS = {
    "batch_size": 256,
    "buffer_size": 100000,
    "learning_rate": 0.00003,
    "learning_starts": 1000,
    "ent_coef": "auto_0.3",
}

TOTAL_TIMESTEPS = 200000
if_using_a2c = if_using_ddpg = if_using_ppo = if_using_td3 = if_using_sac = True


# =====================================================================
# 4. モデルの学習と進行状況ログのリネーム処理 (Helper 関数)
# =====================================================================
def train_and_save_agent(agent_name, model_kwargs=None):
    """ロガーの設定、学習、進捗CSVのリネーム、モデルの保存を一括で行う関数"""
    print(f"\n🤖 🚀 訓練開始: {agent_name.upper()}...")
    agent = DRLAgent(env=env_train)
    model = agent.get_model(agent_name, model_kwargs=model_kwargs)

    # ロガーを一時的な作業用フォルダに設定
    temp_log_dir = os.path.join(CUSTOM_RESULTS_DIR, agent_name)
    new_logger = configure(temp_log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # 訓練の実行
    trained_model = agent.train_model(
        model=model, tb_log_name=agent_name, total_timesteps=TOTAL_TIMESTEPS
    )

    # 訓練完了後、モデルを指定の Experiment_XXXX フォルダへ保存
    model_save_path = os.path.join(CUSTOM_TRAINED_MODEL_DIR, f"agent_{agent_name}")
    trained_model.save(model_save_path)
    print(f"💾 {agent_name.upper()} モデルを保存しました: {model_save_path}")

    # progress.csv のリネームおよび配下への直接引き揚げ移動処理
    src_progress = os.path.join(temp_log_dir, "progress.csv")
    dst_progress = os.path.join(CUSTOM_RESULTS_DIR, f"progress_{agent_name}.csv")
    if os.path.exists(src_progress):
        shutil.move(src_progress, dst_progress)
        print(f"📝 ログファイル名を変更して移動しました: {dst_progress}")

    # 不要となった子階層の空フォルダを削除
    if os.path.exists(temp_log_dir):
        shutil.rmtree(temp_log_dir)


# 各エージェントの訓練を順次実行
if if_using_a2c:
    train_and_save_agent("a2c", A2C_PARAMS)
if if_using_ddpg:
    train_and_save_agent("ddpg", DDPG_PARAMS)
if if_using_ppo:
    train_and_save_agent("ppo", PPO_PARAMS)
if if_using_td3:
    train_and_save_agent("td3", TD3_PARAMS)
if if_using_sac:
    train_and_save_agent("sac", SAC_PARAMS)


# =====================================================================
# 5. 実験管理用 README.md ファイルへの情報追記 (Step2)
# =====================================================================
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

train_readme_append = f"""
## 🏋️ Step 2: モデル訓練フェーズ

モデルの学習環境パラメータおよびハイパーパラメータの設定ログです。

- **実行日時 (秒まで)**: {current_time_str}
- **初期投資の元本 (initial_amount)**: `${INITIAL_AMOUNT:,}`
- **最大取引株数 (max_stock_share / hmax)**: `{HMAX_SHARES}`
- **取引手数料比率 (transaction_cost_pct)**: `{TRANSACTION_COST_PCT}`
- **報酬スケーリング係数 (reward_scaling)**: `{REWARD_SCALING}`
- **テクニカル指標のリスト (INDICATORS)**: {INDICATORS}
- **各エージェントの総タイムステップ数**: `{TOTAL_TIMESTEPS}`

### ⚙️ アルゴリズムのハイパーパラメータ設定
* **A2C**: `{A2C_PARAMS}`
* **DDPG**: `{DDPG_PARAMS}`
* **PPO**: `{PPO_PARAMS}`
* **TD3**: `{TD3_PARAMS}`
* **SAC**: `{SAC_PARAMS}`

---
"""

if os.path.exists(readme_md_path):
    with open(readme_md_path, "a", encoding="utf-8") as f:
        f.write(train_readme_append)
    print(f"\n📝 訓練フェーズの実験メタデータを README に追記しました: {readme_md_path}")
else:
    with open(readme_md_path, "w", encoding="utf-8") as f:
        f.write(train_readme_append)
    print(f"\n⚠️ READMEが見つからなかったため、新規に作成して保存しました: {readme_md_path}")

print(f"✨ フォルダ 'Experiment_{seq_str}' へのすべての訓練・ログ集約処理が完了しました。")