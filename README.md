# AFR-St: 大規模言語モデルの構造的枝刈りフレームワーク

LLaMAやLLaVAなどのLLMに対して、キャリブレーションデータを用いたスコアベースの枝刈りを行うフレームワークです。

## 概要

本リポジトリでは、MLP層を対象とした構造的・非構造的枝刈りを複数の手法で実装しています。

**対応枝刈り手法:**

| 手法 | 種類 | 説明 |
|---|---|---|
| `structured_afr` | 構造的 | 一次勾配スコア（活性化）とSNIPスコアをグローバル標準化して合算 |
| `structured_snip` | 構造的 | SNIP（勾配ベースの重要度スコア） |
| `structured_refer_svd` | 構造的 | ReFer（SVDベースの重要度スコア） |
| `structured_afr_llava` | 構造的 | LLaVAマルチモーダルモデル向けAFR |
| `afr` | 非構造的 | AFR（要素ごとのマスク） |
| `snip` | 非構造的 | SNIP（要素ごとのマスク） |
| `refer_svd` | 非構造的 | ReFer SVD（要素ごとのマスク） |

## インストール

```bash
pip install torch transformers accelerate datasets tqdm pillow
```

LLaVAを使用する場合は、追加で `lib/builder.py` が依存するLLaVA関連ライブラリをインストールしてください。

## プロジェクト構成

```
.
├── main.py              # エントリーポイント
├── start.sh             # 枝刈り実行スクリプトのサンプル
├── test.sh              # 評価実行スクリプトのサンプル
├── lmms-eval.sh         # LLaVAマルチモーダル評価スクリプト
├── lib/
│   ├── prune.py         # 全枝刈り手法の実装
│   ├── data.py          # データセットローダー
│   └── builder.py       # モデルローダー（LLaVA対応）
└── data_local/          # ローカルデータセットキャッシュ（任意）
```

## 使い方

### 枝刈りの実行

```bash
python main.py \
  --model <モデル名またはパス> \
  --prune_method <手法名> \
  --pruning_ratio <枝刈り率> \
  --nsamples <キャリブレーションサンプル数> \
  --dataset <キャリブレーションデータセット> \
  --cuda \
  --save_model <保存先パス>
```

**引数一覧:**

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--model` | （必須） | HuggingFaceモデル名またはローカルパス |
| `--prune_method` | `structured_afr` | 枝刈り手法（上表参照） |
| `--pruning_ratio` | `0.0` | 枝刈り率（例: `0.2` = 20%） |
| `--nsamples` | `128` | キャリブレーションサンプル数 |
| `--seed` | `0` | ランダムシード |
| `--dataset` | `wikitext2` | キャリブレーションデータセット |
| `--cuda` | `False` | GPU使用フラグ |
| `--global_pruning` | `False` | 全レイヤーをまたいだグローバル枝刈りフラグ |
| `--save_model` | `None` | 枝刈り後モデルの保存先パス |
| `--cache_dir` | `llm_weights` | HuggingFaceモデルキャッシュディレクトリ |

**キャリブレーションデータセット:**

| データセット | 説明 |
|---|---|
| `wikitext2` | WikiText-2（HuggingFaceからダウンロード） |
| `mmlu` | MMLUベンチマーク（全科目） |
| `hellaswag` | HellaSwag |
| `winogrande` | Winogrande XL |
| `arc_challenge` | ARC Challenge |
| `arc_easy` | ARC Easy |

### 実行例

```bash
# LLaMA-3-8Bに対してAFR構造的枝刈り（枝刈り率20%）
bash start.sh

# 同等のコマンドを直接実行:
python main.py \
  --model meta-llama/Meta-Llama-3-8B \
  --prune_method structured_afr \
  --pruning_ratio 0.2 \
  --nsamples 128 \
  --dataset wikitext2 \
  --cuda \
  --save_model ./pruned_model/hoge
```

### LLaVAの枝刈り

```bash
python main.py \
  --model liuhaotian/llava-v1.5-13b \
  --prune_method structured_afr_llava \
  --pruning_ratio 0.2 \
  --nsamples 128 \
  --cuda \
  --save_model ./pruned_model/llava_pruned
```

## 評価

### 言語タスク（lm-evaluation-harness）

**HuggingFace形式（`--global_pruning` なしで保存した場合）:**
```bash
lm-eval \
  --model hf \
  --batch_size 64 \
  --model_args device_map=auto,dtype=float16,pretrained=./pruned_model/hoge \
  --tasks winogrande,hellaswag,arc_easy,arc_challenge,mmlu
```

**model.bin形式（`--global_pruning` ありで保存した場合）:**
```bash
lm-eval \
  --model custom_checkpoint \
  --model_args pretrained=meta-llama/Meta-Llama-3-8B,device_map=auto,dtype=float16,checkpoint=./pruned_model/hoge/model.bin \
  --tasks arc_easy,winogrande,hellaswag,arc_challenge,mmlu \
  --batch_size 64
```

### マルチモーダルタスク（lmms-eval、LLaVA）

```bash
accelerate launch --num_processes=1 -m lmms_eval --model llava \
  --model_args pretrained="liuhaotian/llava-v1.5-13b,pruned=./pruned_model/llava_pruned/model.bin,device_map=auto" \
  --tasks vizwiz_vqa_val,gqa,scienceqa_img \
  --batch_size 1 \
  --output_path ./logs/pruned
```

## 保存形式について

- **`--global_pruning` なし**: HuggingFaceの `save_pretrained` 形式で保存。`lm-eval --model hf` でそのまま利用可能。モデルの `intermediate_size` が変更されます。
- **`--global_pruning` あり**: `{'model': model, 'tokenizer': tokenizer}` を含む `model.bin` として保存。カスタムチェックポイントローダーが必要です。

## 注意事項

- CPUモード（`--cuda` なし）は動作しますが非常に低速です。デバッグ用途を推奨します。
- 構造的枝刈りはMLPのニューロン丸ごとを削除し、`intermediate_size` を実際に縮小します。
- 非構造的枝刈りは要素ごとのマスクで重みをゼロにするのみで、モデルアーキテクチャは変更されません。
