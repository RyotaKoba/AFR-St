# AFR-St: 大規模言語モデルの構造的枝刈りフレームワーク

LLaMAやLLaVAなどのLLMに対して、キャリブレーションデータを用いたスコアベースの枝刈りを行うフレームワークです。

## 概要

本リポジトリでは、MLP層を対象とした構造的・非構造的枝刈りを複数の手法で実装しています。

**対応枝刈り手法:**

| 手法                     | 種類     | 説明                                                           |
| ------------------------ | -------- | -------------------------------------------------------------- |
| `structured_afr`       | 構造的   | 一次勾配スコア（活性化）とSNIPスコアをグローバル標準化して合算 |
| `structured_snip`      | 構造的   | SNIP（勾配ベースの重要度スコア）                               |
| `structured_refer_svd` | 構造的   | ReFer（SVDベースの重要度スコア）                               |
| `structured_afr_llava` | 構造的   | LLaVAマルチモーダルモデル向けAFR                               |
| `afr`                  | 非構造的 | AFR（要素ごとのマスク）                                        |
| `snip`                 | 非構造的 | SNIP（要素ごとのマスク）                                       |
| `refer_svd`            | 非構造的 | ReFer SVD（要素ごとのマスク）                                  |

## 動作確認済み環境

| パッケージ   | バージョン  |
| ------------ | ----------- |
| Python       | 3.x         |
| torch        | 2.9.1+cu130 |
| transformers | 4.57.1      |
| accelerate   | 1.7.0       |
| datasets     | 3.6.0       |
| tokenizers   | 0.22.2      |
| safetensors  | 0.6.2       |
| peft         | 0.15.2      |
| tqdm         | 4.67.1      |
| numpy        | 1.26.4      |
| pillow       | 11.3.0      |
| einops       | 0.8.1       |
| lm_eval      | 0.4.7       |
| lmms_eval    | 0.5.0       |

CUDA: 13.0（`nvidia-cuda-runtime 13.0.48`）

## インストール

評価まで行うツールは、リポジトリ内のフォルダをeditable installしてください：

```bash
pip install -e ./lm-evaluation-harness
pip install -e ./lmms-eval
```

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
│   ├── builder.py       # モデルローダー（LLaVA対応）
│   ├── model.py         # モデル操作ユーティリティ（層削除など）
│   ├── bmm.py           # ベイズ混合ガウスモデルによる外れ値除去
│   ├── dpm.py           # ディリクレ過程混合モデルによる外れ値除去
│   ├── gmm.py           # 混合ガウスモデルによるスコア処理
│   ├── kde.py           # カーネル密度推定によるスコア処理
│   └── gesd.py          # GESDによる外れ値除去
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

| 引数                 | デフォルト         | 説明                                       |
| -------------------- | ------------------ | ------------------------------------------ |
| `--model`          | （必須）           | HuggingFaceモデル名またはローカルパス      |
| `--prune_method`   | `structured_afr` | 枝刈り手法（上表参照）                     |
| `--pruning_ratio`  | `0.0`            | 枝刈り率（例:`0.2` = 20%）               |
| `--nsamples`       | `128`            | キャリブレーションサンプル数               |
| `--seed`           | `0`              | ランダムシード                             |
| `--dataset`        | `wikitext2`      | キャリブレーションデータセット             |
| `--cuda`           | `False`          | GPU使用フラグ                              |
| `--global_pruning` | `False`          | 全レイヤーをまたいだグローバル枝刈りフラグ |
| `--save_model`     | `None`           | 枝刈り後モデルの保存先パス                 |
| `--cache_dir`      | `llm_weights`    | HuggingFaceモデルキャッシュディレクトリ    |

**キャリブレーションデータセット:**

| データセット      | 説明                                      |
| ----------------- | ----------------------------------------- |
| `wikitext2`     | WikiText-2（HuggingFaceからダウンロード） |
| `mmlu`          | MMLUベンチマーク（全科目）                |
| `hellaswag`     | HellaSwag                                 |
| `winogrande`    | Winogrande XL                             |
| `arc_challenge` | ARC Challenge                             |
| `arc_easy`      | ARC Easy                                  |

### 実行例

`start.sh`、`test.sh`、`lmms-eval.sh` のモデル名や各パラメータを編集してから実行してください。

```bash
bash start.sh       # 枝刈り
bash test.sh        # 言語タスク評価
bash lmms-eval.sh   # マルチモーダルタスク評価（LLaVA）
```

## LLaVA用キャリブレーションデータ

LLaVAの枝刈りに使用するキャリブレーションデータはサイズが大きいためGitには含めていません。
以下のGoogle Driveからダウンロードし、下記の構成で配置してください。

[blip_laion_cc_sbu_558k](https://drive.google.com/file/d/1m3K2r2w5N6FgYhU6x8725uvPXEPnEdar/view?usp=sharing)

```
data_local/
└── llava/
    ├── blip_laion_cc_sbu_558k.json
    └── images/
```

## 保存形式について

- **`--global_pruning` なし**: HuggingFaceの `save_pretrained` 形式で保存。`lm-eval --model hf` でそのまま利用可能
- **`--global_pruning` あり**: `{'model': model, 'tokenizer': tokenizer}` を含む `model.bin` として保存
