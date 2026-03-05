# AFR-St: 大規模言語モデルの構造的枝刈りフレームワーク

LLaMAやLLaVAなどのLLMに対して、キャリブレーションデータを用いたスコアベースの枝刈りを行うフレームワークです。

## 概要

本リポジトリでは、MLP層を対象とした構造的・非構造的枝刈りを複数の手法で実装しています。

**対応枝刈り手法:**

| 手法                     | 種類     | 説明                                                           |
| ------------------------ | -------- | -------------------------------------------------------------- |
| `structured_afr`       | 構造   | 一次勾配スコア（活性化）とSNIPスコアをグローバル標準化して合算 |
| `structured_snip`      | 構造   | SNIP（勾配ベースの重要度スコア）                               |
| `structured_refer_l1` | 構造   | ReFer（L1ベースの重要度スコア）                               |
| `structured_refer_svd` | 構造   | ReFer（SVDベースの重要度スコア）                               |
| `structured_afr_llava` | 構造   | LLaVAマルチモーダルモデル向けAFR                               |
| `afr`                  | 非構造 | AFR（要素ごとのマスク）                                        |
| `snip`                 | 非構造 | SNIP（要素ごとのマスク）                                       |
| `refer_svd`            | 非構造 | ReFer SVD（要素ごとのマスク）                                  |
| `refer_l1`             | 非構造 | ReFer L1 (要素ごとのマスク)                                    |

## 動作確認済み環境

| パッケージ   | バージョン  |
| ------------ | ----------- |
| Python       | 3.12        |
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
DGX Spark上の設定方法です．他の環境でセットアップする際は適宜変更を加えて下さい．

### イメージの作成
リポジトリ内のDockerfileを使用
```bash
docker build --build-arg USERID=$UID --build-arg USERNAME=$USER -t hoge:hoge .
```
hogeは任意の名前・タグに書き換え

注意：このコマンド及びDockerfileは自身のユーザーがSudoグループに入っていることが前提です．

### コンテナの作成
```bash
docker run -it --gpus all -v hoge1:hoge2 -w hoge2 --name test hoge:hoge
```
hoge1：ローカルの任意のパス

hoge2：コンテナ内の任意のパス

hoge：前の手順で作成したイメージ

test：任意のコンテナ名

### コンテナ内でやる事
LLama等のモデルを利用するには認証と認証済みアカウントのトークンを用いたログインが必要です．
認証及びトークンの発行等は事前に行ってください．そのうえで以下のコマンドを入力し，トークンを使用してログイン．
```bash
huggingface-cli login
```
評価のツールは、一部カスタムしてあるのでリポジトリ内のフォルダをeditable installしてください：

```bash
sudo pip install --break-system-packages -e ./lm-evaluation-harness
sudo pip install --break-system-packages -e ./lmms-eval
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
│   ├── tools.py         # 外れ値処理・スコア集約
│   ├── data.py          # データセットローダー
│   ├── builder.py       # モデルローダー（LLaVA対応）
│   ├── model.py         # モデル操作ユーティリティ（層削除など）
│   ├── bmm.py           # ベイズ混合ガウスモデルによる外れ値除去
│   ├── dpm.py           # ディリクレ過程混合モデルによる外れ値除去
│   ├── gmm.py           # 混合ガウスモデルによるスコア処理
│   ├── kde.py           # カーネル密度推定によるスコア処理
│   └── gesd.py          # GESDによる外れ値除去
├── Recovery/            # 枝刈り済みモデルのLoRAチューニング用
└── data_local/          # ローカルデータセット

```
LoRAチューニング用のコードはDGX Sparkでは動作未検証

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
  --outlier_method "percentile" \
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
| `--dataset`        | `wikitext2`      | キャリブレーションデータセット（下表参照）     |
| `--outlier_method` | `None`           | 外れ値処理の方法（下表参照）                         |
| `--cuda`           | `False`          | GPU使用フラグ                              |
| `--global_pruning` | `False`          | 全レイヤーをまたいだグローバル枝刈りフラグ |
| `--save_model`     | `None`           | 枝刈り後モデルの保存先パス                 |
| `--cache_dir`      | `llm_weights`    | HuggingFaceモデルキャッシュディレクトリ    |

**キャリブレーションデータセット:**

| データセット      | 説明                       |
| ----------------- | -------------------------- |
| `wikitext2`     | WikiText-2                 |
| `mmlu`          | MMLUベンチマーク（全科目） |
| `hellaswag`     | HellaSwag                  |
| `winogrande`    | Winogrande XL              |
| `arc_challenge` | ARC Challenge              |
| `arc_easy`      | ARC Easy                   |

**外れ値処理の方法:**

| データセット      | 説明                       |
| ----------------- | -------------------------- |
| `percentile`     | trimme percentile                 |
| `gmm`            | GMM-based                         |
| `kde`            | Fast 1D KDE using FFT convolution |
| `bmm`            | Bayesian Mixture Model-based      |
| `gesd`           | GESD Test                         |
| `dpm`            | Dirichlet Process Mixture-based   |

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
