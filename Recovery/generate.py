#!/usr/bin/env python3
"""
FineWeb-EduをLLaMA-Factory形式に変換するスクリプト
CFSPの回復微調整用データ準備
"""

import json
import argparse
from datasets import load_dataset
from tqdm import tqdm
import os

def convert_fineweb_to_llamafactory(
    output_file: str,
    max_samples: int = 4_800_000,  # 960万の半分
    max_tokens: int = None,  # Noneの場合は全て処理
    max_length: int = None,  # Noneの場合は制限なし
    streaming: bool = True
):
    """
    FineWeb-EduをLLaMA-Factory形式に変換（サンプル数制限）
    
    Args:
        output_file: 出力ファイルパス
        max_samples: 最大サンプル数
        max_tokens: 最大トークン数（Noneの場合は全て）
        max_length: 1サンプルあたりの最大トークン長（Noneの場合は制限なし）
        streaming: ストリーミング読み込み
    """
    
    print("Loading FineWeb-Edu dataset...")
    
    # FineWeb-Eduデータセットを読み込み
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT",  # 10Bトークンのサンプル版
        streaming=streaming,
        split="train"
    )
    
    converted_data = []
    total_tokens = 0
    processed_count = 0
    
    print(f"Converting to LLaMA-Factory format (max samples: {max_samples:,})...")
    
    for sample in tqdm(dataset, desc="Processing samples"):
        # token_countを利用
        sample_tokens = sample.get("token_count", 0)
        
        # 空のテキストやトークン数が0のものはスキップ
        text = sample["text"].strip()
        if not text or sample_tokens == 0:
            continue
        
        # 長さ制限がある場合のみ切り詰め
        if max_length and sample_tokens > max_length:
            # 比例的にテキストを切り詰め
            ratio = max_length / sample_tokens
            text = text[:int(len(text) * ratio)]
            sample_tokens = max_length
        
        # LLaMA-Factory形式に変換（pretraining用）
        converted_sample = {
            "text": text
        }
        
        converted_data.append(converted_sample)
        total_tokens += sample_tokens
        processed_count += 1
        
        # 定期的な進捗表示
        if processed_count % 10000 == 0:
            print(f"Processed {processed_count} samples, {total_tokens:,} tokens")
        
        # サンプル数制限チェック
        if processed_count >= max_samples:
            print(f"Reached max samples: {processed_count:,}")
            break
        
        # トークン制限がある場合のみチェック
        if max_tokens and total_tokens >= max_tokens:
            print(f"Reached target token count: {total_tokens:,}")
            break
    
    # JSONファイルに保存
    print(f"Saving {len(converted_data)} samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion completed! Saved {len(converted_data)} samples.")


def create_dataset_config(dataset_name: str, file_name: str):
    """
    データセット設定ファイルを作成（pretraining用）
    """
    config = {
        dataset_name: {
            "file_name": file_name,
            "formatting": "plain",
            "columns": {
                "text": "text"
            }
        }
    }
    
    config_file = f"{dataset_name}_dataset_info.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset config saved to: {config_file}")
    return config_file


def main():
    parser = argparse.ArgumentParser(description="Convert FineWeb-Edu to LLaMA-Factory format")
    parser.add_argument("--output", default="fineweb_edu_llamafactory.json", 
                       help="Output file path")
    parser.add_argument("--max_samples", type=int, default=4_800_000,
                       help="Maximum number of samples (default: 4.8M)")
    parser.add_argument("--max_tokens", type=int, default=None,
                       help="Maximum number of tokens (None = process all)")
    parser.add_argument("--max_length", type=int, default=None,
                       help="Maximum token length per sample (None = no limit)")
    parser.add_argument("--dataset_name", default="fineweb_edu",
                       help="Dataset name for config")
    
    args = parser.parse_args()
    
    # データ変換実行
    convert_fineweb_to_llamafactory(
        output_file=args.output,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        max_length=args.max_length
    )
    
    # データセット設定ファイル作成
    create_dataset_config(args.dataset_name, args.output)
    
    print(f"\nNext steps:")
    print(f"1. Move {args.output} to your LLaMA-Factory data directory")
    print(f"2. Add the dataset config to your dataset_info.json")
    print(f"3. Use '--dataset {args.dataset_name}' in your training command")
    print(f"4. Use max_steps in YAML to control token usage during training")



if __name__ == "__main__":
    main()