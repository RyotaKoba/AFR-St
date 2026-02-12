# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
from datasets import load_from_disk
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper class for tokenized input IDs
class TokenizerWrapper:
    """
    Wrapper class for tokenized input IDs.

    Args:
        input_ids (tensor): The tokenized input IDs from the tokenizer.
    """
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset from local
def get_wikitext2_local(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # Load train and test datasets
    all_data = load_from_disk('./data_local/wiki_all')
    traindata = all_data['train']
    testdata = all_data['test']

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    print(type(nsamples))
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        print("i,j:",i,j)
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

# Load and process wikitext2 dataset from local
def get_wikitext2_local2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # WikiText2 (raw) データセット
    all_data = load_from_disk('./data_local/wiki_all')
    traindata = all_data['train']

    # トークナイズ関数
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=seqlen,  # 一回forwardするだけなら短めで十分
            padding="max_length"
        )

    tokenized = traindata.map(tokenize, batched=True, remove_columns=["text"])

    # DataLoaderを定義
    # DataLoader
    def collate_fn1(batch):
        input_ids = torch.stack([torch.tensor(x["input_ids"], dtype=torch.long) for x in batch])
        labels = input_ids.clone()
        labels[:, :-1] = -100  # 最後のトークン以外
        return {"input_ids": input_ids, "labels": labels}
    
    from torch.utils.data import DataLoader
    trainloader = DataLoader(
        tokenized,
        batch_size=1,  # forwardにかけるだけなら1でも良い
        shuffle=False,
        collate_fn=collate_fn1,
    )

    return trainloader


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    # trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    trainenc_list = []
    chunk_size = seqlen * 100
    
    for i in range(0, len(traindata['text']), chunk_size):
        chunk = " ".join(traindata['text'][i:i + chunk_size])
        enc = tokenizer(chunk, return_tensors='pt')
        trainenc_list.append(enc.input_ids)
    
    trainenc = torch.cat(trainenc_list, dim=1)
    print("trainenc_complete")
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    print("testenc_complete")

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    print(type(nsamples))
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

class LLaVAPretrainDataset(Dataset):
    def __init__(self, 
                 json_path='./data_local/llava/blip_laion_cc_sbu_558k.json',
                 image_folder='./data_local/llava/images'):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])
        image = Image.open(image_path).convert('RGB')

        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        full_text = question + answer
        
        return {
            'image': image,
            'text': full_text,
            'target': answer  # これだけあればいい
        }

def get_mmlu(nsamples, seed, seqlen, tokenizer, subjects=None):
    """
    Load and process the MMLU dataset for calibration.

    Args:
        nsamples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        seqlen (int): Sequence length for generated samples
        tokenizer (Tokenizer): Tokenizer instance
        subjects (list): List of MMLU subjects to include (None = all)

    Returns:
        list: List of (input, target) pairs for calibration
    """
    print("Loading MMLU dataset...")
    
    # MMLUデータセットをロード
    if subjects is None:
        # 全科目を使用
        dataset = load_dataset("cais/mmlu", "all", split="test")
    else:
        # 指定された科目のみ
        datasets = []
        for subject in subjects:
            ds = load_dataset("cais/mmlu", subject, split="test")
            datasets.append(ds)
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(datasets)
    
    random.seed(seed)
    
    # サンプリング
    indices = random.sample(range(len(dataset)), min(nsamples, len(dataset)))
    
    trainloader = []
    for idx in indices:
        item = dataset[idx]
        
        # MMLU形式: question + choices -> answer
        question = item['question']
        choices = item['choices']
        answer = item['answer']  # 0-3のインデックス
        
        # プロンプト構築
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        
        # トークナイズ
        inp = tokenizer(prompt, return_tensors='pt', 
                       truncation=True, max_length=seqlen).input_ids
        
        # ターゲット生成(最後のトークン以外をマスク)
        tar = inp.clone()
        tar[:, :-1] = -100
        
        trainloader.append((inp, tar))
    print("MMLU dataset loaded and processed.")
    
    return trainloader

def get_hellaswag(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the HellaSwag dataset for calibration.

    Args:
        nsamples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        seqlen (int): Sequence length for generated samples
        tokenizer (Tokenizer): Tokenizer instance

    Returns:
        list: List of (input, target) pairs for calibration
    """
    from datasets import load_dataset
    print("Loading HellaSwag dataset...")
    
    # HellaSwagデータセットをロード
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    
    random.seed(seed)
    
    # サンプリング
    indices = random.sample(range(len(dataset)), min(nsamples, len(dataset)))
    
    trainloader = []
    for idx in indices:
        item = dataset[idx]
        
        # HellaSwag形式: context + activity_label -> correct ending
        context = item['ctx']
        activity = item['activity_label']
        endings = item['endings']
        correct_idx = int(item['label'])
        
        # プロンプト構築
        prompt = f"Activity: {activity}\n"
        prompt += f"Context: {context}\n"
        prompt += "What happens next?\n"
        for i, ending in enumerate(endings):
            prompt += f"{i+1}. {ending}\n"
        prompt += "Answer:"
        
        # トークナイズ
        inp = tokenizer(prompt, return_tensors='pt', 
                       truncation=True, max_length=seqlen).input_ids
        
        # ターゲット生成(最後のトークン以外をマスク)
        tar = inp.clone()
        tar[:, :-1] = -100
        
        trainloader.append((inp, tar))
    print("HellaSwag dataset loaded and processed.")
    
    return trainloader

def get_winogrande(nsamples, seed, seqlen, tokenizer, subset='winogrande_xl'):
    """
    Load and process the Winogrande dataset for calibration.

    Args:
        nsamples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        seqlen (int): Sequence length for generated samples
        tokenizer (Tokenizer): Tokenizer instance
        subset (str): Winogrande subset ('winogrande_xs', 'winogrande_s', 
                      'winogrande_m', 'winogrande_l', 'winogrande_xl')

    Returns:
        list: List of (input, target) pairs for calibration
    """
    from datasets import load_dataset
    print("Loading Winogrande dataset...")
    
    # Winograndeデータセットをロード
    dataset = load_dataset("winogrande", subset, split="validation")
    
    random.seed(seed)
    
    # サンプリング
    indices = random.sample(range(len(dataset)), min(nsamples, len(dataset)))
    
    trainloader = []
    for idx in indices:
        item = dataset[idx]
        
        # Winogrande形式: sentence with _ + option1/option2 -> answer
        sentence = item['sentence']
        option1 = item['option1']
        option2 = item['option2']
        answer = item['answer']  # "1" or "2"
        
        # プロンプト構築
        prompt = f"Fill in the blank:\n"
        prompt += f"{sentence}\n"
        prompt += f"1. {option1}\n"
        prompt += f"2. {option2}\n"
        prompt += "Answer:"
        
        # トークナイズ
        inp = tokenizer(prompt, return_tensors='pt', 
                       truncation=True, max_length=seqlen).input_ids
        
        # ターゲット生成(最後のトークン以外をマスク)
        tar = inp.clone()
        tar[:, :-1] = -100
        
        trainloader.append((inp, tar))
    print("Winogrande dataset loaded and processed.")
    
    return trainloader

def get_arc(nsamples, seed, seqlen, tokenizer, subset='ARC-Challenge'):
    """
    Load and process the ARC dataset for calibration.

    Args:
        nsamples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        seqlen (int): Sequence length for generated samples
        tokenizer (Tokenizer): Tokenizer instance
        subset (str): ARC subset ('ARC-Easy' or 'ARC-Challenge')

    Returns:
        list: List of (input, target) pairs for calibration
    """
    from datasets import load_dataset
    
    # ARCデータセットをロード
    dataset = load_dataset("ai2_arc", subset, split="test")
    
    random.seed(seed)
    
    # サンプリング
    indices = random.sample(range(len(dataset)), min(nsamples, len(dataset)))
    
    trainloader = []
    for idx in indices:
        item = dataset[idx]

        # ARC形式: question + choices -> answerKey
        question = item['question']
        choices = item['choices']
        answer_key = item['answerKey']

        prompt = f"Question: {question}\n"
        choice_texts = choices['text']
        choice_labels = choices['label']

        for label, text in zip(choice_labels, choice_texts):
            prompt += f"{label}. {text}\n"
        prompt += "Answer:"
        inp = tokenizer(prompt, return_tensors='pt',
                       truncation=True, max_length=seqlen).input_ids
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

# Function to select the appropriate loader based on dataset name
def get_loaders(nsamples=128, seed=0, seqlen=2048, tokenizer=None, dataset='wikitext2_local'):
    if dataset == 'wikitext2':
        return get_wikitext2_local(nsamples, seed, seqlen, tokenizer)
    elif dataset == 'mmlu':
        return get_mmlu(nsamples, seed, seqlen, tokenizer)
    elif dataset == 'hellaswag':
        return get_hellaswag(nsamples, seed, seqlen, tokenizer)
    elif dataset == 'winogrande':
        return get_winogrande(nsamples, seed, seqlen, tokenizer)
    elif dataset == 'arc_challenge':
        return get_arc(nsamples, seed, seqlen, tokenizer, subset='ARC-Challenge')
    elif dataset == 'arc_easy':
        return get_arc(nsamples, seed, seqlen, tokenizer, subset='ARC-Easy')
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: wikitext2_local, wikitext2, mmlu, hellaswag, winogrande, arc_challenge, arc_easy")



def get_mm_loaders():
    dataset = LLaVAPretrainDataset()
    subset_indices = range(min(1000, len(dataset)))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    def custom_collate_fn(batch):
        return {
            'image': [item['image'] for item in batch],
            'text': [item['text'] for item in batch],
            'target': [item['target'] for item in batch]
        }

    dataloader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    return dataloader

if __name__ == "__main__":
    get_loaders('wikitext2', seed=0, seqlen=2048, tokenizer=None)

