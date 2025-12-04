import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaForMaskedLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [x.strip() for x in f if x.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        enc = self.tokenizer(
            line,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
        }


def fast_perplexity(model, loader, device):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in tqdm(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            out = model(
                input_ids=ids,
                attention_mask=mask,
                labels=ids
            )
            losses = out.loss  # avg loss per batch

            # convert scalar loss per batch → per sample score
            loss_val = losses.item()
            scores.append(np.exp(loss_val))

    return np.array(scores)


def evaluate(benign, malicious):
    y_true = np.concatenate([np.zeros(len(benign)), np.ones(len(malicious))])
    y_score = np.concatenate([benign, malicious])

    auc = roc_auc_score(y_true, y_score)
    print("ROC-AUC:", auc)

    threshold = np.percentile(benign, 99)
    preds = (y_score > threshold).astype(int)

    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    print("Threshold (99th% benign):", threshold)
    print("Precision:", precision)
    print("Recall:", recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--benign_file", required=True)
    parser.add_argument("--malicious_file", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    model = RobertaForMaskedLM.from_pretrained(args.model_path).to(device)

    benign_set = TextDataset(args.benign_file, tokenizer)
    mal_set = TextDataset(args.malicious_file, tokenizer)

    benign_loader = DataLoader(benign_set, batch_size=args.batch_size)
    mal_loader = DataLoader(mal_set, batch_size=args.batch_size)

    print("Computing benign perplexities...")
    benign_scores = fast_perplexity(model, benign_loader, device)

    print("Computing malicious perplexities...")
    mal_scores = fast_perplexity(model, mal_loader, device)

    evaluate(benign_scores, mal_scores)


if __name__ == "__main__":
    main()

