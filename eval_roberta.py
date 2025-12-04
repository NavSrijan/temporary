import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaForMaskedLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=510):
        self.tokenizer = tokenizer
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]
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
            "text": line
        }


def compute_perplexities(model, loader, device):
    model.eval()
    scores = []

    print("Computing perplexities...")
    with torch.no_grad():
        for batch in tqdm(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            out = model(input_ids=ids, attention_mask=mask, labels=ids)
            loss = out.loss
            scores.append(float(torch.exp(loss)))

    return np.array(scores)


def evaluate(benign, malicious):
    y_true = np.concatenate([
        np.zeros(len(benign)),
        np.ones(len(malicious))
    ])
    y_score = np.concatenate([benign, malicious])

    auc = roc_auc_score(y_true, y_score)

    print("\n=== METRICS ===")
    print("ROC-AUC:", round(auc, 4))

    thresh = np.percentile(benign, 99)
    preds = (y_score > thresh).astype(int)

    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    print("Threshold (99th percentile benign):", round(thresh, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))


def plot_scores(benign, malicious):
    plt.figure(figsize=(12, 5))
    plt.hist(benign, bins=50, alpha=0.5, label="Benign", density=True)
    plt.hist(malicious, bins=50, alpha=0.5, label="Malicious", density=True)
    plt.legend()
    plt.xlabel("Perplexity")
    plt.ylabel("Density")
    plt.title("Benign vs Malicious Perplexity Distribution")
    plt.tight_layout()
    plt.savefig("waf_roberta_scores.png")
    print("Saved plot to waf_roberta_scores.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benign_file", type=str, required=True)
    parser.add_argument("--malicious_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=510)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading tokenizer and model...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    model = RobertaForMaskedLM.from_pretrained(args.model_path).to(device)

    print("Loading benign dataset...")
    benign_set = TextDataset(args.benign_file, tokenizer, args.max_len)
    benign_loader = DataLoader(benign_set, batch_size=args.batch_size, shuffle=False)

    print("Loading malicious dataset...")
    malicious_set = TextDataset(args.malicious_file, tokenizer, args.max_len)
    malicious_loader = DataLoader(malicious_set, batch_size=args.batch_size, shuffle=False)

    benign_scores = compute_perplexities(model, benign_loader, device)
    malicious_scores = compute_perplexities(model, malicious_loader, device)

    evaluate(benign_scores, malicious_scores)
    plot_scores(benign_scores, malicious_scores)


if __name__ == "__main__":
    main()

