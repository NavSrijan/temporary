import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaForMaskedLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# ------------------------------
# Dataset
# ------------------------------
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=510):
        self.tokenizer = tokenizer
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [x.strip() for x in f if x.strip()]
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        enc = self.tokenizer(
            line,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "text": line
        }


# ------------------------------
# Pseudo-perplexity calculation
# ------------------------------
def pseudo_perplexity(model, input_ids, attention_mask, mask_token_id):
    length = attention_mask.sum().item()
    loss_sum = 0
    count = 0

    for pos in range(length):
        original_token = input_ids[pos].item()

        masked = input_ids.clone()
        masked[pos] = mask_token_id

        out = model(
            input_ids=masked.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        logits = out.logits[0, pos]
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        loss_sum += -log_prob[original_token].item()
        count += 1

    return np.exp(loss_sum / count)


# ------------------------------
# Evaluate a full dataset
# ------------------------------
def compute_scores(model, loader, tokenizer, device):
    mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
    scores = []

    print("Computing pseudo-perplexity...")
    with torch.no_grad():
        for batch in tqdm(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            for i in range(ids.size(0)):
                seq_pppl = pseudo_perplexity(
                    model, ids[i], mask[i], mask_token_id
                )
                scores.append(seq_pppl)

    return np.array(scores)


# ------------------------------
# Metrics
# ------------------------------
def evaluate(benign, malicious):
    y_true = np.concatenate([np.zeros(len(benign)), np.ones(len(malicious))])
    y_scores = np.concatenate([benign, malicious])

    auc = roc_auc_score(y_true, y_scores)
    print("\n=== METRICS ===")
    print("ROC-AUC:", round(auc, 4))

    t = np.percentile(benign, 99)
    preds = (y_scores > t).astype(int)

    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    print("Threshold:", t)
    print("Precision:", precision)
    print("Recall:", recall)


# ------------------------------
# Plotting
# ------------------------------
def plot_scores(benign, malicious):
    plt.figure(figsize=(12, 5))
    plt.hist(benign, bins=60, alpha=0.5, label="Benign", density=True)
    plt.hist(malicious, bins=60, alpha=0.5, label="Malicious", density=True)

    plt.title("Benign vs Malicious Pseudo-Perplexity")
    plt.xlabel("Pseudo-Perplexity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("waf_roberta_scores.png")
    print("Saved plot to waf_roberta_scores.png")


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--benign_file", required=True)
    parser.add_argument("--malicious_file", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=510)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    model = RobertaForMaskedLM.from_pretrained(args.model_path).to(device)
    model.eval()

    benign_set = TextDataset(args.benign_file, tokenizer, args.max_len)
    ben_loader = DataLoader(benign_set, batch_size=args.batch_size)

    malicious_set = TextDataset(args.malicious_file, tokenizer, args.max_len)
    mal_loader = DataLoader(malicious_set, batch_size=args.batch_size)

    benign_scores = compute_scores(model, ben_loader, tokenizer, device)
    malicious_scores = compute_scores(model, mal_loader, tokenizer, device)

    evaluate(benign_scores, malicious_scores)
    plot_scores(benign_scores, malicious_scores)


if __name__ == "__main__":
    main()

