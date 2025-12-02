import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        encoding = self.tokenizer(
            line,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'text': line
        }

def calculate_perplexity(model, dataloader, device):
    model.eval()
    perplexities = []
    
    print("Calculating perplexity...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
            
    return np.array(perplexities)

def plot_comparison(benign_scores, mal_scores, output_file="anomaly_comparison.png"):
    plt.figure(figsize=(12, 6))
    
    # Histogram with KDE
    try:
        import seaborn as sns
        sns.histplot(benign_scores, kde=True, color='blue', label='Benign', stat="density", alpha=0.5, bins=50)
        sns.histplot(mal_scores, kde=True, color='red', label='Malicious', stat="density", alpha=0.5, bins=50)
    except ImportError:
        print("Seaborn not found, using standard Matplotlib.")
        plt.hist(benign_scores, bins=50, color='blue', alpha=0.5, label='Benign', density=True)
        plt.hist(mal_scores, bins=50, color='red', alpha=0.5, label='Malicious', density=True)

    plt.title('Benign vs Malicious Perplexity Distribution', fontsize=16)
    plt.xlabel('Perplexity (Anomaly Score)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file)
    print("Plot saved.")

def evaluate_metrics(benign_scores, mal_scores):
    # Label 0 for Benign, 1 for Malicious
    y_true = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(mal_scores))])
    y_scores = np.concatenate([benign_scores, mal_scores])
    
    roc_auc = roc_auc_score(y_true, y_scores)
    
    print("\n" + "="*40)
    print("EVALUATION METRICS")
    print("="*40)
    print(f"ROC-AUC Score: {roc_auc:.4f} (1.0 is perfect)")
    
    # Calculate Precision/Recall at 99th percentile of Benign
    threshold = np.percentile(benign_scores, 99)
    y_pred = (y_scores > threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("-" * 40)
    print(f"Threshold (99th % of Benign): {threshold:.4f}")
    print(f"Precision at Threshold:       {precision:.4f}")
    print(f"Recall (Detection Rate):      {recall:.4f}")
    print("="*40 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Compare Benign vs Malicious Perplexity")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--benign_file", type=str, required=True, help="Path to benign test data")
    parser.add_argument("--malicious_file", type=str, required=True, help="Path to malicious test data")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path}...")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_path)
    model = DistilBertForMaskedLM.from_pretrained(args.model_path).to(device)

    print("Processing Benign Data...")
    benign_dataset = TextDataset(args.benign_file, tokenizer, args.max_length)
    benign_loader = DataLoader(benign_dataset, batch_size=args.batch_size, shuffle=False)
    benign_scores = calculate_perplexity(model, benign_loader, device)

    print("Processing Malicious Data...")
    mal_dataset = TextDataset(args.malicious_file, tokenizer, args.max_length)
    mal_loader = DataLoader(mal_dataset, batch_size=args.batch_size, shuffle=False)
    mal_scores = calculate_perplexity(model, mal_loader, device)

    evaluate_metrics(benign_scores, mal_scores)
    plot_comparison(benign_scores, mal_scores)

if __name__ == "__main__":
    main()
