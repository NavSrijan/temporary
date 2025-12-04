import torch
import numpy as np
import pickle
import os
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.metrics import roc_auc_score

# --- CONFIG ---
MODEL_PATH = "./tiny_gpt2_waf"
BENIGN_FILE = "test_vx.txt"
MALICIOUS_FILE = "mal_test.txt"
CACHE_FILE = "loss_cache.pkl"
MAX_SAMPLES = 2000  # Limit samples for speed

def get_token_losses(text, model, tokenizer, device):
    """Runs model and returns a list of losses per token"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    if inputs["input_ids"].shape[1] < 2: return [] # Skip empty/single-token lines
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Shift to align prediction with target
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return token_losses.cpu().numpy().tolist()

def load_or_compute_losses():
    """Optimized: Computes losses once and saves them"""
    if os.path.exists(CACHE_FILE):
        print(f"Loading pre-computed losses from {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Computing losses (this requires GPU)...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    benign_losses = []
    mal_losses = []

    print("Processing Benign...")
    with open(BENIGN_FILE, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= MAX_SAMPLES: break
            if line.strip():
                benign_losses.append(get_token_losses(line.strip(), model, tokenizer, device))

    print("Processing Malicious...")
    with open(MALICIOUS_FILE, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= MAX_SAMPLES: break
            if line.strip():
                mal_losses.append(get_token_losses(line.strip(), model, tokenizer, device))

    data = {"benign": benign_losses, "mal": mal_losses}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    
    return data

# --- STRATEGIES ---

def strategy_mean(losses):
    """Baseline: Average of all tokens"""
    return np.mean(losses)

def strategy_max(losses):
    """Paranoid: The single most confusing token"""
    return np.max(losses)

def strategy_top_k_mean(losses, k_percent=0.2):
    """Robust: Average of the top 20% worst tokens"""
    k = max(1, int(len(losses) * k_percent))
    losses.sort(reverse=True)
    return np.mean(losses[:k])

def strategy_window_mean(losses, window_size=10):
    """Structural: The most confusing contiguous phrase"""
    if len(losses) < window_size: return np.mean(losses)
    # Moving average
    windows = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    return np.max(windows)

def strategy_median(losses):
    """Resistant: Ignores outliers completely"""
    return np.median(losses)

# --- EVALUATION ---

def evaluate_strategy(name, func, b_data, m_data):
    # Calculate scores
    b_scores = [func(l) for l in b_data if l]
    m_scores = [func(l) for l in m_data if l]
    
    if not b_scores or not m_scores: return

    # Labels: 0 = Benign, 1 = Malicious
    y_true = [0] * len(b_scores) + [1] * len(m_scores)
    y_scores = b_scores + m_scores
    
    # Metrics
    auc = roc_auc_score(y_true, y_scores)
    
    # WAF Simulation: Set threshold to catch 95% of attacks (Recall = 0.95)
    # Sort malicious scores to find the 5th percentile cutoff
    m_scores.sort()
    threshold_idx = int(len(m_scores) * 0.05)
    threshold = m_scores[threshold_idx] # This catches 95% of attacks
    
    # Calculate False Positive Rate at this threshold
    false_positives = sum(1 for s in b_scores if s >= threshold)
    fpr = (false_positives / len(b_scores)) * 100

    print(f"Strategy: {name:<20} | AUC: {auc:.4f} | FPR @ 95% Recall: {fpr:.2f}%")

def main():
    data = load_or_compute_losses()
    b_data = data["benign"]
    m_data = data["mal"]
    
    print("\n--- RESULTS ---")
    print(f"Evaluated on {len(b_data)} Benign and {len(m_data)} Malicious samples.\n")

    evaluate_strategy("Mean (Baseline)", strategy_mean, b_data, m_data)
    evaluate_strategy("Max Token", strategy_max, b_data, m_data)
    evaluate_strategy("Median", strategy_median, b_data, m_data)
    evaluate_strategy("Top 20% Mean", lambda x: strategy_top_k_mean(x, 0.2), b_data, m_data)
    evaluate_strategy("Top 10% Mean", lambda x: strategy_top_k_mean(x, 0.1), b_data, m_data)
    evaluate_strategy("Window (Size 5)", lambda x: strategy_window_mean(x, 5), b_data, m_data)
    evaluate_strategy("Window (Size 10)", lambda x: strategy_window_mean(x, 10), b_data, m_data)
    evaluate_strategy("Window (Size 20)", lambda x: strategy_window_mean(x, 20), b_data, m_data)

if __name__ == "__main__":
    main()
