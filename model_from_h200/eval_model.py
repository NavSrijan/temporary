import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# CONFIGURATION
# ----------------------------------------
MODEL_PATH = "./tiny_gpt2_waf"
BENIGN_TEST_FILE = "test_vx.txt"
MALICIOUS_TEST_FILE = "mal_test.txt"
WINDOW_SIZE = 5  # The winning strategy from benchmarks
# ----------------------------------------

def load_system():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device

def get_anomaly_score(text, model, tokenizer, device):
    """
    Calculates the anomaly score using Sliding Window Mean (Size 5).
    This detects dense bursts of high-perplexity tokens (like SQLi phrases)
    while ignoring isolated random tokens (like UUIDs).
    """
    # 1. Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Filter short requests
    if inputs["input_ids"].shape[1] < 2: 
        return 0.0
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 2. Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # 3. Shift for Next-Token Prediction
    # We want to compare Prediction[t] vs Actual[t+1]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    
    # 4. Compute Raw Loss per Token
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Move to CPU for math
    losses = token_losses.cpu().numpy()
    
    # 5. Apply Winning Strategy: Sliding Window Mean
    if len(losses) < WINDOW_SIZE:
        return np.mean(losses)
    
    # Fast convolution to get sliding window averages
    # This creates an array of means for every window of size 5
    windows = np.convolve(losses, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
    
    # Return the score of the single most suspicious window
    return np.max(windows)

def evaluate_file(filepath, model, tokenizer, device, max_samples=2000):
    scores = []
    print(f"Reading {filepath}...")
    try:
        with open(filepath, "r", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                if i >= max_samples: break
                
                score = get_anomaly_score(line, model, tokenizer, device)
                scores.append((score, line))
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found.")
        return []

    return scores

def main():
    model, tokenizer, device = load_system()

    print("\n--- 1. QUICK SANITY CHECK ---")
    # A standard GET request
    benign_sample = "GET /api/v1/user/profile?id=12345&token=abc HTTP/1.1"
    # A SQL Injection
    attack_sample = "GET /api/v1/user?id=1' UNION SELECT user, password FROM users -- HTTP/1.1"
    
    score_b = get_anomaly_score(benign_sample, model, tokenizer, device)
    score_a = get_anomaly_score(attack_sample, model, tokenizer, device)
    
    print(f"Benign Sample Score: {score_b:.4f}")
    print(f"Attack Sample Score: {score_a:.4f}")
    print(f"Gap: {score_a - score_b:.4f}")

    print("\n--- 2. MASS EVALUATION ---")
    benign_data = evaluate_file(BENIGN_TEST_FILE, model, tokenizer, device)
    mal_data = evaluate_file(MALICIOUS_TEST_FILE, model, tokenizer, device)

    if not benign_data or not mal_data:
        print("Missing data files. Exiting.")
        return

    b_scores = [s[0] for s in benign_data]
    m_scores = [s[0] for s in mal_data]

    avg_b = np.mean(b_scores)
    avg_m = np.mean(m_scores)
    
    print(f"\nAverage Benign Score:    {avg_b:.4f}")
    print(f"Average Malicious Score: {avg_m:.4f}")
    
    # Suggest Threshold based on 95th percentile of Benign (Block mostly attacks)
    # OR 5th percentile of Malicious (Catch 95% of attacks)
    m_scores.sort()
    threshold_idx = int(len(m_scores) * 0.05) # 5th percentile
    suggested_threshold = m_scores[threshold_idx]
    
    print(f"Suggested Threshold (95% Recall): {suggested_threshold:.4f}")

    print("\n--- 3. DEBUGGING FALSE POSITIVES ---")
    print(f"Inspecting Benign requests with score > {suggested_threshold:.4f}...")

    count = 0
    for score, line in benign_data:
        if score > suggested_threshold:
            print(f"[{score:.2f}] {line[:120]}...") 
            count += 1
            #if count > 5: break
    print(count)

if __name__ == "__main__":
    main()
