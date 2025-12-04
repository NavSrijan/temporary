import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math
import numpy as np

# CONFIGURATION
# ----------------------------------------
MODEL_PATH = "./tiny_gpt2_waf"  # Where 2_train_model.py saved the output
BENIGN_TEST_FILE = "test.txt" # Use a separate test set if you have one, or a subset of train
MALICIOUS_TEST_FILE = "mal_test.txt"
# ----------------------------------------

def load_system():
    print(f"Loading model from {MODEL_PATH}...")
    # We use GPT2TokenizerFast explicitly to match the training fix
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval() # Disable dropout for deterministic results
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device

def get_anomaly_score(text, model, tokenizer, device):
    """
    Improved Scoring: Uses 'Windowed Loss' or 'Max Token Loss' 
    to prevent normal headers from hiding the attack.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Get the full logits (raw predictions)
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Shift logits and labels so we compare Prediction(t) vs Actual(t+1)
    # This aligns the prediction with the target token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    
    # Calculate CrossEntropy loss for EACH token individually (no reduction)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # STRATEGY: 
    # Instead of average, we take the average of the 'Top K' worst tokens.
    # This highlights the attack payload while ignoring the headers.
    
    # Option A: Max single token surprise (Can be noisy)
    # return token_losses.max().item()
    
    # Option B: Average of the top 20% most surprising tokens (Robust)
    k = max(1, int(len(token_losses) * 0.20)) # Top 20%
    top_k_losses, _ = torch.topk(token_losses, k)
    
    return top_k_losses.mean().item()

def evaluate_file(filepath, model, tokenizer, device, max_samples=1000):
    scores = []
    print(f"Reading {filepath}...")
    try:
        with open(filepath, "r", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                if i >= max_samples: break
                
                score = get_anomaly_score(line, model, tokenizer, device)
                scores.append(score)
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found. Skipping.")
        return []

    return scores

def main():
    model, tokenizer, device = load_system()

    print("\n--- 1. QUICK SANITY CHECK ---")
    # A standard GET request (Should have LOW score)
    benign_sample = "GET /api/v1/user/profile?id=12345&token=abc HTTP/1.1"
    
    # A SQL Injection (Should have HIGH score)
    # Note: 'UNION' and 'SELECT' might be in vocab, but the sequence is rare in benign traffic
    attack_sample = "GET /api/v1/user?id=1' UNION SELECT user, password FROM users -- HTTP/1.1"
    
    score_b = get_anomaly_score(benign_sample, model, tokenizer, device)
    score_a = get_anomaly_score(attack_sample, model, tokenizer, device)
    
    print(f"Benign Sample Score: {score_b:.4f}")
    print(f"Attack Sample Score: {score_a:.4f}")
    print(f"Gap: {score_a - score_b:.4f}")

    if score_a > score_b:
        print("SUCCESS: The model finds the attack more 'surprising' than the benign request.")
    else:
        print("WARNING: The model thinks the attack is more normal than the benign request.")

    print("\n--- 2. MASS EVALUATION ---")
    benign_scores = evaluate_file(BENIGN_TEST_FILE, model, tokenizer, device)
    mal_scores = evaluate_file(MALICIOUS_TEST_FILE, model, tokenizer, device)

    if benign_scores and mal_scores:
        avg_b = np.mean(benign_scores)
        avg_m = np.mean(mal_scores)
        
        print(f"\nAverage Benign Score:    {avg_b:.4f}")
        print(f"Average Malicious Score: {avg_m:.4f}")
        
        # Simple threshold suggestion
        suggested_threshold = (avg_b + avg_m) / 2
        print(f"Suggested Threshold:     {suggested_threshold:.4f}")

        # ... inside main(), after calculating avg_m ...

    print("\n--- 3. DEBUGGING FALSE POSITIVES ---")
    threshold = 9.35 # Or whatever the suggested threshold is
    print(f"Inspecting Benign requests with score > {threshold}...")

    count = 0
    with open(BENIGN_TEST_FILE, "r") as f:
        for line in f:
            line = line.strip()
            score = get_anomaly_score(line, model, tokenizer, device)
            if score > threshold:
                print(f"[{score:.2f}] {line[:100]}...") # Print first 100 chars
                count += 1
                if count > 5: break # Only show 5 examples

if __name__ == "__main__":
    main()
