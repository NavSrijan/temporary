import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd

# CONFIG
MODEL_PATH = "./tiny_gpt2_waf"
# Paste one of your high-scoring benign requests here
BAD_REQUEST = r"get /profile HTTP/1.1\nhost: localhost:8082\nuser-agent: Mozilla/5.0 (X11; Linux x86_64; rv:142.0) Gecko/20100101 Firefox/142.0"

def inspect_request():
    print(f"Loading {MODEL_PATH}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("\n--- TOKEN LOSS ANALYSIS ---")
    inputs = tokenizer(BAD_REQUEST, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        
        # Get raw logits and shift them (Prediction vs Reality)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        
        # Calculate loss PER TOKEN (no averaging)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Convert to readable tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][1:]) # Skip first token
    losses = token_losses.tolist()

    # Create a DataFrame for nice printing
    data = []
    for t, l in zip(tokens, losses):
        # Clean up the byte-level BPE characters (Ġ means space)
        clean_t = t.replace('Ġ', ' ').replace('Ċ', '\\n')
        data.append({"Token": clean_t, "Loss": l})

    df = pd.DataFrame(data)
    
    # 1. Print the "Panic" tokens (Loss > 5.0)
    print("\nXXX THE MODEL IS PANICKING HERE XXX")
    print(df[df["Loss"] > 5.0])
    
    # 2. Print the top 5 worst tokens
    print("\n--- TOP 5 WORST TOKENS ---")
    print(df.sort_values(by="Loss", ascending=False).head(5))

if __name__ == "__main__":
    inspect_request()
