from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import os

# CONFIGURATION
# ----------------
TRAIN_FILES = ["train.txt", "mal_test.txt"] # Paths to your raw text datasets
VOCAB_SIZE = 30000  # 30k is standard and safe
MIN_FREQUENCY = 2   # Drop tokens that appear only once

def train_tokenizer():
    print("Training Tokenizer...")
    
    # Initialize a Byte-Level BPE tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Train it on your files
    tokenizer.train(files=TRAIN_FILES, vocab_size=VOCAB_SIZE, min_frequency=MIN_FREQUENCY, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save to disk
    os.makedirs("./waf_tokenizer", exist_ok=True)
    tokenizer.save_model("./waf_tokenizer")
    
    print("Tokenizer saved to ./waf_tokenizer")

if __name__ == "__main__":
    train_tokenizer()
