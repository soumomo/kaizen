import torch 
import os
import torch.nn as nn
import numpy as np
from tokenizer import train_bpe_tokenizer , Tokenizer

if __name__ == "__main__":

    train_path = "/vol/TinyStoriesV2-GPT4-valid.txt"
    tokenized_data = {}
    tokenized_data["vocab"] , tokenized_data["merges"] = train_bpe_tokenizer(train_path, vocab_size=10000 , special_tokens= ["<|endoftext|>"]) 

    torch.save(tokenized_data, "/vol/tokenizer.pt")

    # tokenized_data = torch.load("data/tokenizer.pt")

    tokenizer = Tokenizer(
        vocab = tokenized_data["vocab"],
        merges = tokenized_data["merges"],
        special_tokens= ["<|endoftext|>"]
    )

    all_tokens = []
    with open("/vol/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            if idx >= 50000:
                    break
            
            if idx % 50000 == 0 and idx > 0:


                print(f"Processed {idx} lines...")

            if not line.strip():
                continue

            token_ids = tokenizer.encode(line)
            all_tokens.extend(token_ids)

    train_array = np.array(all_tokens, dtype=np.uint16)

    #ensuring the o/p directory exists
    os.makedirs("/vol", exist_ok=True)

    #saving the array as binary file
    np.save("/vol/train.npy", train_array)

    #clearing the original list and array from ram
    all_tokens = None
    del train_array 

    print("Training Data saved successfully and RAM cleared!")


    all_tokens = []
    with open("/vol/TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            
            if idx % 50000 == 0 and idx > 0:
                print(f"Processed {idx} lines...")

            if not line.strip():
                continue

            token_ids = tokenizer.encode(line)
            all_tokens.extend(token_ids)

    valid_array = np.array(all_tokens, dtype=np.uint16)

    #ensuring the o/p directory exists
    os.makedirs("/vol", exist_ok=True)

    #saving the array as binary file
    np.save("/vol/validation.npy", valid_array)

    #clearing the original list and array from ram
    all_tokens = None
    del valid_array 

    print("Validation Data saved successfully and RAM cleared!")