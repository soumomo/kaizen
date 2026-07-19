import argparse
import torch
import torch.nn as nn
import numpy as np
from model import TransformerLM
from optimizer import AdamW
from training import get_batch, load_checkpoint, save_checkpoint
from nn_utils import cross_entropy, lr_cosine_schedule, gradient_clipping
import time
from tokenizer import Tokenizer
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a Transformer model with memory-mapped data."
    )

    # data paths and loading ;-;
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/train.npy",
        help="Path to training .npy file",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/validation.npy",
        help="Path to validation .npy file",
    )
    parser.add_argument(
        "--mmap",
        type=str,
        default="r",
        choices=["r", "r+", "w+", "c", "None"],
        help="Memory mapping mode",
    )

    # model architecture (^ ^')
    parser.add_argument(
        "--d_model", type=int, default=512, help="Embedding dimension size"
    )
    parser.add_argument(
        "--num_heads", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument(
        "--rope_theta", type=float, default=10000.0, help="Base value for RoPE calculation"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=1344,
        help="Dimension of feed-forward network",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=10000, help="Vocabulary size"
    )

    # training loop configs t_t
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size per training step"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_iters", type=int, default=10000, help="Total training iterations"
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold",
    )

    # lr and scheduler +_+
    parser.add_argument(
        "--max_lr", type=float, default=6e-4, help="Peak learning rate"
    )
    parser.add_argument(
        "--min_lr", type=float, default=6e-5, help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=2000,
        help="Number of iterations for LR warmup",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay factor"
    )

    # evaluation and checkpointing :/
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="How often to run evaluation",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=200,
        help="Number of batches to run during evaluation",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="How often to save model checkpoints",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="How often to print training logs",
    )

    return parser.parse_args()


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    args = get_args()

    mmap_mode = None if args.mmap == "None" else args.mmap

    train_data = np.load(args.train_path, mmap_mode=mmap_mode)
    val_data = np.load(args.val_path, mmap_mode=mmap_mode)

    print(f"data successfully mapped!!")
    print(f"Model configured with d_model={args.d_model}, heads={args.num_heads}")
    print(f"Training will run for {args.max_iters} iterations.")

    model = TransformerLM(args.vocab_size , args.context_length , args.d_model, args.num_layers , args.num_heads , args.d_ff , args.rope_theta)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr = args.max_lr,
        weight_decay=args.weight_decay
    )

    model.train()
    start_time = time.time()
    
    for step in range(args.max_iters):
        lr = lr_cosine_schedule(
            step, args.max_lr, args.min_lr,args.warmup_iters, args.max_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(
            train_data, args.batch_size, args.context_length, device
        )

        optimizer.zero_grad()
        logits = model(x)
        #logits shape : [batch_size , context_length ,vocab_size]
        # y shape: [batch_size , context_length]
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()

        gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip_norm)
        optimizer.step()

        # logging
        if step % args.log_interval == 0 or step == args.max_iters - 1:
          elapsed = time.time() - start_time
          print(
              f"step {step:5d} | loss: {loss.item():6.4f} | lr: {lr:.2e} | time: {elapsed:.1f}s"
          )

        # evaluation
        if step % args.eval_interval == 0 and step > 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for _ in range(args.eval_iters):
                    x_val, y_val = get_batch(val_data, args.batch_size, args.context_length, device)
                    logits_val = model(x_val)
                    loss_val = cross_entropy(logits_val.reshape(-1, logits_val.shape[-1]), y_val.reshape(-1))
                    val_losses.append(loss_val.item())
            
            val_loss = np.mean(val_losses)
            print(f"\n > < Evaluation at step {step} | Val Loss: {val_loss:6.4f} > <")
            
            model.train()

          
          # checkpointing
        if step % args.checkpoint_interval == 0 and step > 0:
            checkpoint_path = f"/vol/checkpoint_step_{step}.pt"
            save_checkpoint(model, optimizer, step, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path} (^ ^)")

    generate(model, "Once upon a time", context_length=args.context_length)


def generate(
    model,  # accepting the instantiated model object directly :P
    prompt: str, 
    checkpoint_path: str= None, 
    tokenizer_path: str = "/vol/tokenizer.pt", 
    max_tokens: int = 100, 
    temperature: float = 0.8, 
    top_p: float = 0.9, 
    context_length: int = 128
):
  if torch.backends.mps.is_available():
        device = torch.device("mps")
  elif torch.cuda.is_available():
        device = torch.device("cuda")
  else:
        device = torch.device("cpu")

  tokenized_data = torch.load(tokenizer_path, map_location="cpu")
  tokenizer = Tokenizer(
        vocab=tokenized_data["vocab"],
        merges=tokenized_data["merges"],
        special_tokens=["<|endoftext|>"]
    )
  eos_id = tokenizer.encoder.get(b"<|endoftext|>", None)

  if checkpoint_path is not None:
      print(f"Loading weights from disk checkpoint: {checkpoint_path}")
      checkpoint = torch.load(checkpoint_path, map_location=device)
      if "model_state_dict" in checkpoint:
          model.load_state_dict(checkpoint["model_state_dict"])
      else:
          model.load_state_dict(checkpoint["model"])
  else:
      print("Using active model weights already existing in-memory.")

  model.to(device)
  model.eval()

  prompt_tokens = tokenizer.encode(prompt)
  x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

  print(f"Generating text for prompt: {prompt} (^^')")

  #core generation loop
  with torch.no_grad():
        #limiting the growing sequence
        for _ in range(max_tokens):
            if x.size(1) > context_length:
                # FIX 1: Added colon to keep 2D slicing active
                x_input = x[:, -context_length:] 
            else:
                x_input = x

            logits = model(x_input)
            # slices out the last position's logits -> shape [vocab_size]
            logits = logits[0, -1 , :]

            #temperature scaling
            if temperature > 0:
                logits  = logits / temperature
              
            #top_p scaling
            if top_p < 1.0:
                sorted_logits , sorted_indices = torch.sort(logits, descending=True)
                #convert logit to sorted probabilities
                sorted_probs = F.softmax(sorted_logits, dim=-1)

                #apply the shifting trick
                cumulative_probs_shifted = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
                sorted_indices_to_remove = cumulative_probs_shifted > top_p
                
                # map the mask back to the original unsorted logits indices
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token_id.unsqueeze(0)), dim=1)
            
            if eos_id is not None and next_token_id.item() == eos_id:
                break

  generated_ids = x.squeeze(0).tolist()
  generated_text = tokenizer.decode(generated_ids)
  print(generated_text)
  return generated_text




if __name__ == "__main__":
    main()
    



