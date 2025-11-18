import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import argparse # <--- Added this import
from argparse import Namespace
import os

# --- Import all model and utility files ---
# This is necessary to populate the models.models registry
# with all the @register calls.
import models
import utils

# --- Import all dataset files ---
# This populates the datasets.datasets registry
import datasets

# --- Training Function ---
def train_model(epochs=100, batch_size=4, lr=1e-4, div2k_hr_path=None, num_workers=8):
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This model requires a GPU.")
        return
    device = torch.device("cuda")

    # --- Data Loading ---
    if div2k_hr_path is None or not os.path.exists(div2k_hr_path):
        print("="*50)
        print("ERROR: DIV2K HR training path is not set or does not exist.")
        print("Please download the DIV2K 800 training images (HR)")
        print("and set the 'div2k_hr_path' variable in main() to its location.")
        print(f"Path was: {div2k_hr_path}")
        print("="*50)
        return

    print("Initializing dataset...")
    # 1. Define the dataset specification
    # This spec uses the loaders you provided to build the dataset pipeline:
    # 'sr-implicit-downsampled' (from wrappers.py)
    #   -> 'image-folder' (from image_folder.py)
    
    dataset_spec = {
        'name': 'sr-implicit-downsampled',
        'args': {
            'dataset': {
                'name': 'image-folder',
                'args': {
                    'root_path': div2k_hr_path,
                    'cache': 'in_memory' # Use 'in_memory' for speed if you have enough RAM
                }
            },
            'inp_size': 256,       # As per paper: 256x256 HR crop
            'scale_min': 4.0,      # As per paper
            'scale_max': 8.0,      # As per paper
            'augment': True,
            'batch_per_gpu': batch_size
        }
    }
    
    # 2. Initialize the Dataset using datasets.make
    dataset = datasets.make(dataset_spec)
    print(f"Dataset '{div2k_hr_path}' loaded successfully.")
    
    # 3. Initialize DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    print("Initializing model...")
    # 4. Define the model specification
    # This spec tells models.make() how to build the ContinuousGaussian model
    # and its encoder.
    
    # Encoder spec: use 'edsr-baseline' from edsr.py
    encoder_spec = {
        'name': 'edsr-baseline',
        'args': {
            'n_resblocks': 16,
            'n_feats': 64,
            'res_scale': 1,
            'no_upsampling': True,
            'rgb_range': 1,
        }
    }
    
    # Main model spec: 'continuous-gaussian' from gaussian.py
    # Note: The __init__ in gaussian.py also takes cnn_spec and fc_spec,
    # but doesn't appear to use them. We pass them as empty dicts.
    model_spec = {
        'name': 'continuous-gaussian',
        'args': {
            'encoder_spec': encoder_spec,
            'cnn_spec': {},
            'fc_spec': {}
        }
    }

    # 5. Initialize the Model using models.make
    model = models.make(model_spec).to(device)
    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 6. Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 7. Initialize Loss Function
    criterion = nn.L1Loss()

    print(f"Starting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")

    # --- Training Loop ---
    model.train()
    total_steps = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0

        for i, batch in enumerate(dataloader):
            step_start_time = time.time()
            
            # Unpack the batch dictionary from the dataset wrapper
            lr_batch = batch['inp'].to(device)
            hr_batch = batch['gt'].to(device)
            scale_batch = batch['scale'] # This is a 1D tensor [s, s, s, ...]
            
            # Get the single scale value for this batch
            # (The wrapper ensures all items in a batch have the same scale)
            s = scale_batch[0].item()
            
            # Format the scale tensor as the model's forward() expects
            scale_tensor = torch.tensor([[s, s]], device=device) 

            # 8. Forward Pass
            optimizer.zero_grad()
            pred_batch = model(lr_batch, scale_tensor)
            
            # 9. Calculate Loss
            # Ensure predicted batch and HR batch are the same size
            # (The loader logic should already guarantee this)
            if pred_batch.shape != hr_batch.shape:
                 # In case of rounding errors in the loader
                 pred_batch = F.interpolate(pred_batch, size=hr_batch.shape[-2:], mode='bicubic', align_corners=False)

            loss = criterion(pred_batch, hr_batch)

            # 10. Backward Pass and Optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_steps += 1
            
            if (i + 1) % 20 == 0: # Print log every 20 steps
                step_time = time.time() - step_start_time
                steps_per_sec = 20.0 / step_time
                print(f"[Epoch {epoch+1}/{epochs}] [Step {i+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} | Steps/sec: {steps_per_sec:.2f}")

        epoch_loss = running_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        print("-" * 50)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")
        print("-" * 50)

        if (epoch + 1) % 10 == 0:
            save_path = f"continuous_sr_epoch_{epoch+1}.pth"
            # --- FIX START ---
            # 'demo.py' expects a dictionary with a 'model' key.
            # That 'model' key should contain the model_spec (architecture).
            # The 'models.make' function also expects the weights to be
            # inside that spec at the key 'sd'.
            
            # 1. Add the learned weights (state_dict) into the model_spec dict
            model_spec['sd'] = model.state_dict()
            
            # 2. Save the checkpoint in the format demo.py expects
            checkpoint = {
                'model': model_spec
            }
            torch.save(checkpoint, save_path)
            # --- FIX END ---
            print(f"Model checkpoint saved to {save_path}")

    print("Training complete.")

# --- Main execution ---
if __name__ == "__main__":
    
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Train ContinuousSR Model")
    
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs. (Default: 100)')
                        
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size per GPU. (Default: 4, Reduce if you get CUDA OOM)')
                        
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate. (Default: 1e-4)')
                        
    parser.add_argument('--path', type=str, required=True, 
                        help='Path to the DIV2K HR training images folder (e.g., /path/to/DIV2K_train_HR)')
                        
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='Number of dataloader workers. (Default: 8)')

    args = parser.parse_args()
    
    # 2. Call the training function with parsed arguments
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        div2k_hr_path=args.path,
        num_workers=args.num_workers
    )
