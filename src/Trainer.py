import os
import torch
from tqdm.auto import tqdm
# Pending: Create Trainer class for cleaner and better code
def train_epoch(model, train_loader, optimizer, scheduler, device, step_based_scheduler=True):
    """Run one epoch of training using model's training_step method.
    args:
    model: model object
    train_loader
    optimizer
    scheduler
    device: device to train on (multi device not supported)
    step_based_scheduler: whether the scheduler is step based True or False
    
    """
    model.train()
    train_loss = 0
    train_steps = 0
    
    progress_bar = tqdm(train_loader, desc="[Train]", total=len(train_loader))
    for batch in progress_bar:
        batch = [x.to(device) for x in batch]
        
        # Use model's training_step
        loss = model.training_step(batch, train_steps)
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        
        # Update scheduler if it's step-based
        if step_based_scheduler and scheduler is not None:
            scheduler.step()
        
        batch = [x.to('cpu') for x in batch]
        
        train_loss += loss.item()
        train_steps += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': train_loss / train_steps})
        
    avg_train_loss = train_loss / train_steps 
    
    return avg_train_loss

def validate(model, val_loader, device):
    """Run validation using model's validation_step method."""
    model.eval()

    valid_loss = 0
    valid_steps = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="[Valid]", mininterval=10.0, total=len(val_loader))
        for batch in progress_bar:
            batch = [x.to(device) for x in batch]
            
            # Use model's validation_step
            loss = model.validation_step(batch, valid_steps)
            if isinstance(loss, dict):
                loss = loss['val_loss']
            elif isinstance(loss, torch.Tensor):
                loss = loss

            valid_loss += loss.item()
            valid_steps += 1
            # valid_tok += tok_acc
            # valid_seq += seq_acc
            
            # Update progress bar
            progress_bar.set_postfix({'loss': valid_loss / valid_steps})
            batch = [x.to('cpu') for x in batch]
            # break
    avg_valid_loss = valid_loss / valid_steps        
    return avg_valid_loss


def train_model(model, train_loader, val_loader, max_epochs=10, 
               patience=3, checkpoint_dir='checkpoints', device_id = 0):
    """
    Training loop with early stopping.
    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        max_epochs (int, optional): Maximum number of epochs to train the model. Defaults to 10.
        patience (int, optional): Number of epochs to wait before early stopping if validation performance does not improve. Defaults to 3.
        checkpoint_dir (str, optional): Directory to save model checkpoints. Defaults to 'checkpoints'.
        device_id (int, optional): ID of the GPU device to use for training. Defaults to 0.
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0

    optim_config = model.configure_optimizers()
    
    optimizer = optim_config["optimizer"]
    # Extract scheduler if available
    scheduler = optim_config.get("lr_scheduler", {}).get("scheduler", None)
    # Check if scheduler is step-based or epoch-based
    scheduler_interval = optim_config.get("lr_scheduler", {}).get("interval", "epoch")
    step_based_scheduler = scheduler_interval == "step"
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
                                device, step_based_scheduler)
        print(f"Training loss: {train_loss:.4f}")
        
        # Validation
        val_loss = validate(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'model-epoch{epoch+1:02d}-valloss{val_loss:.2f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    best_model_path = os.path.join(checkpoint_dir, sorted(os.listdir(checkpoint_dir))[-1])
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from {best_model_path}")
    
    return model