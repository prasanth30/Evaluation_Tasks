import torch
from tqdm.auto import tqdm

def calculate_accuracy(model, dataloader, device, limit=None):
    """
    Function to calculate metrics on dataloader
    args:
    model - model class with test_step
    dataloader - dataloader to use
    device - device name to infer on
    limit - how much to limit the evaluation samples to
    """
    model.eval()
    model.to(device)
    total_seq = 0
    correct_seq = 0
    total_tokens = 0
    correct_tokens = 0
    loss_sum = 0
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), desc="Calculating accuracy",total=len(dataloader)):
            if limit and idx>limit:
                break
            
            # Move batch to device
            batch = [x.to(device) for x in batch]
            outputs, labels = model.test_step(batch, 0)
            outputs = outputs.cpu()[:,1:]
            l = outputs.shape[1]
            labels = labels.cpu()
            labels = labels[:,:l]
            
            # if idx == 4:
            #     print(outputs[0], labels[0])

            total_tokens += outputs.numel()
            correct_tokens += ((outputs == labels)).sum().item()
            
            batch_size = labels.size(0)
            seq_correct = torch.all((outputs == labels), dim=1)
            correct_seq += seq_correct.sum().item()
            total_seq += batch_size

    # Calculate metrics
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    sequence_accuracy = correct_seq / total_seq if total_seq > 0 else 0
    avg_loss = loss_sum / len(dataloader)
    
    return {
        'token_accuracy': token_accuracy,
        'sequence_accuracy': sequence_accuracy,
        'loss': avg_loss
    }