from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch
import numpy as np

def get_curriculum_dataloader(args, algorithm, train_dataset, domain_indices, stage):
    """
    Create a curriculum-based DataLoader that feeds data domain-wise, sorted by classification loss.
    
    Args:
        args: argument namespace.
        algorithm: the current model (must have .forward() returning 'class' loss).
        train_dataset: combined dataset from multiple domains.
        domain_indices: dict of domain_name -> list of indices in train_dataset.
        stage: int (current training stage or epoch range).
        
    Returns:
        curriculum_loader: DataLoader with easy domains earlier.
    """

    domain_losses = []
    
    # Evaluate classification loss for each domain
    for domain, indices in domain_indices.items():
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            batch = tuple(item.cuda() for item in batch)
            output = algorithm.forward(batch)
            total_loss += output['class'].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        domain_losses.append((domain, avg_loss))
    
    # Sort domains by ascending classification loss (easiest first)
    domain_losses.sort(key=lambda x: x[1])
    
    # Select which domains to include based on the stage
    num_domains = len(domain_losses)
    num_selected = int(np.ceil((stage + 1) / args.CL_PHASE_EPOCHS * num_domains))
    selected_domains = [domain for domain, _ in domain_losses[:num_selected]]
    
    # Combine selected domains' indices
    selected_indices = []
    for domain in selected_domains:
        selected_indices.extend(domain_indices[domain])
    
    curriculum_subset = Subset(train_dataset, selected_indices)
    curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.N_WORKERS)
    
    return curriculum_loader
