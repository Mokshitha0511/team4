from torch.utils.data import DataLoader, Subset
import numpy as np
import torch

def get_curriculum_loader(args, algorithm, train_dataset):
    """
    Create a curriculum-based DataLoader that feeds data domain-wise, sorted by classification loss.
    
    Args:
        args: argument namespace.
        algorithm: the current model (must have .forward() returning 'class' loss).
        train_dataset: combined dataset from multiple domains.

    Returns:
        curriculum_loader: DataLoader with easy domains earlier.
    """

    # Stage set to 0 by default for initial curriculum
    stage = 0

    # === Domain-wise grouping ===
    domain_indices = {}
    if hasattr(train_dataset, 'domains'):  # custom Dataset with 'domains' attribute
        for idx, d in enumerate(train_dataset.domains):
            domain_indices.setdefault(d, []).append(idx)
    else:
        # fallback: assume train_dataset[i][2] returns domain id
        for idx in range(len(train_dataset)):
            domain = train_dataset[idx][2]
            domain_indices.setdefault(domain, []).append(idx)

    domain_losses = []

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

    # Sort domains by ascending loss (easy to hard)
    domain_losses.sort(key=lambda x: x[1])

    num_domains = len(domain_losses)
    num_selected = int(np.ceil((stage + 1) / args.CL_PHASE_EPOCHS * num_domains))
    selected_domains = [domain for domain, _ in domain_losses[:num_selected]]

    selected_indices = []
    for domain in selected_domains:
        selected_indices.extend(domain_indices[domain])

    curriculum_subset = Subset(train_dataset, selected_indices)
    curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.N_WORKERS)

    return curriculum_loader
