from torch.utils.data import DataLoader, Subset
import numpy as np
import torch

def get_curriculum_loader(args, algorithm, train_dataset, stage):
    """
    Create a curriculum-based DataLoader that feeds data domain-wise, sorted by classification loss.
    
    Args:
        args: argument namespace.
        algorithm: the current model (must have .forward() returning 'class' loss).
        train_dataset: combined dataset from multiple domains.
        stage: current training round or epoch group.
    
    Returns:
        curriculum_loader: DataLoader with easy domains earlier.
    """

    # === Domain-wise grouping ===
    domain_indices = {}
    if hasattr(train_dataset, 'domains'):
        for idx, d in enumerate(train_dataset.domains):
            domain_indices.setdefault(d, []).append(idx)
    else:
        for idx in range(len(train_dataset)):
            domain = train_dataset[idx][2]  # assumes domain is at index 2
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

    # Sort by easiest domains first
    domain_losses.sort(key=lambda x: x[1])

    # Calculate how many domains to select for this stage
    num_domains = len(domain_losses)
    progress = min(1.0, (stage + 1) / args.CL_PHASE_EPOCHS)  # Cap at 1.0
    num_selected = max(1, int(np.ceil(progress * num_domains)))
    selected_domains = [domain for domain, _ in domain_losses[:num_selected]]

    selected_indices = []
    for domain in selected_domains:
        selected_indices.extend(domain_indices[domain])

    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices)
    curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.N_WORKERS)

    return curriculum_loader


class SubsetWithLabelSetter(Subset):
    def set_labels_by_index(self, labels, indices, key):
        if hasattr(self.dataset, 'set_labels_by_index'):
            self.dataset.set_labels_by_index(labels, indices, key)
