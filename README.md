**Enhancing Domain Generalization for Out-of-Distribution Representation Learning for Time Series Classification**

This repository extends the Diversify domain generalization algorithm by integrating a Curriculum Learning strategy based on domain-wise validation loss.

**Overview:**
This extension progressively trains the model on domains that are easier to generalize, determined by average validation loss per domain in contrary to the adverserial training in the original diversify. The core idea is to improve generalization by ordering training domains from easiest to hardest across curriculum learning phases.
We have introduced a new data loader called curriculum loader which has the logic to implement the curriculum learning for the first few specified epochs.
As part of this extension, new parameters such as --curriculum and --CL_PHASE_EPOCHS have been introduced to enable and control curriculum learning stages during training. If --curriculum is set to true, the training process follows a curriculum learning strategy for the number of epochs specified by --CL_PHASE_EPOCHS (default is to set 5). After completing these epochs, the model resumes standard training.

**Curriculum Loader Details:**
1. Split dataset using split_dataset_by_domain: Ensures both train and validation sets contain samples from each domain.
2. Estimate domain difficulty: For each domain, validation loss is computed using a trained model. Domains are sorted by average loss (easiest = lowest loss).
3. Progressive inclusion: At each curriculum stage, a subset of domains (based on training progress) is included. Number of domains is determined by:
    progress = min(1.0, ((stage + 1) / args.CL_PHASE_EPOCHS) ** 0.5)
4. Custom SubsetWithLabelSetter: Allows future label remapping if needed.

**Requirements:**
We can follow the same requirements as of the original diversify.
pip install -r requirements.txt

**Usage:**
Use the following command to run training with curriculum learning enabled:
!python train.py --data_dir ./data/ --task cross_people --test_envs 0 --dataset emg --algorithm diversify --latent_domain_num 6 --alpha1 1.0 --alpha 1.0 --lam 0.0 --local_epoch 3 --max_epoch 50 --lr 0.01 --curriculum --CL_PHASE_EPOCHS 5 --output ./data/train_output/act/cross_people-emg-Diversify-0-10-1-1-0-3-50-0.01

**Notes:**
1. Domain is assumed to be stored at index 2 in each data sample: x, y, domain = dataset[i].
2. The default value of CL_PHASE_EPOCHS is set to 5, meaning that if not explicitly provided via arguments, curriculum learning will be applied during the first five training rounds.

**Example Output Log:**
...
--- Domain Ranking by Difficulty (easiest to hardest) ---
1. Domain 0.0: Avg Val Loss = 0.3190
2. Domain 2.0: Avg Val Loss = 0.3628
3. Domain 1.0: Avg Val Loss = 0.5746
...
  

