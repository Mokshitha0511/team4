**Enhancing Domain Generalization for Out-of-Distribution Representation Learning for Time Series Classification**

This repository extends the Diversify domain generalization algorithm by introducing an automated method to estimate the optimal number of latent domains (latent_domain_num, also known as K) for training.

**Overview:**
Instead of manually setting the number of latent domains, this module analyzes the feature representations extracted by the model and applies **KMeans clustering combined with Silhouette Score** to determine the optimal number of clusters. This helps in better aligning the latent domains with the underlying data structure.

**How it works:**

1. Feature Extraction: After initial training epochs, features are extracted from the training set using the model’s featurizer.
2. K Estimation: The automated_k_estimation() function searches over a range of cluster counts (k_min to k_max) and computes the silhouette score for each.
3. Best K Selection: The number of clusters (K) with the highest silhouette score is selected as latent_domain_num and the same is updated.
4. Batch Size Adaptation: Based on the selected K, batch size is dynamically adjusted for balanced domain sampling:
    If K < 6: batch_size = 32 * K
    Otherwise: batch_size = 16 * K

**Requirements:**
We can follow the same requirements as of the original diversify. 
pip install -r requirements.txt

**Notes:**
1. k_min- Minimum number of clusters to consider is set to	2.
2. k_max-	Maximum number of clusters to consider is set to	10.
3. So, the optimal value of K is chosen within the specified range of limits.

**Example Output Log:**
...
[INFO] Optimal K determined as 5 (Silhouette Score: 0.6427)
Using automated latent_domain_num (K): 5
...
