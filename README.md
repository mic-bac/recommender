# Recommender Systems
Short overview about classic recommender systems

## Table of Contents
- [Introduction](#introduction)
- [Types of Recommender Systems](#types-of-recommender-systems)
- [Evaluation Metrics](#evaluation-metrics)

## Introduction
Recommender systems are algorithms designed to suggest relevant items to users based on various data points. They are widely used in e-commerce, streaming services, and social media platforms to enhance user experience and engagement.

## Types of Recommender Systems
There are many types of recommender systems, but the most common ones include:
1. **Content-Based Filtering**: Recommends items similar to those a user has liked in the past based on item features.
2. **Collaborative Filtering**: Recommends items based on the preferences of similar users.
   - User-Based Collaborative Filtering
   - Item-Based Collaborative Filtering
3. **Association Rule Mining**: Identifies patterns and associations between items to suggest related products.

Therefore, we will discuss these three types in detail.


### Content-Based Filtering (CBF)
Content-based filtering uses item features to recommend items similar to those a user has previously interacted with.
- **Advantages**:
  - Personalized recommendations based on user preferences.
  - No need for data from other users.
- **Disadvantages**:
  - Limited to the features of the items.
  - May lead to over-specialization.
- **Example**: If a user likes action movies, the system will recommend other action movies based on genre, actors, and directors.
- **Techniques**:
  - Cosine Similarity
  - TF-IDF (Term Frequency-Inverse Document Frequency)

An example implementation using Python and the `scikit-learn` library can be found [here](content_based_filtering.py).


### Collaborative Filtering (CF)
Collaborative filtering relies on the preferences of similar users to make recommendations.
- **Advantages**:
  - Can provide diverse recommendations.
  - Does not require item feature information.
- **Disadvantages**:
  - Cold start problem for new users/items.
  - Scalability issues with large datasets.
- **Example**: If User A and User B have similar ratings for several movies, the system may recommend movies liked by User B to User A.
- **Techniques**:
  - User-Based Collaborative Filtering
  - Item-Based Collaborative Filtering
  - Matrix Factorization (e.g., SVD, ALS)

An example implementation using Python and the `Surprise` library can be found [here](collaborative_filtering.py).


### Association Rule Mining (ARM)
Association rule mining identifies relationships between items based on user transactions. Compared to the above method, which rely on similarity, ARM can be beneficial for cross-selling activities.
- **Advantages**:
  - Can uncover hidden patterns in data.
  - Useful for market basket analysis.
- **Disadvantages**:
  - May generate a large number of rules, making it hard to identify the most relevant ones.
  - Requires a significant amount of transaction data.
- **Example**: If users frequently buy bread and butter together, the system may recommend butter when a user buys bread.
- **Techniques**:
  - Apriori Algorithm
  - FP-Growth Algorithm
- **Evaluation Metrics**:
  - Support
  - Confidence
  
An example implementation using Python and the `mlxtend` library can be found [here](association_rule_mining.py).