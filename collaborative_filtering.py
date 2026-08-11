# %% [markdown]
# # Collaborative Filtering for Movie Recommendations

# %% [markdown]
# ## Introduction
# This notebook demonstrates collaborative filtering, a popular technique for building recommender systems. 
# Unlike content-based filtering, which uses item attributes, collaborative filtering makes recommendations based on the preferences and behaviors of other users.
# 
# We will cover three main approaches:
# 1.  **User-Based Collaborative Filtering**: Recommends items by finding users with similar tastes.
# 2.  **Item-Based Collaborative Filtering**: Recommends items that are similar to those a user has liked.
# 3.  **Model-Based Collaborative Filtering (SVD)**: Uses matrix factorization to discover latent features and predict ratings.

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import plotly.express as px

# %% [markdown]
# ## 1. Load and Prepare the Data
# We will use the `ratings.csv` and `movies.csv` datasets. The ratings data contains user-item interactions, which is the core of collaborative filtering.

# %%
# Load the datasets
ratings_df = pd.read_csv('data/movie/ratings.csv')
movies_df = pd.read_csv('data/movie/movies.csv')

# Merge ratings and movies dataframes to have movie titles
df = pd.merge(ratings_df, movies_df, on='movieId')

# Display the first few rows of the merged dataframe
print("Merged DataFrame (Ratings + Movies):")
print(df.head())

# %% [markdown]
# ### Create the User-Item Matrix
# A fundamental step in collaborative filtering is to create a user-item matrix, where rows represent users, columns represent movies, and the values are the ratings. This matrix is typically very sparse, as users only rate a small fraction of the available movies.

# %%
# Create the user-item matrix
user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating')

# Display the shape and a small part of the matrix
print("Shape of User-Item Matrix:", user_item_matrix.shape)
print("\nUser-Item Matrix (first 5x5):")
print(user_item_matrix.iloc[:5, :5])

# For computation, we'll fill NaN values with 0 and create a sparse matrix
user_item_matrix_sparse = csr_matrix(user_item_matrix.fillna(0).values)

# %% [markdown]
# ## 2. User-Based Collaborative Filtering
# This method finds users who have rated items similarly to the active user and recommends items that these similar users liked.

# %% [markdown]
# ### Calculate User Similarity
# We compute the cosine similarity between users based on their rating vectors. 
# This tells us how similar each user's taste is to every other user.

# %%
# Calculate user-user similarity
user_similarity = cosine_similarity(user_item_matrix_sparse)
user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

print("User-User Similarity Matrix (first 5x5):")
print(user_sim_df.iloc[:5, :5])

# %% [markdown]
# ### Generate Recommendations
# We create a function to recommend movies to a user. It works by finding the most similar users and identifying movies they rated highly that the active user has not yet seen.

# %%
def get_user_based_recommendations(user_id, user_item_matrix, user_sim_df,
                                   num_recommendations=10, k_neighbors=10):
    # Get the k most similar users, excluding the user themselves
    # (position 0 is the user, with similarity 1.0).
    similar_users = user_sim_df[user_id].sort_values(ascending=False).iloc[1:k_neighbors + 1]

    # Ratings of those neighbors: rows = neighbors, columns = movies, NaN = not rated.
    neighbor_ratings = user_item_matrix.loc[similar_users.index]

    # Each neighbor's vote is weighted by how similar they are to the active user.
    weights = similar_users.values.reshape(-1, 1)

    # Similarity-WEIGHTED average, computed per movie. A movie a neighbor hasn't
    # rated (NaN) must not drag the average down, so it contributes 0 to both the
    # weighted sum (numerator) and the summed weights (denominator).
    rated_mask = neighbor_ratings.notna()
    weighted_sum = (neighbor_ratings.fillna(0) * weights).sum(axis=0)
    weight_totals = (rated_mask * weights).sum(axis=0)
    recommendation_scores = weighted_sum / weight_totals.replace(0, np.nan)

    # Drop movies the active user has already rated, keep the best of the rest.
    user_rated_movies = user_item_matrix.loc[user_id].dropna().index
    recommendation_scores = recommendation_scores.drop(user_rated_movies, errors='ignore')

    # Return the top N recommended movies
    return recommendation_scores.nlargest(num_recommendations)

# %%
# Get recommendations for user 1
print("User-Based Recommendations for User 1:")
user1_recs = get_user_based_recommendations(1, user_item_matrix, user_sim_df)
print(user1_recs)
print(df[df["userId"] == 1].sort_values("rating", ascending=False))

# %% [markdown]
# ## 3. Item-Based Collaborative Filtering
# This method recommends items that are similar to items the user has already liked. 
# We calculate the similarity between items based on how users have rated them.

# %% [markdown]
# ### Calculate Item Similarity
# We compute cosine similarity on the transposed user-item matrix. This gives us a similarity score for every pair of movies.

# %%
# Calculate item-item similarity (we use the sparse matrix and transpose it)
item_similarity = cosine_similarity(user_item_matrix_sparse.T)
item_sim_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

print("Item-Item Similarity Matrix (first 5x5):")
print(item_sim_df.iloc[:5, :5])

# %% [markdown]
# ### Generate Recommendations
# We create a function that takes a movie title and finds other movies that are most similar to it.

# %%
def get_item_based_recommendations(movie_title, item_sim_df, num_recommendations=10):
    if movie_title not in item_sim_df:
        return f"Movie '{movie_title}' not found in the dataset."
    
    # Get similarity scores for the movie and sort them
    # Exclude the movie itself (similarity will be 1.0)
    similar_movies = item_sim_df[movie_title].sort_values(ascending=False).iloc[1:num_recommendations+1]
    
    return similar_movies

# %%
# Get recommendations similar to 'Toy Story (1995)'
print("Item-Based Recommendations for 'Toy Story (1995)':")
item_recs = get_item_based_recommendations('Toy Story (1995)', item_sim_df)
print(item_recs)

# %% [markdown]
# ## 4. Matrix Completion with Bias Correction (Baseline Predictor)
# Before jumping to a full model, it helps to build the simplest sensible "matrix
# completion" model: a **baseline predictor**. It fills every empty cell of the
# rating matrix with
#
#     r_hat(u, i) = mu + b_u + b_i
#
# where:
# - **mu** is the global average rating (the overall baseline),
# - **b_u** is the *user bias* — does this user rate more generously or more harshly than average?
# - **b_i** is the *item bias* — is this movie rated above or below average?
#
# This is the "adjusted average" idea from the lecture: instead of a plain average,
# we correct for the fact that some users are easy graders and some movies are
# simply more popular. It is robust and easy to interpret, but it knows nothing
# about *individual* taste (every user gets the same ranking of movies, shifted by
# their personal bias). That limitation is exactly what matrix factorization fixes
# in the next section.

# %%
def fit_baseline(ratings, n_users, n_items, reg=10.0, n_iters=15):
    """Learn a bias-corrected baseline r_hat = mu + b_u + b_i.

    We only look at *observed* ratings (true matrix completion). The user and item
    biases are estimated by alternating damped averages of the residuals; `reg`
    shrinks the bias of users/items with few ratings towards 0 so they don't swing
    wildly on little evidence.

    Args:
        ratings: array of (user_index, item_index, rating) rows.
        n_users, n_items: matrix dimensions.
        reg: regularization strength (larger -> biases pulled harder towards 0).
        n_iters: number of alternating passes.

    Returns:
        (mu, b_u, b_i) — the global mean and the bias vectors.
    """
    users = ratings[:, 0].astype(int)
    items = ratings[:, 1].astype(int)
    vals = ratings[:, 2]

    mu = vals.mean()
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)

    for _ in range(n_iters):
        # Update item biases from the residual after removing mu and current user bias.
        item_resid = vals - mu - b_u[users]
        sum_i = np.bincount(items, weights=item_resid, minlength=n_items)
        cnt_i = np.bincount(items, minlength=n_items)
        b_i = sum_i / (cnt_i + reg)

        # Update user biases from the residual after removing mu and current item bias.
        user_resid = vals - mu - b_i[items]
        sum_u = np.bincount(users, weights=user_resid, minlength=n_users)
        cnt_u = np.bincount(users, minlength=n_users)
        b_u = sum_u / (cnt_u + reg)

    return mu, b_u, b_i


# %% [markdown]
# ### Build integer index mappings
# Our models work on dense integer indices (0..n-1), so we map the raw `userId` /
# `movieId` values onto contiguous positions once and reuse them everywhere.

# %%
user_ids = ratings_df['userId'].unique()
item_ids = ratings_df['movieId'].unique()
user_to_idx = {uid: k for k, uid in enumerate(user_ids)}
item_to_idx = {iid: k for k, iid in enumerate(item_ids)}
idx_to_item = {k: iid for iid, k in item_to_idx.items()}
n_users, n_items = len(user_ids), len(item_ids)

# Encode the full ratings table as (user_index, item_index, rating) rows.
ratings_encoded = np.column_stack([
    ratings_df['userId'].map(user_to_idx).values,
    ratings_df['movieId'].map(item_to_idx).values,
    ratings_df['rating'].values,
]).astype(float)

# A held-out test split lets us measure how well each model predicts unseen ratings.
train_ratings, test_ratings = train_test_split(ratings_encoded, test_size=0.25, random_state=42)


def rmse(predictions, targets):
    """Root Mean Squared Error — the standard rating-prediction accuracy metric."""
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


# %%
# Fit the baseline on the training split and measure its test-set accuracy.
mu, base_b_u, base_b_i = fit_baseline(train_ratings, n_users, n_items)

test_users = test_ratings[:, 0].astype(int)
test_items = test_ratings[:, 1].astype(int)
test_vals = test_ratings[:, 2]

baseline_preds = np.clip(mu + base_b_u[test_users] + base_b_i[test_items], 0.5, 5.0)
print("Matrix Completion — Baseline Predictor (mu + b_u + b_i)")
print(f"  Global average rating (mu): {mu:.3f}")
print(f"  Test RMSE: {rmse(baseline_preds, test_vals):.4f}")


# %% [markdown]
# ## 5. Model-Based Collaborative Filtering: Matrix Factorization
# The baseline captures *who rates high* and *what is popular*, but not individual
# taste. **Matrix factorization** adds a small set of learned *latent factors* on
# top of the baseline:
#
#     r_hat(u, i) = mu + b_u + b_i + p_u . q_i
#
# Each user gets a short vector `p_u` and each movie a short vector `q_i` (here
# length 20). Their dot product captures taste dimensions the model discovers on
# its own — think "action-ness", "for-kids-ness", "arthouse-ness". We learn all of
# these parameters with plain **stochastic gradient descent** over the observed
# ratings, regularizing so the model generalizes. This is exactly the SVD-style
# "embedding" model from the lecture, written from scratch (no black-box library)
# so every line is inspectable.

# %%
class MatrixFactorization:
    """A minimal SGD matrix-factorization recommender with user/item biases."""

    def __init__(self, n_users, n_items, n_factors=20, n_epochs=20,
                 lr=0.01, reg=0.05, random_state=42):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr          # learning rate for the gradient steps
        self.reg = reg        # L2 regularization strength
        self.random_state = random_state

    def fit(self, ratings, verbose=True):
        rng = np.random.default_rng(self.random_state)
        self.mu = ratings[:, 2].mean()
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        # Small random factors break symmetry so SGD can specialize them.
        self.p = rng.normal(0, 0.1, (self.n_users, self.n_factors))
        self.q = rng.normal(0, 0.1, (self.n_items, self.n_factors))

        users = ratings[:, 0].astype(int)
        items = ratings[:, 1].astype(int)
        vals = ratings[:, 2]
        n = len(ratings)

        for epoch in range(self.n_epochs):
            order = rng.permutation(n)  # shuffle each epoch
            for idx in order:
                u, i, r = users[idx], items[idx], vals[idx]
                pred = self.mu + self.b_u[u] + self.b_i[i] + self.p[u] @ self.q[i]
                err = r - pred
                # Gradient step for every parameter that touches this rating.
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                p_u, q_i = self.p[u].copy(), self.q[i]
                self.p[u] += self.lr * (err * q_i - self.reg * p_u)
                self.q[i] += self.lr * (err * p_u - self.reg * q_i)
            if verbose:
                train_pred = self.predict(users, items)
                print(f"  epoch {epoch + 1:2d}/{self.n_epochs} — train RMSE: "
                      f"{rmse(train_pred, vals):.4f}")
        return self

    def predict(self, users, items):
        """Vectorized prediction for arrays of user/item indices."""
        users = np.asarray(users, dtype=int)
        items = np.asarray(items, dtype=int)
        dot = np.einsum('ij,ij->i', self.p[users], self.q[items])
        preds = self.mu + self.b_u[users] + self.b_i[items] + dot
        return np.clip(preds, 0.5, 5.0)


# %%
# Train the matrix-factorization model and compare its accuracy to the baseline.
print("\nModel-Based CF — Matrix Factorization (mu + b_u + b_i + p_u . q_i)")
mf = MatrixFactorization(n_users, n_items, n_factors=20, n_epochs=20)
mf.fit(train_ratings)

mf_test_preds = mf.predict(test_users, test_items)
print(f"\n  Baseline test RMSE:            {rmse(baseline_preds, test_vals):.4f}")
print(f"  Matrix Factorization test RMSE: {rmse(mf_test_preds, test_vals):.4f}")
print("  (lower is better — the latent factors add individual taste on top of the biases)")


# %% [markdown]
# ### Generate Recommendations with Matrix Factorization
# We predict a score for every movie the user hasn't rated yet and return the
# highest-scoring ones.

# %%
def get_mf_recommendations(user_id, model, movies_df, ratings_df,
                           user_to_idx, item_to_idx, idx_to_item, num_recommendations=10):
    if user_id not in user_to_idx:
        return f"User '{user_id}' not found in the dataset."

    u = user_to_idx[user_id]

    # Candidate movies = everything this user has NOT rated yet.
    rated_movie_ids = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
    candidate_items = [item_to_idx[iid] for iid in item_to_idx
                       if iid not in rated_movie_ids]

    # Score all candidates in one vectorized pass and take the top N.
    preds = model.predict(np.full(len(candidate_items), u), np.array(candidate_items))
    top_local = np.argsort(preds)[::-1][:num_recommendations]
    top_movie_ids = [idx_to_item[candidate_items[k]] for k in top_local]

    recs = movies_df[movies_df['movieId'].isin(top_movie_ids)][['movieId', 'title', 'genres']].copy()
    # Preserve the ranking order and attach the predicted rating.
    pred_by_id = {idx_to_item[candidate_items[k]]: preds[k] for k in top_local}
    recs['predicted_rating'] = recs['movieId'].map(pred_by_id)
    return recs.sort_values('predicted_rating', ascending=False)[['title', 'genres', 'predicted_rating']]


# %%
# Get matrix-factorization recommendations for user 1
print("\nMatrix-Factorization Recommendations for User 1:")
mf_recs = get_mf_recommendations(1, mf, movies_df, ratings_df,
                                 user_to_idx, item_to_idx, idx_to_item)
print(mf_recs.to_string(index=False))
print("\nWhat User 1 already rated highly (for comparison):")
print(df[df["userId"] == 1].sort_values("rating", ascending=False).head())

# %% [markdown]
# ## 6. Visualize: Movie Latent Factors
# The matrix-factorization model learns a latent-factor vector `q_i` for every
# movie. Movies that sit close together in this space are treated as similar *based
# on how people rated them* — no genre labels required. We reduce the 50-D factors
# to 2D with PCA (fast, deterministic) or t-SNE (better clusters) to see this
# "taste space".

# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Choose dimensionality reduction method: "pca" or "tsne"
use_dim_red = "pca"

# Use the *learned item factors* q_i — these are the model's movie embeddings.
movie_factors = mf.q

if use_dim_red == "pca":
    reducer = PCA(n_components=2)
    movie_factors_2d = reducer.fit_transform(movie_factors)
elif use_dim_red == "tsne":
    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    movie_factors_2d = reducer.fit_transform(movie_factors)
else:
    raise ValueError("use_dim_red must be 'pca' or 'tsne'")

# Map each factor row back to its raw movieId, then attach titles/genres.
factor_df = pd.DataFrame(movie_factors_2d, columns=['x', 'y'])
factor_df['movieId'] = [idx_to_item[k] for k in range(n_items)]
factor_df = pd.merge(factor_df, movies_df, on='movieId')

# %% [markdown]
# ### Interactive 2D Plot of Movie Factors
# Movies that are close together are considered similar by the model based on user
# rating patterns. Hover over points to see movie details.

# %%
# Create an interactive scatter plot (sample for rendering performance)
sample_df = factor_df.sample(min(2000, len(factor_df)), random_state=42)
fig = px.scatter(
    sample_df,
    x='x', y='y',
    hover_name='title',
    hover_data=['genres'],
    title=f'2D Representation of Movies based on Learned Latent Factors ({use_dim_red.upper()})'
)

# Improve the layout
fig.update_layout(
    xaxis_title=f"{use_dim_red.upper()} dimension 1",
    yaxis_title=f"{use_dim_red.upper()} dimension 2",
    title={
        'text': f"2D Movie 'Taste Space' from Matrix Factorization ({use_dim_red.upper()})",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

# Show the plot
fig.show()

# %% [markdown]
# ## Conclusion
# This notebook explored the collaborative-filtering family end to end:
# - **User-based and Item-based CF** — simple, interpretable "neighborhood-based"
#   methods driven by cosine similarity.
# - **Matrix Completion with bias correction** — a robust baseline (mu + b_u + b_i)
#   that corrects for generous raters and popular movies, but ignores individual taste.
# - **Matrix Factorization** — adds learned latent factors on top of that baseline,
#   improving prediction accuracy (lower RMSE) and revealing a 2D "taste space".
#
# Each method builds on the previous one, showing how recommenders move from simple
# averages to models that uncover latent structure in user behavior.
