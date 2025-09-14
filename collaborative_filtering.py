# %% [markdown]
# # Collaborative Filtering for Movie Recommendations

# %% [markdown]
# ## Introduction
# This notebook demonstrates collaborative filtering, a popular technique for building recommender systems. Unlike content-based filtering, which uses item attributes, collaborative filtering makes recommendations based on the preferences and behaviors of other users.
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
from scipy.sparse import csr_matrix
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
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
# We compute the cosine similarity between users based on their rating vectors. This tells us how similar each user's taste is to every other user.

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
def get_user_based_recommendations(user_id, user_item_matrix, user_sim_df, num_recommendations=10):
    # Get top 10 most similar users, excluding the user themselves
    similar_users = user_sim_df[user_id].sort_values(ascending=False).iloc[1:11]
    
    # Get the movies rated by these similar users
    similar_users_ratings = user_item_matrix.loc[similar_users.index]
    
    # Calculate the weighted average of ratings for each movie
    # We only consider movies that at least one of the similar users has rated
    recommendation_scores = similar_users_ratings.mean(axis=0)
    
    # Get movies the active user has already rated
    user_rated_movies = user_item_matrix.loc[user_id].dropna().index
    
    # Filter out movies the user has already seen
    recommendation_scores = recommendation_scores.drop(user_rated_movies, errors='ignore')
    
    # Return the top N recommended movies
    return recommendation_scores.nlargest(num_recommendations)

# %%
# Get recommendations for user 1
print("User-Based Recommendations for User 1:")
user1_recs = get_user_based_recommendations(1, user_item_matrix, user_sim_df)
print(user1_recs)

# %% [markdown]
# ## 3. Item-Based Collaborative Filtering
# This method recommends items that are similar to items the user has already liked. We calculate the similarity between items based on how users have rated them.

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
# ## 4. Model-Based Collaborative Filtering (SVD)
# Model-based methods use machine learning techniques to predict ratings. Singular Value Decomposition (SVD) is a matrix factorization method that decomposes the user-item matrix into lower-dimensional matrices representing latent "factors" for users and items.

# %%
# The Surprise library requires data in a specific format
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Initialize and train the SVD model
svd = SVD(n_factors=50, n_epochs=20, random_state=42) # n_factors is the number of latent features
svd.fit(trainset)

# %% [markdown]
# ### Generate Recommendations with SVD
# We can now use our trained SVD model to predict ratings for movies a user hasn't seen and recommend the ones with the highest predicted scores.

# %%
def get_svd_recommendations(user_id, svd_model, movies_df, ratings_df, num_recommendations=10):
    # Get a list of all movie IDs
    all_movie_ids = movies_df['movieId'].unique()
    
    # Get the list of movies the user has already rated
    rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    
    # Get movies the user has not rated yet
    unrated_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]
    
    # Predict ratings for the unrated movies
    predictions = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
    
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get the top N recommendations
    top_n_preds = predictions[:num_recommendations]
    
    # Get the movie IDs and predicted ratings
    recommended_movie_ids = [pred.iid for pred in top_n_preds]
    recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
    
    return recommended_movies[['title', 'genres']]

# %%
# Get SVD-based recommendations for user 1
print("SVD-Based Recommendations for User 1:")
svd_recs = get_svd_recommendations(1, svd, movies_df, ratings_df)
print(svd_recs)

# %% [markdown]
# ## 5. Visualize: Movie Latent Factors from SVD
# The SVD model learns a set of latent factors for each movie. We can think of these factors as capturing underlying characteristics like "action-ness," "comedy-ness," or "for-kids-ness." By reducing these factors to 2D using PCA, we can visualize the relationships between movies in a "taste space."

# %%
# Get the latent factor matrix for items (movies) from the SVD model
# Note: Surprise stores internal IDs, so we need to map them back to movieIds
movie_factors = svd.qi

# Reduce the dimensionality of the movie factors to 2D using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
movie_factors_2d = pca.fit_transform(movie_factors)

# Create a DataFrame for the 2D data
# We need to map the inner IDs from the model back to the original movie titles
trainset_inner_to_raw_iids = {inner_id: raw_id for raw_id, inner_id in trainset.ir.items()}
movie_indices = [trainset_inner_to_raw_iids[i] for i in range(movie_factors.shape[0])]

# Get titles for these movieIds
factor_df = pd.DataFrame(movie_factors_2d, columns=['x', 'y'])
factor_df['movieId'] = movie_indices
factor_df = pd.merge(factor_df, movies_df, on='movieId')

# %% [markdown]
# ### Interactive 2D Plot of Movie Factors
# This plot shows movies in a 2D space. Movies that are close together are considered similar by the SVD model based on user rating patterns. Hover over points to see movie details.

# %%
# Create an interactive scatter plot
fig = px.scatter(factor_df.sample(2000, random_state=42), # Plot a sample for performance
                 x='x', y='y',
                 hover_name='title',
                 hover_data=['genres'],
                 title='2D Representation of Movies based on SVD Latent Factors')

# Improve the layout
fig.update_layout(
    xaxis_title="Latent Factor 1 (PCA)",
    yaxis_title="Latent Factor 2 (PCA)",
    title={
        'text': "2D Movie 'Taste Space' from SVD",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

# Show the plot
fig.show()

# %% [markdown]
# ## Conclusion
# This notebook explored three fundamental collaborative filtering techniques.
# - **User-based and Item-based CF** are simple, interpretable methods often called "neighborhood-based" approaches.
# - **Model-based CF (SVD)** is a more powerful, scalable approach that can uncover latent patterns in user behavior, often leading to better prediction accuracy.
# 
# Each method has its strengths and is a foundational concept in building modern recommender systems.
