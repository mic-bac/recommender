# %% [markdown]
# # Content-Based Filtering for Movie Recommendations

# %% [markdown]
# ## Introduction
# This notebook demonstrates content-based filtering, a type of recommender system that suggests items based on their attributes. 
# We will use a dataset of movies and their genres to build a system that recommends movies similar to a user's choice.
# 
# We will cover two main parts:
# 1.  **Calculate**: Using TF-IDF to represent movie genres and Cosine Similarity to find similar movies.
# 2.  **Visualize**: Visualizing the movie data in a 2D space to understand the relationships between them.

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

# %% [markdown]
# ## 1. Load and Prepare the Data
# We start by loading the `movies.csv` dataset and preparing it for analysis.

# %%
# Load the movies dataset
movies_df = pd.read_csv('data/movie/movies.csv')

# Display the first few rows of the dataframe
print("Original Movies DataFrame:")
print(movies_df.head())

# The genres are listed as a string with '|' as a separator. 
# We will replace the '|' with spaces to treat the genres as a single string of words.
movies_df['genres'] = movies_df['genres'].str.replace('|', ' ', regex=False)

# Display the first few rows of the modified dataframe
print("\nModified Movies DataFrame:")
print(movies_df.head())

# %% [markdown]
# ## 2. Calculate: TF-IDF and Cosine Similarity

# %% [markdown]
# ### TF-IDF Vectorization
# We will use Term Frequency-Inverse Document Frequency (TF-IDF) to convert the text-based genre data into a numerical 
# format that can be used for calculations. Each movie's genres will be represented as a vector.

# %%
# Initialize the TF-IDF Vectorizer
# We use stop_words='english' to remove common English words that don't add much meaning.
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the genres column to create the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])

# The tfidf_matrix is a sparse matrix where each row represents a movie and each column represents a genre term.
print("\nShape of TF-IDF Matrix:")
print(tfidf_matrix.shape)
print(tfidf_matrix[:3, :3].toarray()) # first 3x3

# %% [markdown]
# ### Cosine Similarity
# Now that we have the TF-IDF matrix, we can calculate the cosine similarity between all pairs of movies. 
# The resulting matrix will show how similar each movie is to every other movie.

# %%
# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("\nShape of Cosine Similarity Matrix:")
print(cosine_sim.shape)
print(cosine_sim[:3, :3]) # first 3x3

# %% [markdown]
# ### Create a Recommendation Function
# We will create a function that takes a movie title as input and returns a list of the most similar movies.

# %%
# Create a function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim, movies_df=movies_df, return_input=True):
    # Get the index of the movie that matches the title
    try:
        idx = movies_df[movies_df['title'] == title].index[0]
    except IndexError:
        return f"Movie with title '{title}' not found."

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # remove base item
    del sim_scores[idx]

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[:10]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # For demonstrational purposes we can return the input too
    if return_input:
        movie_indices.insert(0, idx)

    # Return the top 10 most similar movies
    return movies_df['title'].iloc[movie_indices]

# %% [markdown]
# ### Test the Recommendation System
# Let's test our recommendation system with a few examples.

# %%
# Get recommendations for 'Toy Story (1995)'
print("\nRecommendations for 'Toy Story (1995)':")
toy_rec = get_recommendations('Toy Story (1995)')
print(movies_df.loc[toy_rec.index])


# Get recommendations for 'The Lego Movie (2014)'
print("\nRecommendations for 'The Lego Movie (2014)':")
print(get_recommendations('The Lego Movie (2014)'))

# Get recommendations for 'X-Men (2000)'
print("\nRecommendations for 'X-Men (2000)':")
print(get_recommendations('X-Men (2000)'))

# %% [markdown]
# ## 3. Visualize: Visualizing Movie Similarities
# To better understand the relationships between movies, we can visualize the TF-IDF vectors in a 2D space. 
# We'll use Principal Component Analysis (PCA) and t-SNE to reduce the dimensionality of the TF-IDF matrix.
#
# | Aspect                | PCA                                                                 | t-SNE                                                                       |
# |-----------------------|---------------------------------------------------------------------|-----------------------------------------------------------------------------|
# | **Type**              | Linear dimensionality reduction                                     | Nonlinear dimensionality reduction                                          |
# | **How it works**      | Projects data onto directions of maximum variance                   | Preserves local neighborhood structure by minimizing KL divergence          |
# | **Pros**              | Fast, deterministic, easy to interpret                              | Captures complex nonlinear patterns, great for visualizing clusters         |
# | **Cons**              | Only captures linear relationships, may miss complex structures     | Computationally expensive, sensitive to hyperparameters, less interpretable |


# %%
# Choose between PCA and TSNE - could be interesting
use_dim_red = "tsne"

if use_dim_red == "pca":
    # Reduce the dimensionality of the TF-IDF matrix using PCA
    pca = PCA(n_components=2)
    tfidf_matrix_2d = pca.fit_transform(tfidf_matrix.toarray())
if use_dim_red == "tsne":
    # Reduce TF-IDF matrix to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tfidf_matrix_2d = tsne.fit_transform(tfidf_matrix.toarray())

# Create a new DataFrame for the 2D data
movies_2d_df = pd.DataFrame(tfidf_matrix_2d, columns=['x', 'y'])
movies_2d_df['title'] = movies_df['title']
movies_2d_df['genres'] = movies_df['genres']

# %% [markdown]
# ### Interactive 2D Plot with Plotly
# Now, we'll create an interactive scatter plot using Plotly. 
# You can hover over the points to see the movie titles and genres.

# %%
# Create a new color column (default all points to grey)
color_discrete_map = {'darkgrey': 'darkgrey', 'red': 'red'}
movies_2d_df["custom_color"] = "darkgrey"

# Assign custom colors for specific indices
movies_2d_df.loc[0, "custom_color"] = "red"      # Toy Story (1995)
movies_2d_df.loc[8357, "custom_color"] = "red"   # The Lego Movie (2014)
movies_2d_df.loc[2836, "custom_color"] = "red"   # X-Men (2000)

# Add jitter: small random noise for better visibility due to many overlapping cases
jitter_strength = 0.1  # Adjust the amount of jitter
movies_2d_df['x_jitter'] = movies_2d_df['x'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(movies_2d_df))
movies_2d_df['y_jitter'] = movies_2d_df['y'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(movies_2d_df))

# sorted to have movies of interest on top of dots
movies_2d_df_sorted = movies_2d_df.sort_values("custom_color", axis=0, ignore_index=True)

# Create an interactive scatter plot
fig = px.scatter(movies_2d_df, 
                 x='x_jitter', y='y_jitter',
                 hover_name='title',
                 hover_data=['genres'],
                 color="custom_color",
                 color_discrete_map=color_discrete_map,
                 title=f'2D Representation of Movies based on Genres (TF-IDF + {use_dim_red})')

# Improve the layout
fig.update_layout(
    xaxis_title=f"{use_dim_red} dimension 1",
    yaxis_title=f"{use_dim_red} dimension 2",
    title={
        'text': "2D Representation of Movies based on Genres",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

# Show the plot
fig.show()


# %% [markdown]
# ## 4. Evaluation
# To evaluate our content-based recommender, we can measure the diversity of the recommendations. 
# A good recommender should not only be accurate but also provide a diverse set of items.
# 
# ### Intra-List Similarity
# Intra-list similarity measures how similar the recommended items are to each other. 
# A lower value means the recommendations are more diverse. 
# We calculate it by averaging the cosine similarity between all pairs of items in the recommendation list.

# %%
def calculate_intra_list_similarity(recommendations, cosine_sim=cosine_sim, movies_df=movies_df):
    """
    Calculates the average similarity between all pairs of items in a recommendation list.
    """
    # Get indices of recommended movies
    rec_indices = movies_df[movies_df['title'].isin(recommendations)].index
    
    # If there are less than 2 recommendations, diversity is not applicable
    if len(rec_indices) < 2:
        return 0.0
        
    total_similarity = 0
    pair_count = 0
    
    # Iterate through all pairs of recommended movies
    for i in range(len(rec_indices)):
        for j in range(i + 1, len(rec_indices)):
            idx1 = rec_indices[i]
            idx2 = rec_indices[j]
            
            # Add the similarity score to the total
            total_similarity += cosine_sim[idx1][idx2]
            pair_count += 1
            
    # Return the average similarity
    return total_similarity / pair_count if pair_count > 0 else 0.0

# %% [markdown]
# ### Evaluate Recommendations for 'Toy Story (1995)'

# %%
# Get recommendations for 'Toy Story (1995)'
toy_story_recs = get_recommendations('Toy Story (1995)')
print("Recommendations for 'Toy Story (1995)':")
print(toy_story_recs)

# Calculate and print the intra-list similarity
ils_toy_story = calculate_intra_list_similarity(toy_story_recs)
print(f"\nIntra-List Similarity for 'Toy Story (1995)' recommendations: {ils_toy_story:.4f}")
print("A lower score indicates more diverse recommendations.")
print(movies_df.loc[toy_story_recs.index])

# %% [markdown]
# ### Evaluate Recommendations for 'Jumanji (1995)'

# %%
# Get recommendations for 'Jumanji (1995)'
jumanji_recs = get_recommendations('Jumanji (1995)')
print("\nRecommendations for 'Jumanji (1995)':")
print(jumanji_recs)

# Calculate and print the intra-list similarity
ils_jumanji = calculate_intra_list_similarity(jumanji_recs)
print(f"\nIntra-List Similarity for 'Jumanji (1995)' recommendations: {ils_jumanji:.4f}")
print("A lower score indicates more diverse recommendations.")

# %% [markdown]
# ## 5. Getting Recommendations for a User Profile
# We can extend the recommendation logic to handle a user's history, where different movies might have different levels of importance. The following function takes a list of movie titles and corresponding weights to create a "user profile" vector. It then recommends movies based on this aggregated profile.

# %%

# Create this mapping once to avoid re-creating it on every function call
title_to_idx = pd.Series(movies_df.index, index=movies_df['title'])

def get_recommendations_for_user_profile(titles, weights, title_to_idx, movies_df, tfidf_matrix, top_n=10):
    """
    Recommends movies based on a weighted list of user's favorite movies.
    
    Args:
        titles (list): A list of movie titles.
        weights (list): A list of weights corresponding to each movie.
        title_to_idx (pd.Series): A mapping from movie titles to their indices.
        movies_df (pd.DataFrame): The DataFrame of movies.
        tfidf_matrix (scipy.sparse.matrix): The TF-IDF matrix of movie genres.
        top_n (int): The number of recommendations to return.
        
    Returns:
        A pandas Series of recommended movie titles, or an error string.
    """
    if len(titles) != len(weights):
        return "Error: 'titles' and 'weights' lists must have the same length."

    # Get indices of input movies, skipping those not found
    movie_indices = [title_to_idx[title] for title in titles if title in title_to_idx]
    
    if not movie_indices:
        return "Error: None of the provided movies were found."

    # Calculate the weighted average of the movie vectors
    # This operates efficiently on the sparse matrix
    user_profile_vector = np.average(
            tfidf_matrix[movie_indices].toarray(), 
            axis=0, 
            weights=weights
    )

    # Calculate cosine similarity between the user profile and all movies
    cosine_similarities = cosine_similarity(user_profile_vector.reshape(1, -1), tfidf_matrix)
    
    # Get similarity scores as a pandas Series
    sim_scores = pd.Series(cosine_similarities[0], index=movies_df.index)
    
    # Drop the input movies from the recommendations
    sim_scores = sim_scores.drop(movie_indices)
    
    # Return the top N most similar movies
    return movies_df['title'].loc[sim_scores.nlargest(top_n).index]

# %% [markdown]
# ### Test the User Profile Recommendation System
# Let's test the function with a sample user profile. We'll give a higher weight to an animated movie and a lower weight to an action movie to see how it influences the recommendations.

# %%
# Example: User likes 'Toy Story (1995)' a lot, and 'Jumanji (1995)' a little.
user_movies = ['Toy Story (1995)', 'Jumanji (1995)']
user_weights = [1.0, 0.5]

profile_recs = get_recommendations_for_user_profile(
    titles=user_movies, 
    weights=user_weights, 
    title_to_idx=title_to_idx, 
    movies_df=movies_df, 
    tfidf_matrix=tfidf_matrix
)

print(f"Recommendations for a user who likes '{user_movies[0]}' (weight={user_weights[0]}) and '{user_movies[1]}' (weight={user_weights[1]}):")
print(profile_recs)

# %% [markdown]
# ## Conclusion
# This notebook provided a step-by-step guide to building a content-based 
# filtering recommender system. We used TF-IDF and cosine similarity to find 
# similar movies and visualized the results using PCA, t-SNE and Plotly. 
# This approach can be extended by incorporating more features (like actors, directors) 
# or by using more advanced techniques for feature representation.
# We also added evaluation metrics, including intra-list similarity 
# to assess diversity of our recommendations.
