# 🎯 Practical Recommender Systems
An educational guide to understanding and implementing recommendation systems, perfect for students in machine learning and data science.

## 📚 What You'll Learn
- How recommendation systems work in real-world applications (like Netflix, Amazon, Spotify)
- Three fundamental approaches to building recommender systems
- Hands-on implementation using Python and popular data science libraries
- Working with real movie and retail datasets

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Basic Python programming knowledge
- Understanding of basic data structures (lists, dictionaries)
- Familiarity with pandas and numpy (basic operations)

### Installation in 3 Easy Steps
1. Clone this repository:
```bash
git clone https://github.com/mic-bac/recommender.git
cd recommender
```

2. Create a conda environment (recommended):
```bash
conda env create -f conda_env.yaml
conda activate recommender
```

3. Or install packages directly with pip:
```bash
pip install pandas numpy scikit-learn mlxtend plotly
```

## 📂 Project Structure
```
recommender/
│
├── data/                          # Datasets
│   ├── groceries/                # Retail transaction data
│   │   └── Groceries_dataset.csv
│   └── movie/                    # MovieLens dataset
│       ├── movies.csv           # Movie details
│       └── ratings.csv          # User ratings
│
├── content_based_filtering.py     # Movie recommendations using genres
├── collaborative_filtering.py     # User and item-based recommendations
└── association_rule_mining.py     # Market basket analysis
```

## 🎥 Three Ways to Recommend

### 1. Content-Based Filtering (`content_based_filtering.py`)
Think of it as "If you like this movie, you'll like similar movies"
- ✨ **How it works**: Recommends movies based on their genres and features
- 🎯 **Use case**: Netflix suggesting movies similar to ones you've watched
- 📝 **Example**:
  ```python
  from content_based_filtering import get_recommendations
  similar_movies = get_recommendations('Toy Story (1995)')
  ```

### 2. Collaborative Filtering (`collaborative_filtering.py`)
Think of it as "People who like what you like also like..."
- 👥 **User-Based**: Finds users with similar taste (similarity-weighted ratings)
- 🎬 **Item-Based**: Finds similar items based on user ratings
- ⚖️ **Matrix Completion with bias correction**: A baseline predictor `mu + b_u + b_i`
  that corrects for generous raters and popular movies
- 🔢 **Matrix Factorization**: Adds learned latent factors on top of the baseline
  (`mu + b_u + b_i + p_u·q_i`), trained from scratch with SGD — no black-box library
- 📝 **Example**:
  ```python
  from collaborative_filtering import get_user_based_recommendations
  recommendations = get_user_based_recommendations(1, user_item_matrix, user_sim_df)
  ```

### 3. Association Rule Mining (`association_rule_mining.py`)
Think of it as "Frequently bought together"
- 🛒 **Market Basket Analysis**: Discovers shopping patterns (Apriori & FP-Growth)
- 📊 **Key Metrics**: Support, Confidence, Lift
- 📝 **Example**:
  ```python
  from association_rule_mining import get_basket_recommendations
  related_items = get_basket_recommendations('whole milk')
  ```

## 📊 Included Datasets

### 🎬 MovieLens Dataset
- **Contents**: 100,000 ratings, 9,000 movies
- **Features**: Titles, Genres, User Ratings
- **Perfect for**: Learning collaborative & content-based filtering

Source: [Kaggle MovieLens Dataset](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset/data)

### 🛒 Groceries Dataset
- **Contents**: Real supermarket transactions
- **Features**: Customer purchases over time
- **Perfect for**: Learning association rule mining

Source: [Kaggle Groceries Dataset](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset/data)

## 📈 Evaluation
The scripts include lightweight, honest evaluation of what each method actually does:
- 🎯 **Accuracy (RMSE)**: The collaborative-filtering script holds out 25% of the
  ratings and reports test-set RMSE for both the bias baseline and the matrix-
  factorization model, so you can see the latent factors improve prediction.
- 🌈 **Diversity (Intra-List Similarity)**: The content-based script measures how
  similar the recommended items are to each other — a lower score means more
  diverse suggestions.

## 🤝 How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/CoolFeature`)
3. Commit your changes (`git commit -m 'Add CoolFeature'`)
4. Push to the branch (`git push origin feature/CoolFeature`)
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Learning Path
1. Start with Content-Based Filtering (simplest to understand)
2. Move to Collaborative Filtering (most widely used)
3. Explore Association Rules (great for retail applications)
