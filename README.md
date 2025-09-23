# Movie Rating Predictor

This project implements an XGBoost machine learning model to predict IMDb movie ratings based on features from IMDb datasets.

## Project Structure

```
movie_rating_predictions/
│
├── data/                      # Data directory
│   ├── final_movie_data.parquet  # Processed dataset used for modeling
│   ├── name.basics.tsv       # IMDb raw data
│   ├── title.basics.tsv      # IMDb raw data
│   ├── title.crew.tsv        # IMDb raw data
│   ├── title.principals.tsv  # IMDb raw data
│   └── title.ratings.tsv     # IMDb raw data
│
├── artifacts/                 # Saved model and feature mappings
│   ├── model.joblib          # Trained XGBoost model
│   ├── actor_scores.json     # Average ratings for actors
│   ├── director_scores.json  # Average ratings for directors
│   ├── writer_scores.json    # Average ratings for writers
│   ├── genre_columns.json    # Genre features used in training
│   └── training_columns.json # Order of columns for model input
│
├── data_cleaning.ipynb       # Initial data preprocessing
└── feature_engineering_and_modeling.ipynb  # Feature creation and model training
```

## Project Workflow

1. **Data Cleaning (`data_cleaning.ipynb`)**
   - Loads and merges IMDb dataset files
   - Filters for movies with over 1000 votes
   - Processes movie metadata (year, runtime, etc.)
   - Creates dataset with top 3 actors per movie
   - Saves cleaned data as `final_movie_data.parquet`

2. **Feature Engineering and Modeling (`feature_engineering_and_modeling.ipynb`)**
   - One-hot encodes movie genres
   - Calculates average ratings for directors, writers, and actors
   - Trains XGBoost model using top 8 most important features
   - Evaluates model performance using R² and MAE metrics
   - Saves model and feature mappings to artifacts folder

## Model Results

To view the model's performance:
1. Open `feature_engineering_and_modeling.ipynb`
2. Look for the following results near the end:
   - R² score on test data
   - Mean Absolute Error (MAE)
   - Feature importance rankings
   - Train vs Test performance comparison

## Getting Started

1. Clone the repository
2. Place the required IMDb dataset files in the `data/` directory
3. Run the notebooks in order:
   - First: `data_cleaning.ipynb`
   - Second: `feature_engineering_and_modeling.ipynb`

## Requirements

- Python 3.x
- Jupyter Notebook
- Required packages:
  - pandas
  - xgboost
  - scikit-learn
  - numpy
  - joblib