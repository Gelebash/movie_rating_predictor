from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import json

# Initialize the FastAPI app
app = FastAPI(title="Movie Rating Prediction API")

# --- Load all saved artifacts when the app starts ---
model = joblib.load("artifacts/model.joblib")

with open("artifacts/director_scores.json", "r") as f:
    director_scores = json.load(f)

with open("artifacts/writer_scores.json", "r") as f:
    writer_scores = json.load(f)

with open("artifacts/actor_scores.json", "r") as f:
    actor_scores = json.load(f)
    
with open("artifacts/genre_columns.json", "r") as f:
    genre_columns = json.load(f)

with open("artifacts/training_columns.json", "r") as f:
    training_columns = json.load(f)

# This is the average rating used to fill missing scores during training
# Replace with the actual mean from your notebook if you have it, otherwise this is a safe default
OVERALL_AVG_RATING = 6.8 

# Define the input data model using Pydantic
class MovieData(BaseModel):
    directors: list[str] = Field(..., example=["nm0634240"]) # Example: Christopher Nolan
    writers: list[str] = Field(..., example=["nm0634300", "nm0634240"])
    actors: list[str] = Field(..., example=["nm0000138", "nm0362766", "nm0000288"]) # Bale, Hardy, Hathaway
    genres: list[str] = Field(..., example=["Action", "Drama", "Thriller"])
    
# Define the prediction endpoint
@app.post("/predict/")
def predict_rating(movie: MovieData):
    """
    Predicts the IMDb rating for a movie based on its metadata.
    """
    # 1. Create a dictionary from the input
    data = {
        'directors': ','.join(movie.directors),
        'writers': ','.join(movie.writers),
        'actor_1_nconst': movie.actors[0] if len(movie.actors) > 0 else None,
        'actor_2_nconst': movie.actors[1] if len(movie.actors) > 1 else None,
        'actor_3_nconst': movie.actors[2] if len(movie.actors) > 2 else None,
        'genres': ','.join(movie.genres)
    }
    
    # 2. Convert to a DataFrame
    df = pd.DataFrame([data])
    
    # 3. Feature Engineering (must match the notebook exactly)
    
    # Target Encoding for people
    df['director_score'] = df['directors'].str.split(',').str[0].map(director_scores).fillna(OVERALL_AVG_RATING)
    df['writer_score'] = df['writers'].str.split(',').str[0].map(writer_scores).fillna(OVERALL_AVG_RATING)
    df['actor_1_score'] = df['actor_1_nconst'].map(actor_scores).fillna(OVERALL_AVG_RATING)
    df['actor_2_score'] = df['actor_2_nconst'].map(actor_scores).fillna(OVERALL_AVG_RATING)
    df['actor_3_score'] = df['actor_3_nconst'].map(actor_scores).fillna(OVERALL_AVG_RATING)

    # One-Hot Encoding for genres
    genres_dummies = df['genres'].str.replace(' ', '').str.replace(',', '|').str.get_dummies('|')
    # Align columns with the training set
    genres_df = pd.DataFrame(columns=genre_columns)
    genres_df = pd.concat([genres_df, genres_dummies])
    genres_df.fillna(0, inplace=True)
    genres_df = genres_df.astype(int)
    genres_df = genres_df[genre_columns] # Ensure correct order and columns

    # Combine all features
    features_df = pd.concat([df[['director_score', 'writer_score', 'actor_1_score', 'actor_2_score', 'actor_3_score']], genres_df], axis=1)

    # Ensure the same columns as used in training
    features_df = features_df[training_columns] 

    # 4. Make a prediction
    prediction = model.predict(features_df)[0]

    standard_float_prediction = float(prediction)

    return {"predicted_rating": round(standard_float_prediction, 2)}