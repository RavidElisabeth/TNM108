import sklearn
from sklearn.datasets import load_files
import pandas as pd
import zipfile

# Unzip the dataset
with zipfile.ZipFile('election-day-tweets.zip', 'r') as zip_ref:
    zip_ref.extractall('election_day_tweets_data')  # Extract to a specific folder

# Load the CSV file
file_path = 'election_day_tweets_data/election_day_tweets.csv'  # Replace with the actual file name
elections_2016 = pd.read_csv(file_path)

elections_2016.sort_values(by='created_at').info()

