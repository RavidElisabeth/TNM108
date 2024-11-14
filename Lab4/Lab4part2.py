import sklearn
import os
from sklearn.datasets import load_files

# Define the path relative to the script location
current_directory = os.path.dirname(os.path.abspath(__file__))
moviedir = os.path.join(current_directory, 'movie_reviews')

# loading all files. 
movie = load_files(moviedir, shuffle=True)

print(len(movie.data))

# target names ("classes") are automatically generated from subfolder names
print(movie.target_names)

# First file seems to be about a Schwarzenegger movie. 
print(movie.data[0][:500])

# first file is in "neg" folder
print(movie.filenames[0])

# first file is a negative review and is mapped to 0 index 'neg' in target_names
print(movie.target[0])