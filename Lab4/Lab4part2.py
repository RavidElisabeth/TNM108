import sklearn
from sklearn.datasets import load_files

moviedir = r'D:\Lab\nltk_data\corpora\movie_reviews'

# loading all files. 
movie = load_files(moviedir, shuffle=True)

len(movie.data)

# target names ("classes") are automatically generated from subfolder names
movie.target_names

# First file seems to be about a Schwarzenegger movie. 
movie.data[0][:500]

# first file is in "neg" folder
movie.filenames[0]

# first file is a negative review and is mapped to 0 index 'neg' in target_names
movie.target[0]