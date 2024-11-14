from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np

# ---------------------- Loading data  ----------------------

# Directory containing movie reviews
moviedir = r'C:\Users\david\Documents\GitHub\TNM108\Lab4\movie_reviews'

# Load all files
movie = load_files(moviedir, shuffle=True)
target_names = ['pos', 'neg']  # Positive and negative categories

#Split data into 80% training and 20% testing
movie_train_data, movie_test_data, movie_train_target, movie_test_target = train_test_split(movie.data, movie.target, test_size=0.2, random_state=42)

# ---------------------- pipeline ----------------------

# ---------- ALT [1] MultinomialNB ----------

movie_clf = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB()),
    ]
)

movie_clf.fit(movie_train_data, movie_train_target)
predicted = movie_clf.predict(movie_test_data)
print("multinomialBC accuracy ", np.mean(predicted == movie_test_target))

# ---------- ALT [2] SVM ----------

# movie_clf = Pipeline([
#  ('vect', CountVectorizer()),
#  ('tfidf', TfidfTransformer()),
#  ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
# ,max_iter=5, tol=None)),
# ])

# movie_clf.fit(movie_train_data, movie_train_target)
# predicted = movie_clf.predict(movie_test_data)
# print("SVM accuracy ",np.mean(predicted == movie_test_target))

# ---------------------- Grid Search ----------------------

# we create possible parameters
parameters = {
    "vect__ngram_range": [(1, 1), (1, 2)],
    "tfidf__use_idf": (True, False),
    "clf__alpha": (1, 1e-3),
}

# we gridsearch the best parameters
gs_clf = GridSearchCV(movie_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(movie_train_data, movie_train_target)

# print the best score
print("\ngrid_search_clf.best_score_:", gs_clf.best_score_, "\n")

# print the best parameters
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
print("\n")

# ---------------------- Test Reviews ----------------------

# Very short, sample movie reviews to test the model
reviews_new = [
    'This movie was excellent',
    'Absolute joy ride',
    'Steven Seagal was terrible',
    'Steven Seagal shone through.',
    'This was certainly a movie',
    'Two thumbs up',
    'I fell asleep halfway through',
    "We can't wait for the sequel!!",
    '!',
    '?',
    'I cannot recommend this highly enough',
    'instant classic.',
    'Steven Seagal was amazing. His performance was Oscar-worthy.'
]

# Use the best pipeline to transform and predict on new reviews
reviews_new_predicted = gs_clf.predict(reviews_new)

# Print out the results for each test review
for review, category in zip(reviews_new, reviews_new_predicted):
    print("%r => %s" % (review, movie.target_names[category]))


    
   