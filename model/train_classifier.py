# import libraries
import re
import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def load_data(db_path):
    '''
    input:
        db_path: File path to sql database
    output:
        X: Training messages
        y: Training target
        category_names: Labels
    '''
    engine = create_engine('sqlite:///'+ db_path)
    df = pd.read_sql_table('TweetCat', engine)
    X = df.message.values
    y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    return X, y, category_names

def tokenize(text):
    '''
    input:
        text: Message data for tokenization.
    output:
        clean_tokens: Result list after normalization and tokenization.
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        if token not in stopwords.words("english"):
            clean_token = lemmatizer.lemmatize(token).lower().strip()
            clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    '''
    input:
        None
    output:
        cv: GridSearch model result.
    '''
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize, min_df=.0025, max_df=0.25, 
                                  ngram_range=(1,2))),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state = 0))))
    ])
    parameters = {
                'tfidf__smooth_idf':[True, False],
                'clf__estimator__estimator__C': [1, 2, 5]
             }
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='precision_samples', cv = 5)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    '''
    input:
        model: the fitted model
        X_test: the test dataset
        y_test the test target
        category_names: the labels
    output:
        The printed resluts
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names = category_names))
    print('--------------------------------------------')
    scores=[]
    for i in range(y_test.shape[1]):
        score=accuracy_score(y_test[:,i], y_pred[:,i])
        print('{0} accuracy : {1:.2f}'.format(target_names[i],score))
        scores.append(score)
    print('--------------------------------------------')
    print('Mean accuracy: {0:.2f}'.format(np.mean(scores)))

def save_model(model, model_filepath):
    '''
    input:
        model: the fitted and evaluated model
        model_filepath: the filepath to the model
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()