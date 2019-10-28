import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(tweet_path, cat_path):
    '''
    input:
        tweet_path: The path of messages dataset.
        cat_path: The path of categories dataset.
    output:
        df: The merged dataset
    '''
    messages = pd.read_csv(tweet_path)
    categories = pd.read_csv(cat_path)
    df = pd.merge(messages, categories, how="outer", on="id")
    return df

def clean_data(df):
    '''
    input:
        df: The merged dataset in previous step.
    output:
        df: Dataset after cleaning.
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(subset = 'id', inplace = True)
    return df

def save_data(df, db_path):
    engine = create_engine('sqlite:///' + db_path)
    df.to_sql('TweetCat', engine, index=False)

def main():
    if len(sys.argv) == 4:

        tweet_path, cat_path, db_path = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(tweet_path, cat_path))
        df = load_data(tweet_path, cat_path)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(db_path))
        save_data(df, db_path)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
