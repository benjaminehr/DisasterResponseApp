# Disaster Response Pipeline Project

## Project Motivation

In this project, I apply skills I learned in the Data Engineering Segment of the Udacity Data Science Nanodegree. The goal was to build a model for an API that classifies disaster messages from twitter. The data to train the model was provided by Figure Eight.

## File Description

    .
    ├── app     
    │   ├── run.py                           # Flask file that runs app
    │   └── templates   
    │       ├── go.html                      # Classification result page of web app
    │       └── master.html                  # Main page of web app    
    ├── data                   
    │   ├── disaster_categories.csv          # Dataset with categories  
    │   ├── disaster_messages.csv            # Dataset with messages
    │   └── process_data.py                  # Data cleaning
    ├── models
    │   └── train_classifier.py              # Train ML model           
    └── README.md

## Instructions: (Run run.py directly if DisasterResponse.db and claasifier.pkl already exist.)

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
