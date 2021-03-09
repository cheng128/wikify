import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

training_file = 'enwiki_disambiguation_training_data.json'

def train_disambiguate_model():
    xgbc = XGBClassifier(use_label_encoder=False)
    dataframe = pd.read_json(training_file, orient='split')
    X = dataframe[['commonness', 'relatedness', 'context_quality']]
    y = dataframe['label']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 33)
    model = xgbc.fit(X_train, y_train)
    
    return model

def main():
    trained_model = train_disambiguate_model()    
    #save model
    joblib.dump(trained_model, '../data_model/disambiguate_trained_model') 

if __init__ == "__main__":
    main()