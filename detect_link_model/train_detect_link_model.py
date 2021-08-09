import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

training_file = './enwiki_detect_link_training_data.json'
testing_file = './enwiki_detect_link_testing_data.json'

xgbc = XGBClassifier(use_label_encoder=False)

def train_detect_link_model():
    train_df = pd.read_json(training_file, orient='split')
    X = train_df[['link_prob', 'frequency', 'first', 'last', 'spread']]
    y = train_df['label']
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state = 33)
    model = xgbc.fit(X_train, y_train,
                     eval_set=[(X_train, y_train), (X_val, y_val)],
                     eval_metric='logloss',
                     verbose=True)
    evals_result = xgbc.evals_result()
    return model

trained_model = train_detect_link_model()    

#save model
joblib.dump(trained_model, '../data_model/detect_link_trained_model') 

test_df = pd.read_json(testing_file, orient='split')
X_test = test_df[['link_prob', 'frequency', 'first', 'last', 'spread']]
y_test = test_df['label']

predict = trained_model.predict(X_test)
