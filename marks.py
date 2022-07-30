import pickle
import pandas as pd
import numpy as np


def gpa_prediction(sat_score):
    # Convert float input to numeric
    sat_score = pd.to_numeric(sat_score)

    # convert to an array data type
    x_test = np.array(sat_score)

    # reshape the array for prediction
    x_test = x_test.reshape((-1, 1))

    # load save ml model
    model = pickle.load(open("model.pkl", 'rb'))

    return model.predict(x_test)
