import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()

df = pd.read_csv("Dataset/data.csv")

x = df['SAT']
y = df['GPA']

# changing the dimensionality of x from 1d array tp 2d array, to fit the sklearn regression library
x_matrix = x.values.reshape(-1,1)

reg = LinearRegression()
reg.fit(x_matrix, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(reg, file)
