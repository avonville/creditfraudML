
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

dataset_url= "creditcard.csv"
data= pd.read_csv(dataset_url)

#print (data.describe())

#count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
#count_classes.plot(kind = 'bar')
#plt.title("Fraud class histogram")
#plt.xlabel("Class")
#plt.ylabel("Frequency")
#plt.show()

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
print (data.head())
