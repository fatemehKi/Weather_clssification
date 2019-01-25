import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Creating a Pandas DataFrame from the weathear CSV file
data = pd.read_csv('./daily_weather.csv')
# looking at the data frame
data

##############cleaning the data (there are some null values and also removing the number column
del data['number']

##Now let's drop null values using the *pandas dropna* function.
#check if it does have a null rows
data[data.isnull().any(axis=1)]
#storing data before delet and null
before_rows = data.shape[0]
print(before_rows)
#dropinging the null values
data = data.dropna()
after_rows = data.shape[0]
print(after_rows)

############## data is cleaned now, check the columns with high humidity and create a new column for it
clean_data = data.copy()
#multiply by one to turn boolean to digit
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm']>24.99)*1


