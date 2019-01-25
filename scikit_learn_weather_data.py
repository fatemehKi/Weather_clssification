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
y=clean_data['high_humidity_label'].copy()

#Use 9am Sensor Signals as Features to Predict Humidity at 3pm;
#the purpose is to create a model that can predict humidity at 3pm based on the weather at 9am
morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']


