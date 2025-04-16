import pandas as pd

df_humidity = pd.read_csv('/Users/kaushal/Downloads/Vistula/Sem2/Big data/Project/VS/Climate_analysis/historical-hourly-weather-dataset/humidity.csv')
df_pressure = pd.read_csv('/Users/kaushal/Downloads/Vistula/Sem2/Big data/Project/VS/Climate_analysis/historical-hourly-weather-dataset/pressure.csv')
df_temperature = pd.read_csv('/Users/kaushal/Downloads/Vistula/Sem2/Big data/Project/VS/Climate_analysis/historical-hourly-weather-dataset/temperature.csv')
df_weather = pd.read_csv('/Users/kaushal/Downloads/Vistula/Sem2/Big data/Project/VS/Climate_analysis/historical-hourly-weather-dataset/weather_description.csv')
df_wind_direction = pd.read_csv('/Users/kaushal/Downloads/Vistula/Sem2/Big data/Project/VS/Climate_analysis/historical-hourly-weather-dataset/wind_direction.csv')
df_wind_speed = pd.read_csv('/Users/kaushal/Downloads/Vistula/Sem2/Big data/Project/VS/Climate_analysis/historical-hourly-weather-dataset/wind_speed.csv')

print(df_humidity)
print(df_pressure)
print(df_temperature)
print(df_weather)
print(df_wind_direction)
print(df_wind_speed)

df_humidity_long = df_humidity.melt(id_vars=["datetime"], var_name="city", value_name="humidity")
df_pressure_long = df_pressure.melt(id_vars=["datetime"], var_name="city", value_name="pressure")
df_temperature_long = df_temperature.melt(id_vars=["datetime"], var_name="city", value_name="temperature")
df_weather_long = df_weather.melt(id_vars=["datetime"], var_name="city", value_name="weather")
df_wind_direction_long = df_wind_direction.melt(id_vars=["datetime"], var_name="city", value_name="wind_direction")
df_wind_speed_long = df_wind_speed.melt(id_vars=["datetime"], var_name="city", value_name="wind_speed")



dfs = [df_weather_long, df_temperature_long, df_humidity_long, df_pressure_long, df_wind_direction_long, df_wind_speed_long]

from functools import reduce
df_final = reduce(lambda left, right: pd.merge(left, right, on=["datetime", "city"], how="outer"), dfs)
print(df_final.head())
print(df_final.shape)


df_final.isnull().sum()

df_final.dropna(inplace=True)
df_final.isnull().sum()
df_final.shape

df_final.dtypes

df_final.head()

print(df_final.describe())
print(df_final.info())

import matplotlib.pyplot as plt

df_final["datetime"] = pd.to_datetime(df_final["datetime"])
df_city = df_final[df_final["city"] == "Boston"]

plt.figure(figsize=(12, 6))
plt.plot(df_city["datetime"], df_city["temperature"], label="Temperature", color="red")
plt.xlabel("Date")
plt.ylabel("Temperature (Kelvin)")
plt.title("Temperature Trends in Boston")
plt.legend()
plt.show()


import seaborn as sns

plt.figure(figsize=(8,6))
sns.scatterplot(x=df_final["temperature"], y=df_final["humidity"])
plt.xlabel("Temperature (K)")
plt.ylabel("Humidity (%)")
plt.title("Humidity vs Temperature")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

numeric_columns = ['temperature', 'humidity', 'pressure', 'wind_speed']

plt.figure(figsize=(14, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df_final[col], color='skyblue')
    plt.title(f'Box Plot of {col}')
    plt.xlabel(col)
    plt.tight_layout()

plt.suptitle('Box Plots for Outlier Detection', fontsize=16, y=1.02)
plt.show()


df_final.groupby('city')['temperature'].mean().sort_values().plot(kind='barh', title='Avg Temperature by City')
plt.xlabel('Temperature (K)')
plt.show()

df_final['weather'].value_counts().plot(kind='bar', title='Weather Condition Frequency')
plt.ylabel('Count')
plt.show()


sns.histplot(df_final['wind_speed'], bins=10, kde=True)
plt.title('Distribution of Wind Speeds')
plt.xlabel('Wind Speed (km/h)')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df_final[["temperature", "humidity", "pressure", "wind_speed"]].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Climate Parameters")
plt.show()


df_final.set_index("datetime", inplace=True)
df_final.sort_index(inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_final["weather_encoded"] = le.fit_transform(df_final["weather"])


from sklearn.model_selection import train_test_split

X = df_final[["temperature", "humidity", "pressure", "wind_speed"]]
y = df_final["weather_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


