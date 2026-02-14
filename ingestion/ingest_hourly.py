import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# AD HOC READ ONLY

paths = [
  'data_sources/1994-2009.csv',
  'data_sources/2010-2014.csv',
  'data_sources/2015-2018.csv',
  'data_sources/2019.csv',
  'data_sources/2020-2025.csv',
]

column_map = {
  'time': 'timestamp',
  'temperature_2m (°F)': 'temp',
  'relative_humidity_2m (%)': 'humidity',
  'dew_point_2m (°F)': 'dew_point',
  'precipitation (inch)': 'precip',
  'rain (inch)': 'rain',
  'apparent_temperature (°F)': 'apparent_temp',
  'snowfall (inch)': 'snow',
  'cloud_cover (%)': 'cloud_cover',
  'wind_speed_10m (mp/h)': 'wind_speed',
  'pressure_msl (hPa)': 'pressure',
}

sun_times = (
  pd.read_csv('data_sources/sunrise_set.csv')
  .rename(columns={'time': 'day','sunset (iso8601)': 'sunset', 'sunrise (iso8601)': 'sunrise'})
)
sun_times['day'] = pd.to_datetime(sun_times['day']).dt.date

hourly_combined = pd.concat([pd.read_csv(path) for path in paths]).rename(columns=column_map)
hourly_combined['timestamp'] = pd.to_datetime(hourly_combined['timestamp'])
hourly_combined['day'] = pd.to_datetime(hourly_combined['timestamp']).dt.date
hourly_combined = hourly_combined.merge(sun_times, on='day', how='left')

hourly_combined.to_csv('data_sources/slc_hourly_weather.csv', index=False)


# Sanity Check
hourly_raw = pd.read_csv('data_sources/slc_hourly_weather.csv')
hourly_raw['timestamp'] = pd.to_datetime(hourly_raw['timestamp'])
hourly_raw['year'] = pd.to_datetime(hourly_raw['timestamp']).dt.year
hourly_raw['month'] = pd.to_datetime(hourly_raw['timestamp']).dt.strftime('%b')
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
hourly_raw['month'] = pd.Categorical(hourly_raw['month'], categories=month_order, ordered=True)

(
  hourly_raw#[hourly_raw['year'] >= 2010]
  .groupby(['year'])
  .agg(
    min_temp=('temp', 'min'),
    max_temp=('temp', 'max'),
    total_precip=('precip', 'sum'),
    total_snow=('snow', 'sum'),
    cloud_cover_avg=('cloud_cover', 'mean'),
  )
  .reset_index()
)

sns.lineplot(
  data=hourly_raw[hourly_raw['day'].between('2025-06-01', '2025-08-15')],
  x='timestamp',
  y='temp',
)

plt.xticks(rotation=45)
plt.show()

