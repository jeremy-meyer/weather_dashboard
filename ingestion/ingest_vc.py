import pandas as pd
import numpy as np
import requests
from io import BytesIO
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, date
from dotenv import load_dotenv

load_dotenv()

YOURAPIKEY = os.getenv('YOURAPIKEY')
base_url_weather_request = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history?&aggregateHours=24&"
csv_subdir = 'raw_sources/'
location = "Salt Lake City, UT, US"
combined_file = 'daily_weather.csv'


# https://www.visualcrossing.com/resources/documentation/weather-api/weather-api-documentation/

def pull_weather_data(start_time, end_time, location):
    weather_request = f"{base_url_weather_request}startDateTime={start_time}&endDateTime={end_time}&unitGroup=us&includeAstronomy=true&contentType=csv&location={location}&key={YOURAPIKEY}"
    response = requests.get(weather_request)
    assert response.status_code == 200, "Error: Non-200 code"
    df = pd.read_csv(BytesIO(response.content), header=0)
    return df

def clean_transform_raw_csv(df_raw):
    combined_df = (
    df_raw
        .sort_values(['Address', 'Date time'])
        .reset_index(drop=True)
    )
    combined_df['date'] = pd.to_datetime(combined_df['Date time']).dt.date
    combined_df.drop(columns=['Date time'], inplace=True)
    combined_df.sort_values(by='date', inplace=True)

    # Rename columns
    combined_df.rename(columns={
        'Minimum Temperature': 'min_temp',
        'Maximum Temperature': 'max_temp',
        'Temperature': 'avg_temp',
        'Wind Chill': 'wind_chill',
        'ASea Level Pressure': 'pressure',
        'Precipitation': 'precipitation',
        'Snow Depth': 'snow_depth',
        'Snowfall': 'snowfall',
        'Wind Speed': 'wind_speed',
    }, inplace=True)
    return combined_df


def append_to_csv(path, df, max_append=1000):
  days_to_update = df.shape[0]
  schema_match = all(
      pd.read_csv(path).columns == df.columns
  )
  too_many_to_update = days_to_update > max_append
  if schema_match and not too_many_to_update:
      # Append new data to combined file
      # append mode, prevents column overwrite and data index overwrite
      df.to_csv(path, mode='a', header=False, index=False)
      print('Appended new data!')
  elif not schema_match:
      raise Exception("Error: Schema mismatch between new data and existing combined file")
  elif too_many_to_update:
      raise Exception("Error: Too many days to update at once, please double check work")    


# Data Update ---------------------
# Calculate new data to pull
max_date = pd.read_csv(csv_subdir + combined_file, index_col=0)['date'].max()
max_date = pd.to_datetime(max_date) + timedelta(days=1)
days_to_update = (date.today() - max_date.date()).days - 1
start_time = max_date.strftime('%Y-%m-%dT00:00:00')
end_time = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT00:00:00')
print(f"Pulling data from {start_time} to {end_time}")
df = pull_weather_data(start_time, end_time, location)
df_cleaned = clean_transform_raw_csv(df)
append_to_csv(csv_subdir + combined_file, df_cleaned)


# # AD HOC: Pull Data for multiple years + save to separate files-------------
# years = range(1970,1972) # Pull from API
# dfs = dict()
# for year in years:
#     start_time = f"{year}-01-01T00:00:00"
#     end_time = f"{year}-12-31T00:00:00"
#     dfs[year] = clean_transform_raw_csv(pull_weather_data(start_time, end_time, location))
#     # df.to_csv(f'{csv_subdir}/slc_{year}.csv', index=False)
#     print(f"Finished {year}")

# # combine all years + append
# append_to_csv(csv_subdir + combined_file, pd.concat(dfs.values(), ignore_index=True))


# # Write combined file (only need to do once)
# filepaths = [csv_subdir+'/'+f for f in os.listdir(csv_subdir) if (f.endswith('.csv') and f.startswith('slc'))]
# df_raw = pd.concat(map(lambda x: pd.read_csv(x, index_col=False), filepaths))
# combined_df = clean_transform_raw_csv(df_raw)
# combined_df.to_csv(csv_subdir+combined_file, index=False)
