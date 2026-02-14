import requests
import pandas as pd
import os
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
import numpy as np
from utils.shared_functions import *

load_dotenv()

# This code will pull from the noaa source
# It has more accurate precipation data, and the temperature is recorded at the SLC airport

cols_to_rename = {
  'DATE': 'date',
  'TMAX': 'tempmax',
  'TMIN': 'tempmin',
  'TAVG': 'tempavg',
  'PRCP': 'precip',
  'SNOW': 'snow',
  'SNWD': 'snow_depth',
  'STATION': 'station',
  'NAME': "location",
  'ACMH': 'avg_cloud',
  'ACSH': 'avg_cloud_daytime',
  'AWND': 'avg_wind',
  'PSUN': 'sun_perc',
  'SN02': 'min_soil_temp_10cm',
  'SN03': 'min_soil_temp_20cm',
  'SX02': 'max_soil_temp_10cm',
  'SX03': 'max_soil_temp_20cm',
  'TSUN': 'sunshine_hours',
  'WDF2': 'wind_direction',
  'WESD': 'water_snow_depth', 
}

# Determine what data needs incremental update
existing_data = pd.read_csv('raw_sources/noaa_precip.csv')

# EXTRACT DATA --------------------------------------------------
# We will consider updating nulls in the last 6 months
# return the date of the first NaN occurrence in any of the needed fields over the last 6 months
needed_fields = ['tempavg', 'tempmax', 'tempmin', 'precip', 'snow']
last_6m = existing_data[existing_data['date'] >= str(pd.to_datetime(date.today()) - pd.DateOffset(months=6))]
max_update_date = existing_data['date'].max()
if last_6m.empty:
    start_date = existing_data['date'].max()
else:
  start_date = last_6m.loc[last_6m[needed_fields].isna().any(axis=1), 'date'].min()
end_date = str(date.today() - timedelta(days=1))
print(f"Updating data from {start_date} to {end_date}")

# NOAA API configuration
BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
STATION_ID = "USW00024127"  # Salt Lake City Airport station

# Function to fetch data from NOAA API
def fetch_noaa_data(start_date, end_date, station_id, data_types):
    params = {
        "dataset": "daily-summaries",
        "stations": station_id,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": ",".join(data_types),
        "format": "json",
        "units": "standard",
        "includeAttributes": "false",
        "includeStationName": "true",
        "includeStationLocation": "true",
        "token": os.getenv("TOKEN"),
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Fetch data
api_data = fetch_noaa_data(start_date, end_date, STATION_ID, cols_to_rename.keys())


# LOAD / TRANSFORM DATA --------------------------------------------------
if api_data:
    incremental_data = pd.DataFrame(api_data)
    incremental_data['insert_timestamp'] = datetime.now()
    # basic transformations
    incremental_data["DATE"] = (pd.to_datetime(incremental_data["DATE"]).dt.date).astype(str)
    incremental_data.rename(columns=cols_to_rename, inplace=True)
    float_cols = list(existing_data.select_dtypes(include=['float64']).columns)
    for c in float_cols:
        if c in incremental_data.columns:
            incremental_data[c] = pd.to_numeric(incremental_data[c], errors='coerce')
    
    incremental_data.head()
else:
    print("No data retrieved from NOAA API.")


# INCREMENTAL UPDATE --------------------------------------------------
# Sometimes not all columns are available
cols_available = [x for x in incremental_data.columns if x in existing_data.columns]

# merge existing_data with incremental_data[cols_available] on 'date', append new rows, fill nulls in exsting rows
upserted_data = (
    existing_data
    .merge(incremental_data[cols_available], on='date', how='outer', suffixes=('', '_new'))
)

for c in existing_data.columns:
  if (c in incremental_data.columns) and (c != 'date'):
    upserted_data[c] = upserted_data[c + '_new'].combine_first(upserted_data[c])
    upserted_data.drop(columns=[c + '_new'], inplace=True)


# Ensure there aren't duplicate dates
assert (upserted_data['date'].is_unique) & all(upserted_data['date'].notna()), "Duplicate or null dates found after merge!"

# Write out raw ingested feed
upserted_data.to_csv('raw_sources/noaa_precip.csv', index=False)

# WRITE SILVER LEVEL TABLE FOR ANALYSIS -----------------------------------
precip_raw = pd.read_csv('raw_sources/noaa_precip.csv', index_col=False, parse_dates=['date'])
precip = precip_raw[['date','precip' ,'snow', 'snow_depth', 'tempmin', 'tempmax']].copy()
precip['month'] = pd.Categorical(precip['date'].dt.strftime('%b'), categories=month_order, ordered=True)
precip['date'] = pd.to_datetime(precip['date'])
precip['year'] = precip['date'].dt.year
precip['precip_day'] = (precip['precip'] > 0).astype(int)

# Define snow season as Aug - July (I know water season is Oct - Sep, but we have had snow in Sep and I would rather count that in next season)
precip['snow_season'] = np.where(precip['month'].isin(['Aug', 'Sep', 'Oct', 'Nov', 'Dec']), precip['year'], precip['year'] - 1)
precip['snow_season'] = precip['snow_season'].astype(str) + '-' + (precip['snow_season'] + 1).astype(str)

precip['water_year'] = np.where(precip['month'].isin(['Oct', 'Nov', 'Dec']), precip['year']+1, precip['year'])
precip['day_of_year'] = precip['date'].apply(gen_DoY_index)

# Problem: We don't have a clear way to separate rain from snow. We must make assumptions
# If max_temp < 29, it's all snow
# If min_temp > 37, it's all rain
# Otherwise use 8:1 ratio, since wetter snow is heavier than 10:1
SWE_ratio = 8

precip['rain'] = np.where(
    precip['tempmin'] > 37, precip['precip'], np.where(
        precip['tempmax'] < 29, 0, np.maximum(precip['precip'] - precip['snow']/SWE_ratio, 0)
    )
)
precip['snow_water_equiv'] = precip['precip'] - precip['rain']
# snow_water_equiv + rain = precip

precip.to_csv('output_sources/precip_table.csv', index=False)

# Ad HOC READ
# noaa = pd.read_csv('weather/data-sources/noaa_raw.csv')
# noaa.drop(columns=['SN52', 'SN53', 'SX52', 'SX53'], inplace=True)
# noaa.rename(columns=cols_to_rename, inplace=True)

# noaa.to_csv("weather/slc_daily_weather/noaa_slc_airport.csv", index=False)

# Quality Test
# (
#   noaa[['date', 'precip', 'snow' ,'tempmax', 'tempmin']]
#   [noaa['year']>=2023]
#   .sort_values(by='snow', ascending=False)
# ).head(10)

# (
#   noaa[noaa['year']==1972]
#   .groupby('month')
#   .agg({
#       'tempmax': 'max',
#       'tempmin': 'min',
#       'tempavg': 'mean',
#       'precip': 'sum',
#       'snow': 'sum',
#       'snow_depth': 'max'
#   })
# )

# nan_percent = noaa.isna().mean() * 100
# print(nan_percent)
