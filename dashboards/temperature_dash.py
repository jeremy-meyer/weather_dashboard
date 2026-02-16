from dash import Dash, html, dash_table, dcc, callback, Output, Input, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.graph_objects as go
from datetime import date, datetime
from plotly_calplot import calplot
import statsmodels.api as sm

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.getcwd())
from utils.shared_functions import *



pio.templates.default = "plotly_dark"
pio.renderers.default = "browser"

# SHARED CONFIGS ---------------------
temp_colors = [
    '#3b4cc0', '#528ecb', "#8399a5", '#b5d5e6', 
    '#e0e0e0',"#e0e0e0", "#e0e0e0", 
    '#f6bfa6', '#ea7b60', '#c63c36', '#962d20',
]
precip_colors = [
  '#865513','#D2B469', '#F4E8C4', '#F5F5F5', 
  '#CBE9E5', '#69B3AC', '#20645E'
]
RECORDS_DEFAULT=15

# Code to generate dashboard style tables
def generic_data_table(df, id, page_size=10, clean_table=False, metric_value=None, decimal_places=2):
  if clean_table:
    df = (
      df
      [['rank','date', 'metric_value']]
      .rename({'metric_value': metric_value}, axis=1)
    )
    df[metric_value] = round(df[metric_value], decimal_places)
  current_year = date.today().year
  date_condition = f"({{date}} contains '{current_year}') || ({{date}} contains '{current_year-1}') || ({{date}} contains '{current_year-2}')"
  
  return (
    html.Div([
      dash_table.DataTable(
          id=id,
          columns=[{"name": i, "id": i} for i in df.columns],
          data=df.to_dict('records'),
          page_action='native',  # Enable native front-end paging
          page_size=page_size,          # Number of rows per page
          style_table={'overflowX': 'auto'},  # Ensure horizontal scrolling if needed
          style_header={
              'backgroundColor': '#333333',  # Dark background for the header
              'color': 'white',             # White text for the header
              'fontWeight': 'bold',         # Bold text for the header
          },
          style_data={
              'backgroundColor': '#222222',  # Darker background for the data rows
              'color': 'white',             # White text for the data rows
              'border': '1px solid #444'    # Gray border for the data rows
          },
          style_data_conditional=[
          {
              'if': {'row_index': 'odd'},  # Alternate row styling
              'backgroundColor': '#2a2a2a'  # Slightly lighter background for odd rows
          },
          {
            'if': {
                'filter_query': date_condition,
            },
            'backgroundColor': "#4E4D00",
            'fontWeight': 'bold',
          },
           ],
          style_cell_conditional=[
              {'if': {'column_id': 'rank'}, 'width': '10%'},  # Set width for 'rank' column
              {'if': {'column_id': 'date'}, 'width': '25%'},  # Set width for 'date' column
          ],
        )
    ])
  )

# Initialize the app
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.themes.DARKLY, 'dark_dropdown.css']

app = Dash(__name__, external_stylesheets=external_stylesheets, assets_folder='./dashboards/assets/')

# INPUT DATA
N_years = 30 # For normals

temps = pd.read_csv("raw_sources/daily_weather.csv", parse_dates=['date'])#.query("date >= '1995-01-01'")
normals = pd.read_csv("output_sources/temp_daily_normals.csv")
sunrise_sunset = pd.read_csv("raw_sources/sunrise_set_slc.csv", parse_dates=['time'])

temps['day_of_year'] = temps['date'].apply(gen_DoY_index)
temps['range'] = temps['max_temp'] - temps['min_temp']
temps['year'] = temps['date'].dt.year
temps_minmax = (temps['year'].min(),temps['year'].max())

min_temp_date = temps['date'].min()
max_temp_date = temps['date'].max()
min_snow_year = (min_temp_date.year+1 if min_temp_date.month>=8 else min_temp_date.year)
max_snow_year = (max_temp_date.year-1 if max_temp_date.month<7 else max_temp_date.year)

temps =   temps.rename({
    'Dew Point': 'dew_point',
    'Cloud Cover': 'cloud_cover',
    'wind_chill': 'wind_chill',
    'Sea Level Pressure': 'pressure',
    'Heat Index': 'heat_index',
  }, axis=1
)

# add a column rank to temp that ranks how hot each day was relative to all other years
temps['diurnal_temp_range'] = temps['max_temp'] - temps['min_temp']
temps['high_rank'] = temps.groupby('day_of_year')['max_temp'].rank(ascending=False, method='min')
temps['low_rank'] = temps.groupby('day_of_year')['min_temp'].rank(ascending=True, method='min')
temps['month'] = pd.Categorical(temps['date'].dt.strftime('%b'), categories=month_order, ordered=True)
temps['snow_season'] = np.where(temps['month'].isin(['Aug', 'Sep', 'Oct', 'Nov', 'Dec']), temps['year'], temps['year'] - 1)
temps['snow_season'] = temps['snow_season'].astype(str) + '-' + (temps['snow_season'] + 1).astype(str)
temps['snow_day_of_year'] = temps.apply(lambda x: gen_DoY_index(x['date'], 'snow_season'), axis=1)

temps_full = temps.merge(normals.drop(columns=['month'], axis=1), on='day_of_year', how='left', suffixes=('', '_y'))

sunrise_sunset_clean = (
  sunrise_sunset[sunrise_sunset['time'].dt.year == 2024]
  .rename({'time': 'date', 'sunset (iso8601)': 'sunset', 'sunrise (iso8601)': 'sunrise'}, axis=1)
)
sunrise_sunset_clean['sunset_hr'] = sunrise_sunset_clean['sunset'].apply(str_to_decimal_hr)
sunrise_sunset_clean['sunrise_hr'] = sunrise_sunset_clean['sunrise'].apply(str_to_decimal_hr)
sunrise_sunset_clean['day_of_year'] = sunrise_sunset_clean['date'].apply(gen_DoY_index)

# Monthly Heatmap Data
max_date = temps_full['date'].max()
if max_date != max_date + pd.offsets.MonthEnd(0):
  most_recent_month = temps_full['date'].max().replace(day=1) - pd.Timedelta(days=1)
else:
  most_recent_month = max_date

today = date.today()
normal_n_year = 30
years_for_normals = range(date.today().year - normal_n_year, date.today().year)

all_dates = pd.date_range(start="2024-01-01", end="2024-12-31")  # Leap year to include Feb 29
formatted_dates = [x.strftime("%b %d") for x in all_dates]  # Format as "MMMM D"

# Compute Monthly Normals with 30 years; records use all data
month_avgs = (
  temps_full
  .assign(
    for_avg_high = lambda x: x['max_temp'].where(x['year'].isin(years_for_normals), None),
    for_avg_low = lambda x: x['min_temp'].where(x['year'].isin(years_for_normals), None),
    for_avg_temp = lambda x: x['avg_temp'].where(x['year'].isin(years_for_normals), None),
  )
  .groupby(['month'])
  .agg(
    # round these to 1 decimal place
    normal_high=('for_avg_high', lambda x: round(x.mean(), 1)),
    normal_low=('for_avg_low', lambda x: round(x.mean(), 1)),
    normal_temp=('for_avg_temp', lambda x: round(x.mean(), 1)),
    max_temp_p90=('max_temp', lambda x: round(x.quantile(0.9), 1)),
    min_temp_p10=('min_temp', lambda x: round(x.quantile(0.1), 1)),
    record_high=('max_temp', 'max'),
    record_low=('min_temp', 'min'),
    record_high_year=('max_temp', lambda x: temps_full.loc[x.idxmax(), 'date'].date()),
    record_low_year=('min_temp', lambda x: temps_full.loc[x.idxmin(), 'date'].date()),
  )
  .reset_index()
)

monthly_map = (
  pd.melt(
    temps_full[temps_full['date'] <= most_recent_month]\
      [['year', 'month', 'max_temp', 'min_temp', 'avg_temp']],
    id_vars=['year', 'month'],
    value_vars=['max_temp', 'min_temp', 'avg_temp'],
    var_name='metric',
    value_name='temp',
  )
  .groupby(['year', 'month', 'metric'])
  .agg(mean=("temp", "mean"))
  .reset_index()
)

monthly_map_year = (
  monthly_map
  .groupby(['year', 'metric'])
  .agg(
    mean=('mean', 'mean'),
    count=('mean', 'count'),
  )
  .reset_index()
)

monthly_map_year = monthly_map_year[monthly_map_year['count']==12]
monthly_map_year = monthly_map_year.drop(['count'], axis=1)
monthly_map_year['month'] = 'Year'
monthly_map = pd.concat([monthly_map, monthly_map_year])
monthly_map['month'] = pd.Categorical(monthly_map['month'], categories=month_order+["Year"], ordered=True)

monthly_map['rank'] = (
    monthly_map.groupby(['month', 'metric'])['mean']
    .rank(method='min', ascending=True)
)

# Hourly Data
hourly_temp = pd.read_csv("raw_sources/hourly_weather.csv", parse_dates=['timestamp'])
hourly_temp['month'] = pd.Categorical(hourly_temp['timestamp'].dt.strftime('%b'), categories=month_order, ordered=True)
hourly_temp['hour'] = hourly_temp['timestamp'].dt.hour
hourly_temp['year'] = hourly_temp['timestamp'].dt.year
hourly_temp['day_of_year']  = hourly_temp['timestamp'].apply(gen_DoY_index)
hourly_temp['precip_chance'] = (hourly_temp['precip'] > 0).astype('int')

hourly_metrics_pretty= {
  'temp': 'Temperature (°F)',
  'humidity': 'Humidity (%)', 
  'wind_speed': 'Wind Speed (mph)',
  'dew_point': 'Dew Point (°F)',
  'precip_chance': 'Precipitation Chance (%)',
  'precip': 'Precipitation (in)',
  'rain': 'Rain (in)',
  'snow': 'Snow (in)',
  'cloud_cover': 'Cloud Cover (%)',
  'pressure': 'Pressure (mb)',
}

season_map = {
  'winter': ['Dec', 'Jan', 'Feb'],
  'spring': ['Mar', 'Apr', 'May'],
  'summer': ['Jun', 'Jul', 'Aug'],
  'fall': ['Sep', 'Oct', 'Nov']
}


hourly_all_metrics = (
  hourly_temp
  .melt(id_vars=['timestamp', 'month', 'hour', 'year', 'day_of_year'], 
         value_vars=hourly_metrics_pretty.keys(),
         var_name='metric_name', value_name='metric_value'
  )
)

hourly_heatmap = (
  hourly_all_metrics
  .groupby(['day_of_year' , 'hour', 'metric_name'])
  .agg(
    metric_value=('metric_value', 'mean'),
  )
  .reset_index()
)
hourly_heatmap['DoY_label'] = hourly_heatmap['day_of_year'].apply(dayofyear_to_month_day)

hourly_all_year = (
  hourly_all_metrics
  .groupby(['year','month' , 'hour', 'metric_name'])
  .agg(
    metric_value=('metric_value', 'mean'),
  )
  .reset_index()
)

hourly_all_mean = (
  hourly_all_year
  .groupby(['month' , 'hour', 'metric_name'])
  .agg(
    metric_value=('metric_value', 'mean'),
  )
  .reset_index()
)
# Season column
hourly_all_mean['season'] = hourly_all_mean['month'].map({v: k for k, vs in season_map.items() for v in vs})
hourly_all_mean_season = (
  hourly_all_mean
  .groupby(['season' , 'hour', 'metric_name'])
  .agg(
    metric_value=('metric_value', 'mean'),
  )
  .reset_index()
)

# Relative to month avg across hour
hourly_all_mean_season['avg_deviation'] = hourly_all_mean_season['metric_value'] - (
  hourly_all_mean_season.groupby(['season', 'metric_name'])['metric_value'].transform('mean')
)

# Precip Data
precip = pd.read_csv('output_sources/precip_table.csv', index_col=False, parse_dates=['date'])
current_year = precip['year'].max()
max_water_year = precip['water_year'].max()
max_winter_year = precip['snow_season'].max()
max_precip_date = precip['date'].max()
precip['snow_day_of_year'] = precip.apply(lambda x: gen_DoY_index(x['date'], 'snow_season'), axis=1)
precip['snow'].fillna(0, inplace=True)
precip['rain'].fillna(0, inplace=True)

ytd_normals = pd.read_csv('output_sources/precip_ytd_normals.csv')
precip['month'] = pd.Categorical(precip['date'].dt.strftime('%b'), categories=month_order, ordered=True)
precip_data_unpivot = pd.concat([
  precip\
    .assign(
      year_type='calendar_year', 
      current_year=str(current_year), 
      year_for_dash=precip['year'],
      norm_range=precip['year'].between(current_year - N_years, current_year),
    ),
  precip\
    .assign(
      year_type='water_year', 
      current_year=str(max_water_year), 
      year_for_dash=precip['water_year'],
      norm_range=precip['water_year'].between(max_water_year - N_years, max_water_year),
    ),
  precip\
    .assign(
      year_type='snow_season', 
      current_year=max_winter_year, 
      year_for_dash=precip['snow_season'],
      norm_range=precip['snow_season'].between(offset_season(max_winter_year, - N_years + 1), offset_season(max_winter_year, -1)),
    ),
])

precip_data_unpivot['year_for_dash'] = precip_data_unpivot['year_for_dash'].astype(str)
precip_data_unpivot['current_year'] = precip_data_unpivot['current_year'].astype(str)
precip_data_unpivot['day_of_year_dash'] = precip_data_unpivot.apply(lambda x: gen_DoY_index(x['date'], x['year_type']), axis=1)

precip_data_unpivot = (
  precip_data_unpivot
  .melt(
    ['date', 'month', 'year_for_dash', 'day_of_year', 'day_of_year_dash', 'year_type', 'current_year', 'norm_range'], 
    value_vars=['precip', 'snow', 'rain', 'precip_day'], 
    var_name='metric_name', 
    value_name='metric_value')
)

monthly_norm = (
  pd.read_csv('output_sources/precip_monthly_normals.csv')
  .rename({'norm_precip': 'precip', 'norm_snow': 'snow', 'norm_rain': 'rain', 'norm_precip_perc': 'precip_perc'}, axis=1)
  .melt(['month'], ['precip', 'snow', 'rain', 'precip_perc'], 'metric_name' ,'normal')
)

monthly_totals = (
  precip_data_unpivot
  .query(f"year_type == 'calendar_year'")
  .groupby(['year_type', 'current_year', 'metric_name', 'month', 'year_for_dash'])
  .agg(
    observed=('metric_value', 'sum')
  )
  .reset_index()
  .merge(monthly_norm, on=['month', 'metric_name'], how='inner')
  .melt(['year_type', 'current_year', 'metric_name', 'month', 'year_for_dash'], ['observed', 'normal'], 'type', 'metric_value')
)

monthly_totals['month'] = pd.Categorical(monthly_totals['month'], categories=month_order, ordered=True)

precip_data_for_norm = (
  precip_data_unpivot.query("(norm_range) | (current_year == year_for_dash)")
)

precip_data_for_norm = precip_data_for_norm.assign(
  day_of_month=precip_data_for_norm['date'].dt.day, 
)

current_precip_month = max_precip_date.strftime("%b")

mtd = (
  precip_data_for_norm.query("(year_type == 'calendar_year')")\
    [['date', 'month', 'year_for_dash', 'current_year', 'metric_name', 'metric_value', 'day_of_month']]
  .fillna({'metric_value': 0})
  .sort_values(by=['metric_name', 'year_for_dash' ,'date'])
  .groupby(['metric_name', 'year_for_dash', 'month'])
  .apply(lambda x: x.assign(month_to_date_precip=x['metric_value'].cumsum()))
  .reset_index(drop=True)
  # .sort_values(by=['year_type', 'metric_name', 'calendar_year', 'date'])
)
mtd_avg = pd.read_csv('output_sources/precip_mtd_normals.csv')

ytd = (
  precip_data_for_norm
  .fillna({"metric_value": 0})
  .sort_values(by=['year_type', 'metric_name', 'year_for_dash', 'date'])
  .groupby(['year_type', 'metric_name', 'year_for_dash'])
  .apply(lambda x: x.assign(year_to_date_precip=x['metric_value'].cumsum()))
  .reset_index(drop=True)
  .sort_values(by=['year_type', 'metric_name', 'year_for_dash', 'day_of_year_dash'])
)
ytd['dashboard_date'] = ytd['date'].dt.strftime('%b %d')
ytd['year'] = ytd['date'].dt.year

# Precip rank
max_date = precip['date'].max()
if max_date != max_date + pd.offsets.MonthEnd(0):
  most_recent_month = max_date.replace(day=1) - pd.Timedelta(days=1)
else:
  most_recent_month = max_date
# Create monthly_totals

precip_data_unpivot['current_year'] = precip_data_unpivot['current_year'].astype(str)
precip_data_unpivot['year_for_dash'] = precip_data_unpivot['year_for_dash'].astype(str)

precip_rank_month = (
    precip_data_unpivot
    .query(f"date <= '{most_recent_month}'")
    .groupby(['year_type', 'metric_name', 'year_for_dash', 'month'], observed=True)
    .agg(
        total=('metric_value', 'sum'),
    )
    .reset_index()
)

precip_rank_year = (
  precip_rank_month
  .groupby(['year_type','metric_name', 'year_for_dash'])
  .agg(
    total=('total', 'sum'),
    count=('total', 'count'),
  )
  .reset_index()
  .query("count == 12")
  .assign(month='Year')
  .drop(['count'], axis=1)
)

precip_rank = pd.concat([precip_rank_month, precip_rank_year])
precip_rank['month'] = pd.Categorical(precip_rank['month'], categories=month_order+["Year"], ordered=True)
precip_rank['rank'] = (
    precip_rank
    .groupby(['year_type', 'metric_name', 'month'])['total']
    .rank(method="min", ascending=True)
    .astype(int)
)

# Records
# Hottest/coldest Day (all time)
# Hottest/coldest Day (month)
# Hottest/coldest month
# Hottest/coldest year

# First freeze, last freeze, 100 degree days
# Coldest Maximum
# Warmest Minimum
def compute_temp_records(records_top_N=RECORDS_DEFAULT):
  record_column_order = ['rank','date', 'metric_name', 'metric_value', 'month', 'record']
  daily_temps = temps_full[['min_temp', 'max_temp', 'avg_temp', 'date', 'month', 'year', 'snow_season', 'snow_day_of_year']].copy()
  daily_temps['deg_100'] = 1*(daily_temps['max_temp'] >= 100)
  daily_temps['date'] = daily_temps['date'].dt.date
  daily_temps_unpivot = pd.concat([
    daily_temps.assign(metric_name='max_temp', metric_value=lambda x: x.max_temp),
    daily_temps.assign(metric_name='min_temp', metric_value=lambda x: x.min_temp),
  ])

  daily_temps_unpivot = daily_temps_unpivot.assign(
    warm_rank=lambda x: x.groupby(['metric_name','month'])['metric_value'].rank(ascending=False, method='min').astype(int),
    warm_overall=lambda x: x.groupby(['metric_name'])['metric_value'].rank(ascending=False, method='min').astype(int),
    cold_rank=lambda x: x.groupby(['metric_name','month'])['metric_value'].rank(ascending=True, method='min').astype(int),
    cold_overall=lambda x: x.groupby(['metric_name'])['metric_value'].rank(ascending=True, method='min').astype(int),
  )

  over_100 = (
    daily_temps
    .groupby(['year'])
    .agg({'deg_100': 'sum'})
    .reset_index()
    .assign(rank=lambda x: x['deg_100'].rank(ascending=False, method='min').astype(int))
    .query(f"rank <= {records_top_N}")
    .assign(metric_name='100 Degree Days', month='Year', record='100_degree_days')
    .rename({"year": 'date', "deg_100": "metric_value"}, axis=1)
    [record_column_order]
  )

  day_records = pd.concat([
    (
      daily_temps_unpivot.query(f"warm_rank <= {records_top_N}")
      .query("metric_name == 'max_temp'")
      .rename({'warm_rank': 'rank'}, axis=1)
      .assign(record=lambda x: 'hottest_day (' + x.month.astype(str) + ')')
      [record_column_order]
    ),
    (
      daily_temps_unpivot.query(f"warm_overall <= {records_top_N}")
      .query("metric_name == 'max_temp'")
      .rename({'warm_overall': 'rank'}, axis=1)
      .assign(record=lambda x: 'hottest_day')
      [record_column_order]
    ),
    (
      daily_temps_unpivot.query(f"cold_rank <= {records_top_N}")
      .query("metric_name == 'min_temp'")
      .rename({'cold_rank': 'rank'}, axis=1)
      .assign(record=lambda x: 'coldest_day (' + x.month.astype(str) + ')')
      [record_column_order]
    ),
    (
      daily_temps_unpivot.query(f"cold_overall <= {records_top_N}")
      .query("metric_name == 'min_temp'")
      .rename({'cold_overall': 'rank'}, axis=1)
      .assign(record=lambda x: 'coldest_day')
      [record_column_order]
    ),
    (
      daily_temps_unpivot.query(f"cold_overall <= {records_top_N}")
      .query("metric_name == 'max_temp'")
      .rename({'cold_overall': 'rank'}, axis=1)
      .assign(record=lambda x: 'coldest_maximum')
      [record_column_order]
    ),
    (
      daily_temps_unpivot.query(f"warm_overall <= {records_top_N}")
      .query("metric_name == 'min_temp'")
      .rename({'warm_overall': 'rank'}, axis=1)
      .assign(record=lambda x: 'hottest_minimum')
      [record_column_order]
    ),
  ])

  first_freezes = (
    daily_temps
    .query("min_temp<= 32.0")
    .assign(freeze_num=lambda x: x.groupby('snow_season')['snow_day_of_year'].rank(ascending=True, method='min'))
    .query('freeze_num == 1')
    .query(f"snow_season >= '{min_snow_year}'")
    .assign(
      rank=lambda x: x['snow_day_of_year'].rank(ascending=True, method='min').astype(int),
      metric_name='Freeze', month='Year', record='first_freeze'
    )
    .rename({"min_temp": "metric_value"}, axis=1)
    .query(f"rank <= {records_top_N}")
    [record_column_order]
  )

  last_freezes = (
    daily_temps
    .query("min_temp<= 32.0")
    .assign(freeze_num=lambda x: x.groupby('snow_season')['snow_day_of_year'].rank(ascending=False, method='min'))
    .query('freeze_num == 1')
    .query(f"snow_season <= '{max_snow_year}'")
    .assign(
      rank=lambda x: x['snow_day_of_year'].rank(ascending=False, method='min').astype(int),
      metric_name='Freeze', month='Year', record='last_freeze'
    )
    .rename({"min_temp": "metric_value"}, axis=1)
    .query(f"rank <= {records_top_N}")
    [record_column_order]
  )

  return pd.concat([day_records, over_100, first_freezes, last_freezes]).sort_values(['record', 'rank'])

def compute_daily_precip_records(precip, records_top_N=RECORDS_DEFAULT):
  cols_for_wet_days = ['date', 'snow', 'rain', 'month', 'year', 'precip_day', 'snow_season']
  record_column_order = ['rank','date', 'metric_name', 'metric_value', 'month', 'record']
  wet_days = pd.concat([
    precip[cols_for_wet_days][precip['rain'] > 0].assign(metric_name='rain', metric_value=lambda x: x.rain),
    precip[cols_for_wet_days][precip['snow'] > 0].assign(metric_name='snow', metric_value=lambda x: x.snow),
  ])
  wet_days['day_of_year'] = wet_days.apply(lambda row: gen_DoY_index(row['date'], 'snow_season'), axis=1)
  min_year = wet_days.agg({'year': 'min'}).iloc[0]

  wet_days['rank'] = (
      wet_days.groupby(['metric_name','month'])['metric_value']
      .rank(ascending=False, method='min')
      .astype(int)
  )

  wet_days['overall_rank'] = (
      wet_days.groupby(['metric_name'])['metric_value']
      .rank(ascending=False, method='min')
      .astype(int)
  )
  within_month_records = (
    wet_days
    .query(f"rank <= {records_top_N}")
    .assign(record=lambda x: x.metric_name + 'iest day (' + x.month.astype(str) + ')')
    [record_column_order]
  )
  across_month_records = (
    wet_days.drop(['rank'], axis=1)
    .rename({'overall_rank': 'rank'}, axis=1)
    .query(f"rank <= {records_top_N}")
    .assign(month='ALL')
    .assign(record=lambda x: x.metric_name + "iest day")
    [record_column_order]
  )

  earliest_snow = (
    wet_days
    .query("(metric_name == 'snow')")
    .sort_values('day_of_year')
    .assign(rank=lambda x: x['day_of_year'].rank(ascending=True, method='min').astype(int))
    .head(records_top_N)
    .assign(month='ALL')
    .assign(record='earliest_snow')
     [record_column_order]
  )

  latest_snow = (
    wet_days
    .query(f"(metric_name == 'snow') & (snow_season >= '{min_year}')")
    .sort_values('day_of_year', ascending=False)
    .assign(rank=lambda x: x['day_of_year'].rank(ascending=False, method='min').astype(int))
    .head(records_top_N)
    .assign(month='ALL')
    .assign(record='latest_snow')
    [record_column_order]
  )

  precip_days = (
    precip.query("precip_day == 1")[['date']]
  )
  precip_days['metric_value'] = precip_days['date'].diff().dt.days - 1
  precip_days['rank'] = (
      precip_days['metric_value']
      .rank(ascending=False, method='min')
  )
  drought_rank = (
    precip_days
    .query(f"(metric_value > 0) & (rank <= {records_top_N})")
    .assign(month='ALL', metric_name='precip', record='consecutive_days_dry')
    [record_column_order]
  )
  combined = pd.concat([
    within_month_records,
    across_month_records,
    drought_rank,
    latest_snow,
    earliest_snow,
  ]).sort_values(['record', 'metric_name' ,'rank'])

  combined['date'] = combined['date'].dt.date

  return combined


def compute_monthly_precip_records(records_top_N=RECORDS_DEFAULT):
  record_order = ['rank','date', 'metric_value' ,'record']
  snow_year = (
    precip_rank
    .query("year_type == 'snow_season'")
    .query("month == 'Year'")
    .query("metric_name == 'snow'")
    .assign(rank_wettest=lambda x: x.total.rank(method="min", ascending=False).astype(int))
  )

  rain_year = (
    precip_rank
    .drop(['rank'], axis=1)
    .query("year_type == 'calendar_year'")
    .query("month == 'Year'")
    .query("metric_name == 'rain'")
    .assign(rank_wettest=lambda x: x.total.rank(method="min", ascending=False).astype(int))
    .query(f"rank_wettest <= {records_top_N}")
    .assign(record='most_rain_year')
    .rename({'total': 'metric_value', 'year_for_dash': 'date', 'rank_wettest': 'rank'}, axis=1)
    [record_order]
  )


  month_precip_records = (
    precip_rank
    .query("year_type == 'calendar_year'")
    .query("metric_name in ('rain', 'snow', 'precip')")
    .query("month != 'Year'")
    .drop(['rank'], axis=1)
    .assign(
      rank_wettest=lambda x: x.groupby(['metric_name'])['total'].rank(method="min", ascending=False).astype(int),
      rank_driest=lambda x: x.groupby(['metric_name'])['total'].rank(method="min", ascending=True).astype(int),
    )
  )
  month_precip_records['month_year'] = month_precip_records['month'].astype(str)+ ' ' + month_precip_records['year_for_dash'].astype(str)


  return pd.concat([
    (
      month_precip_records
      .query(f"(rank_wettest <= {records_top_N}) & (metric_name != 'precip')")
      .assign(record=lambda x: x.metric_name + "iest_month")
      .rename({'total': 'metric_value', 'month_year': 'date', 'rank_wettest': 'rank'}, axis=1)[record_order]
    ),
    (
      month_precip_records
      .query(f"(rank_driest <= {records_top_N}) & (metric_name == 'precip')")
      .assign(record='driest_month')
      .rename({'total': 'metric_value', 'month_year': 'date', 'rank_driest': 'rank'}, axis=1)[record_order]
    ),
    (
      snow_year
      .query(f"rank<={records_top_N}")
      .assign(record='least_snow_year')
      .rename({'total': 'metric_value', 'year_for_dash': 'date'}, axis=1)[record_order]
    ),
    (
      snow_year
      .drop(['rank'], axis=1)
      .query(f"rank_wettest<={records_top_N}")
      .assign(record='most_snow_year')
      .rename({'total': 'metric_value', 'year_for_dash': 'date', 'rank_wettest': 'rank'}, axis=1)[record_order]\
    ),
    rain_year,
  ]).sort_values(['record', 'rank'])

precip_records_nodaily = compute_monthly_precip_records()
precip_records = compute_daily_precip_records(precip)
curr_precip_month = precip['date'].max().strftime('%b')
curr_temp_month = temps['date'].max().strftime('%b')
temp_records = compute_temp_records()

# First/Last Freeze Percentiles
freeze_dates = (
  temps.query("min_temp<= 32.0")
  .groupby('snow_season')
  .agg(
    first_freeze_doy=('snow_day_of_year', 'min'),
    first_freeze_date=('date', 'min'),
    last_freeze_doy=('snow_day_of_year', 'max'),
  )
  .reset_index()
  .query(f"(snow_season < '{max_snow_year}') & (snow_season >= '{min_snow_year}')")
)

freeze_percs = pd.concat([
  pd.DataFrame(
    freeze_dates['first_freeze_doy'].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9,1]))\
    .rename({'first_freeze_doy': 'freeze_doy'}, axis=1)\
    .assign(metric='first_freeze'),
  pd.DataFrame(
    freeze_dates['last_freeze_doy'].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9,1]))\
    .rename({'last_freeze_doy': 'freeze_doy'}, axis=1)\
    .assign(metric='last_freeze'),
]).reset_index()

freeze_percs['date'] = (
  freeze_percs
  .apply(lambda x: dayofyear_to_month_day(int((x['freeze_doy'] - 154) % 365 + 1)), axis=1)
)

freeze_percs['percentile'] = freeze_percs['index'].apply(lambda x: f"{int(x*100)}th Percentile")
freeze_percs = freeze_percs[['date', 'percentile', 'metric']]

snow_dates = (
  precip
  .query("snow> 0.0")
  .groupby('snow_season')
  .agg(
    first_snow_doy=('snow_day_of_year', 'min'),
    first_snow_date=('date', 'min'),
    last_snow_doy=('snow_day_of_year', 'max'),
  )
  .reset_index()
  .query(f"(snow_season < '{max_snow_year}') & (snow_season >= '{min_snow_year}')")
)

snow_percs = pd.concat([
  pd.DataFrame(
    snow_dates['first_snow_doy'].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9,1]))\
    .rename({'first_snow_doy': 'freeze_doy'}, axis=1)\
    .assign(metric='first_snow'),
  pd.DataFrame(
    snow_dates['last_snow_doy'].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9,1]))\
    .rename({'last_snow_doy': 'freeze_doy'}, axis=1)\
    .assign(metric='last_snow'),
]).reset_index()

snow_percs['date'] = (
  snow_percs
  .apply(lambda x: dayofyear_to_month_day(int((x['freeze_doy'] - 154) % 365 + 1)), axis=1)
)

snow_percs['percentile'] = snow_percs['index'].apply(lambda x: f"{int(x*100)}th Percentile")
snow_percs = snow_percs[['date', 'percentile', 'metric']]


dashboard_tables = {
  "rain_day": generic_data_table(precip_records.query("record == 'rainiest day'"), id='rain_day' ,clean_table=True, metric_value='rain (in)'),
  'rain_day_month': generic_data_table(precip_records.query(f"record == 'rainiest day ({curr_precip_month})'"), id='rain_day_month', clean_table=True, metric_value='rain (in)'),
  'dry_day': generic_data_table(precip_records.query("record == 'consecutive_days_dry'"), id='dry_day', clean_table=True, metric_value='days'),
  'snow_day' : generic_data_table(precip_records.query("record == 'snowiest day'"), id='snow_day', clean_table=True, metric_value='snow (in)'),
  'snow_day_month' : generic_data_table(precip_records.query(f"record == 'snowiest day ({curr_precip_month})'"), id='snow_day_month', clean_table=True, metric_value='snow (in)'),
  'snow_early' : generic_data_table(precip_records.query("record == 'earliest_snow'"), id='snow_early', clean_table=True, metric_value='snow (in)'),
  'snow_late' : generic_data_table(precip_records.query("record == 'latest_snow'"), id='snow_late', clean_table=True, metric_value='snow (in)'),
  'rain_month' : generic_data_table(precip_records_nodaily.query("record == 'rainiest_month'"), id='rain_month', clean_table=True, metric_value='rain (in)'),
  'snow_month' : generic_data_table(precip_records_nodaily.query("record == 'snowiest_month'"), id='snow_month', clean_table=True, metric_value='snow (in)'),
  'rain_year' : generic_data_table(precip_records_nodaily.query("record == 'most_rain_year'"), id='rain_year', clean_table=True, metric_value='rain (in)'),
  'snow_year' : generic_data_table(precip_records_nodaily.query("record == 'most_snow_year'"), id='snow_year', clean_table=True, metric_value='snow (in)'),
  'dry_month' : generic_data_table(precip_records_nodaily.query("record == 'driest_month'"), id='dry_month', clean_table=True, metric_value='precip (in)'),
  'dry_snow' : generic_data_table(precip_records_nodaily.query("record == 'least_snow_year'"), id='dry_snow', clean_table=True, metric_value='snow (in)'),
  'cold_day' : generic_data_table(temp_records.query("record == 'coldest_day'"), id='cold_day', clean_table=True, metric_value='temp (F)'),
  'cold_day_month' : generic_data_table(temp_records.query(f"record == 'coldest_day ({curr_temp_month})'"), id=f'cold_day_month', clean_table=True, metric_value='temp (F)'),
  'hot_day' : generic_data_table(temp_records.query("record == 'hottest_day'"), id='hot_day', clean_table=True, metric_value='temp (F)'),
  'hot_day_month' : generic_data_table(temp_records.query(f"record == 'hottest_day ({curr_temp_month})'"), id=f'hot_day_month', clean_table=True, metric_value='temp (F)'),
  'hottest_min' : generic_data_table(temp_records.query("record == 'hottest_minimum'"), id='hot_min', clean_table=True, metric_value='temp (F)'),
  'coldest_max' : generic_data_table(temp_records.query("record == 'coldest_maximum'"), id='coldest_max', clean_table=True, metric_value='temp (F)'),
  '100_deg' : generic_data_table(temp_records.query("record == '100_degree_days'"), id='100_deg', clean_table=True, metric_value='temp (F)'),
  'first_freeze': generic_data_table(temp_records.query("record == 'first_freeze'"), id='first_freeze', clean_table=True, metric_value='temp (F)'),
  'last_freeze': generic_data_table(temp_records.query("record == 'last_freeze'"), id='last_freeze', clean_table=True, metric_value='temp (F)'),
  'first_freeze_perc': generic_data_table(freeze_percs.query("metric == 'first_freeze'")[['date', 'percentile']], id='first_freeze_perc', clean_table=False, metric_value='date'),
  'last_freeze_perc': generic_data_table(freeze_percs.query("metric == 'last_freeze'")[['date', 'percentile']], id='last_freeze_perc', clean_table=False, metric_value='date'),
  'first_snow_perc': generic_data_table(snow_percs.query("metric == 'first_snow'")[['date', 'percentile']], id='first_snow_perc', clean_table=False, metric_value='date'),
  'last_snow_perc': generic_data_table(snow_percs.query("metric == 'last_snow'")[['date', 'percentile']], id='last_snow_perc', clean_table=False, metric_value='date'),
}

# Temp year records
temp_year_records = (
  temps_full
  .assign(
    degree_100=lambda x: (x.max_temp >= 100)*1,
    frost_days=lambda x: (x.min_temp <= 32)*1,
  )
  .groupby(['year'])
  .agg(
    full_max=('max_temp', 'max'),
    full_min=('min_temp', 'min'),
    days_100=('degree_100', 'sum'),
    frost_days=('frost_days', 'sum'),
    diurnal_temp_range=('diurnal_temp_range', 'mean'),
  )
  .reset_index()
  .melt(['year'], ['full_max', 'full_min', 'days_100', 'frost_days', 'diurnal_temp_range'], 'metric_name', 'total')
  .assign(rank=lambda x: x.groupby('metric_name')['total'].rank(method='min', ascending=False))
  .query(f"year < {max_date.year}")
)

# Compute additional metrics from temp table
additional_yearly_metrics = ['cloud_cover', 'pressure', 'dew_point', 'wind_speed']
additional_yearly_rank = (
  temps[['year', *additional_yearly_metrics]]
  .groupby('year').mean().reset_index()
  .melt(['year'], additional_yearly_metrics, 'metric_name', 'total')
  .assign(rank=lambda x: x.groupby('metric_name')['total'].rank(method='min', ascending=False))
)

yearly_trend_metrics = pd.concat([
  (
    monthly_map
    .query("month == 'Year'")
    .rename({'mean': 'total', "metric": 'metric_name'}, axis=1)
    [['metric_name', 'year', 'total', 'rank']]
  ),
  (
    precip_rank
    .query("month == 'Year'")
    .query("year_type in ('water_year')")
    .rename({'year_for_dash': 'year'}, axis=1)
    [['metric_name', 'year', 'total', 'rank']]
  ),
  temp_year_records[['metric_name', 'year', 'total', 'rank']],
  additional_yearly_rank[['metric_name', 'year', 'total', 'rank']],
])
yearly_trend_metrics['year'] = yearly_trend_metrics['year'].astype('int')
min_trend_year = yearly_trend_metrics["year"].min()
max_trend_year = yearly_trend_metrics["year"].max()-1 # exclude partial year
yearly_trend_metrics = yearly_trend_metrics.query(f"year <= {max_trend_year}")

records_dash_disagg = (
  temps[['date','year','day_of_year','min_temp', 'max_temp', 'avg_temp', 'dew_point', 'wind_speed','wind_chill', 'pressure', 'heat_index', 'diurnal_temp_range']]
  .merge(
    precip[['date', 'precip', 'rain', 'snow']],
    on='date',
    how='left'
  )
)

records_dash_options = [
  x for x in records_dash_disagg.columns \
    if x not in ['date', 'year', 'day_of_year', 'high_rank', 'low_rank']
]

yearly_trend_labels = {
  'avg_temp': 'Mean Temp',
  'max_temp': 'Avg High Temp',
  'min_temp': 'Avg Low Temp',
  'precip': 'Precipitation (water year)',
  'rain': 'Rainfall (water year)',
  'snow': 'Snowfall (water year)',
  'precip_day': 'Precipitation Days',
  'full_max': 'Yearly Maximum Temp',
  'full_min': 'Yearly Minimum Temp',
  'days_100': '100 Degree Days',
  'frost_days': "32 Degree Days",
  'cloud_cover': 'Cloud Cover (%)',
  'pressure': 'Pressure (mb)',
  'dew_point': 'Dew Point (F)',
  'wind_speed': 'Wind Speed (mph)',
  'diurnal_temp_range': 'Avg Diurnal Temp Range (F)',
}

records_dash_labels = {
  x: (yearly_trend_labels | {'wind_chill': 'Wind Chill (F)', 'heat_index': 'Heat Index (F)', 'max_temp': "High Temp", 'min_temp': "Low Temp"} ).get(x, x)\
      for x in records_dash_options
}

wind = (
  temps[['date', 'month' ,'wind_speed', 'Wind Direction']]
  .assign(
    wind_speed_cat = lambda x: pd.cut(x['wind_speed'], bins=[0,10,15,20,30,40, temps['wind_speed'].max()], labels=['0-10', '10-15' ,'15-20','20-30','30-40','40+']),
    wind_direction_cat = lambda x: (round(x['Wind Direction']/ (360/16)) * (360/16)) % 360
  )
  .groupby(['month','wind_speed_cat', 'wind_direction_cat'])
  .agg(frequency=('date', 'count'))
  .reset_index()
)

# Precip dynamic colors
non_snow_colors = {
  'year_lines': "rgb(144, 238, 144, 0.25)",
  'avg_line': "rgb(0, 225, 0, 0.9)",
  'current_year_line': 'rgb(50, 255, 210)',
  'fillcolor1': 'rgba(0, 200, 0, 0.25)',
  'fillcolor2': 'rgba(0, 150, 0, 0.1)',
  'monthly_total': ['#228B22', '#00FF00'],
}

snow_colors = {
  'year_lines': "rgb(0, 144, 238, 0.25)",
  'avg_line': "rgb(0, 200, 255, 0.9)",
  'current_year_line': 'rgb(50, 255, 210)',
  'fillcolor1': 'rgba(0, 100, 255, 0.5)',
  'fillcolor2': 'rgba(0, 75, 200, 0.25)',
  'monthly_total': ['#007bff', '#00f0ff'],  
}

# LAYOUT -------------------------------------------------------------------------
dbc_row_col = lambda x, width=12: dbc.Row(children=[dbc.Col([x], width=width)])
first_col_width = 7

# Dash layout
app.layout = dbc.Container([
  dbc.Tabs([
    dcc.Tab(label='Temperature', children = [
      dbc_row_col(html.Div("Select start for 5-year range (daily only):"), width=first_col_width),
      dbc_row_col(
          dcc.Slider(
              min=int(round(temps_minmax[0]/10)*10),
              max=temps_minmax[1],
              step=1,
              id='heatmap-year-slider',
              value=temps_minmax[1] - 4,
              marks={str(year): str(year) for year in range(int(round(temps_minmax[0]/10)*10), temps_minmax[1]+1, 5)},
              tooltip={"placement": "bottom", "always_visible": True},
              included=False,
          ), width=first_col_width         
      ),
      dbc_row_col(html.Div(children="Daily Observed Temperature", style={'fontSize': 24})),
      dbc.Row([
        dcc.Graph(figure={}, id='daily_temperature_bar')
      ]),
      dbc_row_col(html.Div("Select metric for heatmaps:"), width=first_col_width),
      dbc_row_col(
        dcc.Dropdown(
          options={
            'max_temp': 'High Temperature',
            'min_temp':'Low Temperature', 
            'avg_temp': 'Mean Temperature',
          },
          value='max_temp',
          style={"color": "#000000"},
          id='daily_heatmap_dropdown',
        ), width=first_col_width
      ),
      dbc.Row([
        dbc.Col([
          html.Div(children="Daily Temperature Deviation from Normal", style={'fontSize': 24}),
        ], width=first_col_width),
        dbc.Col([
          html.Div(children="Monthly Normals", style={'fontSize': 24}),
        ], width=12-first_col_width),
      ]),
      dbc.Row([
        dbc.Col([
          dcc.Graph(figure={}, id='calendar_heatmap')
        ], width=first_col_width),
        dbc.Col([
          dcc.Graph(figure={}, id='monthly_avg_temp')
        ], width=12-first_col_width),
      ]),
      dbc_row_col(html.Div("Monthly Temperature Rank (1 = Coldest)", style={'fontSize': 24})),
      dbc_row_col(
        dcc.Graph(figure={}, id='monthly_temp_heatmap')
      ),
    ]),
    dcc.Tab(label='Precipitation', children=[
      dbc_row_col(html.Div("Select precipitation metric:")),
      dbc_row_col(
        dcc.Dropdown(
          options={
            'precip': 'Total Precipitation (in)',
            'rain': 'Rainfall (in)', 
            'snow': 'Snowfall (in)',
          },
          value='precip',
          style={"color": "#000000"},
          id='precip_metric_dropdown',
        ),
      ),
      dbc_row_col(html.Div("Select Year Type:")),
      dbc_row_col(
        dcc.Dropdown(
          options={
            'calendar_year': 'Calendar Year (Jan-Dec)',
            'water_year': 'Water Year (Oct-Sep)', 
            'snow_season': 'Snow Season (Aug-Jul)',
          },
          value='calendar_year',
          style={"color": "#000000"},
          id='water_calendar_dropdown',
        ),
      ),
      dbc.Row([
        dbc.Col([
          html.Div("Year to Date Precipitation vs. Normals", style={'fontSize': 24}),
        ], width=6),
        dbc.Col([
          html.Div(f"Month to Date Precipitation vs. Normals ({current_precip_month})", style={'fontSize': 24}),
        ], width=6)
      ]),
      dbc.Row([
        dbc.Col([
          dcc.Graph(figure={}, id='ytd_precip_chart'),
        ], width=6),
        dbc.Col([
          dcc.Graph(figure={}, id='mtd_precip_chart'),
        ], width=6)
      ]),
      dbc.Row([
        dbc.Col([
          html.Div("Monthly Precip Total", style={'fontSize': 24})
          ], width=6),
        dbc.Col([html.Div("First Freeze Dates", style={'fontSize': 24})], width=3),
        dbc.Col([html.Div("First Snow Dates", style={'fontSize': 24})], width=3),
      ]),
      dbc.Row([
        dbc.Col([
          dcc.Graph(figure={}, id='monthly_precip_total')
          ], width=6),
        dbc.Col([
            dashboard_tables['first_freeze_perc'],
            html.Br(),
            html.Div("Last Freeze Dates", style={'fontSize': 24}), 
            dashboard_tables['last_freeze_perc'],
          ],
          width=3
        ),
        dbc.Col([
            dashboard_tables['first_snow_perc'],
            html.Br(),
            html.Div("Last Snow Dates", style={'fontSize': 24}), 
            dashboard_tables['last_snow_perc'],
          ],
          width=3
        ),
      ]),
      dbc_row_col(
        html.Div("Monthly Precipitation Rank (1=driest)", style={'fontSize': 24}),
      ),
      dbc_row_col(
        dcc.Graph(figure={}, id='monthly_precip_heatmap')
      ),
    ]),
    dcc.Tab(label='Hourly', children=[
      dbc_row_col(html.Div("Select metric for hourly plots:")),
      dbc_row_col(
        dcc.Dropdown(
          options=hourly_metrics_pretty,
          value='temp',
          style={"color": "#000000"},
          id='hourly_monthly_input',
        )
      ),
      dbc_row_col(html.Div("Hourly Average by Month", style={'fontSize': 24})),
      dbc_row_col(dcc.Graph(figure={}, id='hourly_monthly_scatter')),
      dbc_row_col(html.Div("30-Year Hourly Average", style={'fontSize': 24})),      
      dbc_row_col(dcc.Graph(figure={}, id='hourly_heatmap')),
      dbc.Row([
        dbc.Col([
          html.Div(children="Seasonal Hourly Average", style={'fontSize': 24}),
        ], width=6),
        dbc.Col([html.Div(children="Seasonal Hourly Deviation", style={'fontSize': 24})], width=6),
      ]),
      dbc.Row([
        dbc.Col([
          dcc.Graph(figure={}, id='hourly_season'),
        ], width=6),
        dbc.Col([
          dcc.Graph(figure={}, id='hourly_season_deviation'),
        ], width=6),
      ]),
      dbc.Row([
        dbc.Col([html.Div("2D hourly", style={'fontSize': 24})], width=6),
        dbc.Col([html.Div("Wind Direction/Speed", style={'fontSize': 24})], width=6),
      ]),
      dbc_row_col(html.Div("Select month(s):"), width=6),
      dbc_row_col(
        dcc.Dropdown(
          options=month_order,
          value=month_order,
          style={"color": "#000000"},
          id='hourly_month_dropdown',
          multi=True,
          closeOnSelect=False,
        ), width=6
      ),
      dbc_row_col(html.Br()),
      dbc.Row([
        dbc.Col([dcc.Graph(figure={}, id='hourly_2d')], width=6),
        dbc.Col([dcc.Graph(figure={}, id='wind_direction')], width=6),
      ]),
      dbc_row_col(html.Div("Select x-axis metric:"), width=6),
      dbc_row_col(
        dcc.Dropdown(
          options=hourly_metrics_pretty,
          value='temp',
          style={"color": "#000000"},
          id='hourly_2d_xaxis',
        ), width=6
      ),
      dbc_row_col(html.Div("Select y-axis metric:"), width=6),
      dbc_row_col(
        dcc.Dropdown(
          options=hourly_metrics_pretty,
          value='pressure',
          style={"color": "#000000"},
          id='hourly_2d_yaxis',
        ), width=6
      ),
    ]),
    dcc.Tab(label='On This Day', children=[
      dbc_row_col(html.Div("Select Day of Year:")),
      dbc_row_col(
        dcc.Dropdown(
          options=formatted_dates,
          value=date.today().strftime('%b %d'),
          style={"color": "#000000"},
          id='datepicker_day_of_year',
        ), width=6
      ),
      dbc.Row([
        dbc.Col([
          html.Div(children="Min Temperature", style={'fontSize': 24}),
          dbc_row_col(dcc.Graph(figure={}, id='on_this_day_min_temp')),
        ], width=4),
        dbc.Col([
          html.Div(children="Max Temperature", style={'fontSize': 24}),
          dbc_row_col(dcc.Graph(figure={}, id='on_this_day_max_temp')),
        ], width=4),
        dbc.Col([
          html.Div(children="Cloud Cover", style={'fontSize': 24}),
          dbc_row_col(dcc.Graph(figure={}, id='on_this_day_cloud_cover')),
        ], width=4),
      ]),
      dbc.Row([
        dbc.Col([
          html.Div(children="Rain", style={'fontSize': 24}),
          dbc_row_col(dcc.Graph(figure={}, id='on_this_day_rain')),
        ], width=4),
        dbc.Col([
          html.Div(children="Snow", style={'fontSize': 24}),
          dbc_row_col(dcc.Graph(figure={}, id='on_this_day_snow')),
        ], width=4),
        dbc.Col([
          html.Div(children="Dew Point", style={'fontSize': 24}),
          dbc_row_col(dcc.Graph(figure={}, id='on_this_day_dew_point')),
        ], width=4),         
      ]),
      dbc.Row([
        dbc.Col([
          html.Div(children="High Temperature by Year", style={'fontSize': 24}),
          dbc_row_col(dcc.Graph(figure={}, id='on_this_day_max_temp_trend')),
        ], width=6),
        dbc.Col([
          html.Div(children="Low Temperature by Year", style={'fontSize': 24}),
          dbc_row_col(dcc.Graph(figure={}, id='on_this_day_min_temp_trend')),
        ], width=6),        
      ]),
      dbc.Row([
        dbc.Col([
          html.Div(children="Precipitation by Year", style={'fontSize': 24}),
          dbc_row_col(dcc.Graph(figure={}, id='on_this_day_precip_trend')),
        ], width=6),
        dbc.Col([
          html.Div(children="Cloud Cover by Year", style={'fontSize': 24}),
          dbc_row_col(dcc.Graph(figure={}, id='on_this_day_cloud_cover_trend')),
        ], width=6),        
      ]),
    ]),
    dcc.Tab(label='Records', children=[
      dbc_row_col(html.Div("Hot Records", style={'fontSize': 24})),
      dbc.Row([
        dbc.Col([
          html.H4("Hottest Day", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4(f"Hottest Day ({curr_temp_month})", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4(f"Hottest Minimum", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4(f"100 Degree Days", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
      ]),
      dbc.Row([
        dbc.Col(dashboard_tables['hot_day'], width=3),
        dbc.Col(dashboard_tables['hot_day_month'], width=3),
        dbc.Col(dashboard_tables['hottest_min'], width=3),
        dbc.Col(dashboard_tables['100_deg'], width=3),
      ]),
      dbc_row_col(html.Div("Cold Records", style={'fontSize': 24})),
      dbc.Row([
        dbc.Col([
          html.H4("Coldest Day", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4(f"Coldest Day ({curr_temp_month})", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4(f"Coldest Maximum", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
      ]),
      dbc.Row([
        dbc.Col(dashboard_tables['cold_day'], width=3),
        dbc.Col(dashboard_tables['cold_day_month'], width=3),
        dbc.Col(dashboard_tables['coldest_max'], width=3),
      ]),
      dbc_row_col(html.Div("Rain Records", style={'fontSize': 24})),
      dbc.Row([
        dbc.Col([
          html.H4("Rainest Day (all time)", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4(f"Rainiest Day ({curr_precip_month})", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4("Rainiest Months", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4("Rainiest Years", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
      ]),
      dbc.Row([
        dbc.Col(dashboard_tables['rain_day'], width=3),
        dbc.Col(dashboard_tables['rain_day_month'], width=3),
        dbc.Col(dashboard_tables['rain_month'], width=3),
        dbc.Col(dashboard_tables['rain_year'], width=3),
      ]),
      dbc_row_col(html.Div("Snow Records", style={'fontSize': 24})),
      dbc.Row([
        dbc.Col([
          html.H4("Snowiest Day (all time)", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4(f"Snowiest Day ({curr_precip_month})", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4("Snowiest Month", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4("Snowiest Year", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),           
      ]),
      dbc.Row([
        dbc.Col(dashboard_tables['snow_day'], width=3),
        dbc.Col(dashboard_tables['snow_day_month'], width=3),
        dbc.Col(dashboard_tables['snow_month'], width=3),
        dbc.Col(dashboard_tables['snow_year'], width=3),
      ]),
      dbc_row_col(html.Div("Dry Records", style={'fontSize': 24})),
      dbc.Row([
        dbc.Col([
          html.H4("Days without rain/snow", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4(f"Driest Month", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4("Driest Snow Year", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
      ]),
      dbc.Row([
        dbc.Col(dashboard_tables['dry_day'], width=3),
        dbc.Col(dashboard_tables['dry_month'], width=3),
        dbc.Col(dashboard_tables['dry_snow'], width=3),
      ]),      
      dbc_row_col(html.Div("Frost/Freeze Records", style={'fontSize': 24})),
      dbc.Row([
        dbc.Col([
          html.H4(f"First Freeze", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4(f"Last Freeze", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4("First Snow", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
        dbc.Col([
          html.H4(f"Last Snow", style={'textAlign': 'center', 'marginBottom': '10px'})
        ], width=3),
      ]),
      dbc.Row([
        dbc.Col(dashboard_tables['first_freeze'], width=3),
        dbc.Col(dashboard_tables['last_freeze'], width=3),        
        dbc.Col(dashboard_tables['snow_early'], width=3),
        dbc.Col(dashboard_tables['snow_late'], width=3),
      ]),
    ]),
    dcc.Tab(label='Trend', children=[
      dbc_row_col(html.Div("Select metric:")),
      dbc_row_col(
        dcc.Dropdown(
          options=yearly_trend_labels,
          value='precip',
          style={"color": "#000000"},
          id='yearly_trend_dropdown',
        ),
      ),
      dbc_row_col(html.Div("Select start year:")),
      dbc_row_col(
        dcc.Slider(
            min=int(round(min_trend_year/10)*10),
            max=max_trend_year-1,
            step=1,
            id='yearly_trend_slider',
            value=1970,
            marks={str(year): str(year) for year in range(int(round(temps_minmax[0]/10)*10), temps_minmax[1]+1, 5)},
            tooltip={"placement": "bottom", "always_visible": True},
            included=False,
        ), width=6         
      ),
      dbc.Row([
        dbc.Col(html.Div("Yearly Trend", style={'fontSize': 24}), width=8),
        dbc.Col(html.Div("Yearly Scatter", style={'fontSize': 24}), width=4),
      ]),
      dbc.Row([
        dbc.Col(dcc.Graph(figure={}, id='yearly_trend'), width=8),
        dbc.Col(dcc.Graph(figure={}, id='yearly_scatter'), width=4),
      ]),
      dbc_row_col(html.Div("Select records metric:")),
      dbc_row_col(
        dcc.Dropdown(
          options=records_dash_labels,
          value='avg_temp',
          style={"color": "#000000"},
          id='records_yearly_dropdown',
        ), width=8
      ),
      dbc.Row([
        dbc.Col(html.Div("Number of Daily Records", style={'fontSize': 24}), width=8),
        dbc.Col(html.Div("Departure from 30yr normal", style={'fontSize': 24}), width=4),
      ]),
      dbc.Row([
        dbc.Col(dcc.Graph(figure={}, id='records_yearly'), width=8),
        dbc.Col(dcc.Graph(figure={}, id='yearly_trend_from_norm'), width=4),
      ]),      
    ]),
  ]),
], fluid=True)
# Callbacks ----------------------------------------------------------------------

# Temperature Figure
@callback(
    Output(component_id='daily_temperature_bar', component_property='figure'),
    Input(component_id='heatmap-year-slider', component_property='value')    
)
def update_temperature_chart(start_year):

  recent = temps_full[
    temps_full['date'].between(f'{start_year}-01-01', f"{start_year+4}-12-31'")
  ]
  record_highs = recent[recent['high_rank'] == 1.0]
  record_lows = recent[recent['low_rank'] == 1.0]
  high_min = recent[recent['low_rank'] == recent['low_rank'].max()]
  low_max = recent[recent['high_rank'] == recent['high_rank'].max()]
  
  fig = go.Figure()
  # Total Temp
  fig.add_bar(
      x=recent['date'],
      y=recent['range'],
      base=recent['min_temp'],  # This sets the starting point of each bar
      marker_color='red',
      name='Observed Temperature',
      customdata=recent[['normal_high', 'normal_low']],
      hovertemplate=(
        '<b>Date:</b> %{x}<br><br>' +
        '<b>Observed High:</b> %{y}°F<br>' +
        '<b>Observed Low:</b> %{base}°F<br>' + 
        '<b>Normal High:</b> %{customdata[0]}°F<br>' +
        '<b>Normal Low:</b> %{customdata[1]}°F<br>' +
        '<extra></extra>'  # Removes the default trace info
    )
  )
  fig.add_trace(
      go.Scatter(
          x=recent['date'], y=recent['normal_temp'], mode='lines', name='Seasonal Normal',
          line=dict(color='darkred', width=3)
      ),
  )
  fig.add_trace(
      go.Scatter(
          x=recent['date'], y=recent['min_temp_p10'], mode='lines', name='10th percentile Low',
          line=dict(color='royalblue', width=1, dash='dot')
      ),
  )
  fig.add_trace(
      go.Scatter(
          x=recent['date'], y=recent['max_temp_p90'], mode='lines', name='90th Percentile High',
          line=dict(color='red', width=1, dash='dot')
      ),
  )
  fig.add_trace(
      go.Scatter(
          x=list(recent['date']) + list(recent['date'])[::-1],
          y=list(recent['normal_high']) + list(recent['normal_low'])[::-1],
          fill='toself',
          fillcolor='rgba(125,125,125,0.25)', 
          line_color='rgba(255,255,255,0)',
          showlegend=True,
          name='Normal',
          hoverinfo="skip",
      ),
  )
  fig.add_trace(
      go.Scatter(
          x=record_highs['date'],
          y=record_highs['max_temp'],
          mode='markers',
          marker=dict(color='darkred', size=8, symbol='diamond'),
          name='Record High'
      )
  )

  fig.add_trace(
      go.Scatter(
          x=record_lows['date'],
          y=record_lows['min_temp'],
          mode='markers',
          marker=dict(color='rgb(0,75,255)', size=8, symbol='diamond'),
          name='Record Low'
      )
  )
  fig.add_trace(
      go.Scatter(
          x=low_max['date'],
          y=low_max['max_temp'],
          mode='markers',
          marker=dict(color='skyblue', size=6, symbol='square'),
          name='Coldest High'
      )
  )
  fig.add_trace(
      go.Scatter(
          x=high_min['date'],
          y=high_min['min_temp'],
          mode='markers',
          marker=dict(color='mediumvioletred', size=6, symbol='square'),
          name='Warmest Low'
      )
  )

  # Add titles / figsize
  fig.update_layout(xaxis_title='Date', yaxis_title='Temperature (°F)', height=1000, template='plotly_dark')
  # Add Slider bar and buttons for year
  fig.update_layout(
      xaxis=dict(
          rangeselector=dict(
              buttons=list([
                  dict(count=3,
                      label="3m",
                      step="month",
                      stepmode="backward"),
                  dict(count=6,
                      label="6m",
                      step="month",
                      stepmode="backward"),
                  dict(count=1,
                      label="YTD",
                      step="year",
                      stepmode="todate"),
                  dict(count=1,
                      label="1y",
                      step="year",
                      stepmode="backward"),
                  dict(step="all")
              ]),
              bgcolor='black',
              activecolor='darkred',
          ),
          rangeslider=dict(
              visible=True
          ),
          type="date",
          range=[recent['date'].max() - pd.DateOffset(months=6), recent['date'].max() + pd.DateOffset(days=1)],
      ),
      template='plotly_dark',
  )
  return fig

# Monthly Avg Figure
@callback(
    Output(component_id='monthly_avg_temp', component_property='figure'),
    Input(component_id='daily_heatmap_dropdown', component_property='value'),   
)
def monthly_avg_temp(placeholder):
  fig = go.Figure()
  # Total Temp
  fig.add_bar(
      x=month_avgs['month'],
      y=month_avgs['normal_high'] - month_avgs['normal_low'],
      base=month_avgs['normal_low'],  # This sets the starting point of each bar
      marker_color='red',
      name='Normal',
      customdata=month_avgs[['record_high', 'record_low', 'record_high_year', 'record_low_year']],
      text = month_avgs['normal_high'].astype(str) + '°F',
      textposition='inside',
      hovertemplate=(
        '<b>Month:</b> %{x}<br>' +
        '<b>Avg High:</b> %{y}°F<br>' +
        '<b>Avg Low:</b> %{base}°F<br>' + 
        '<b>Record High:</b> %{customdata[0]}°F (%{customdata[2]})<br>' +
        '<b>Record Low:</b> %{customdata[1]}°F (%{customdata[3]})<br>' +
        '<extra></extra>'
    )
  )
  for i in range(month_avgs.shape[0]):
    fig.add_annotation(
        x=month_avgs['month'][i],
        y=month_avgs['normal_low'][i],
        text=(month_avgs['normal_low'].astype(str) + '°F').tolist()[i],
        showarrow=False,
        yshift=10,  # Adjust vertical position above the bar
    )
  fig.add_trace(
      go.Scatter(
          x=month_avgs['month'], y=month_avgs['record_high'], mode='lines+markers+text', name='Record High', text=month_avgs['record_high'],
          customdata=month_avgs[['record_high_year']],
          hovertemplate='Record High: %{y}°F (%{customdata[0]})<extra></extra>',
          line=dict(color='darkred', width=3)
      ),
  )
  fig.add_trace(
      go.Scatter(
          x=month_avgs['month'], y=month_avgs['record_low'], mode='lines+markers+text', name='Record Low', text=month_avgs['record_low'],
          customdata=month_avgs[['record_low_year']],
          hovertemplate='Record Low: %{y}°F (%{customdata[0]})<extra></extra>',
          line=dict(color='royalblue', width=3)
      ),
  )
  fig.add_trace(
      go.Scatter(
          x=list(month_avgs['month']) + list(month_avgs['month'])[::-1],
          y=list(month_avgs['max_temp_p90']) + list(month_avgs['min_temp_p10'])[::-1],
          fill='toself',
          fillcolor='rgba(125,125,125,0.25)', 
          line_color='rgba(255,255,255,0)',
          showlegend=True,
          name='10-90th percentile',
          hoverinfo="skip",
      ),
  )
  fig.add_vline(
      x=today.strftime("%b"),
      line_width=2,
      line_dash="dash",
      line_color='rgba(125,125,125,0.25)',
  )
  fig.update_layout(
    xaxis_title='Month', 
    yaxis_title='Temperature (°F)',
    paper_bgcolor='#333333', # figure
    plot_bgcolor="#222222", # plot area
  )
  return fig


# Daily Heatmap
@callback(
    Output(component_id='calendar_heatmap', component_property='figure'),
    Input(component_id='daily_heatmap_dropdown', component_property='value'),
    Input(component_id='heatmap-year-slider', component_property='value'),
)
def update_graph(metric_chosen, year_start):

  heatmap_df = temps_full[temps_full['year'].between(year_start, year_start+4)].copy()
  heatmap_df['departure'] = {
    'max_temp': heatmap_df['max_temp'] - heatmap_df['normal_high'],
    'min_temp': heatmap_df['min_temp'] - heatmap_df['normal_low'],
    'avg_temp': heatmap_df['avg_temp'] - heatmap_df['normal_temp'],
  }[metric_chosen]

  fig_heatmap = calplot(
      heatmap_df,
      x="date",
      y="departure",
      text=metric_chosen,
      # texttemplate="%{text}°F",
      month_lines=True,
      month_lines_color='#333333',
      month_lines_width=3.5,
      colorscale=temp_colors,
      years_title=True,
      showscale=True,
      cmap_min=-20,
      cmap_max=20,
      total_height=175*5,
      dark_theme=True,
      space_between_plots=0.04,
  )
  fig_heatmap.update_layout(
    # paper_bgcolor='#222222', # figure
    # plot_bgcolor='#222222', # plot area
  )

  return fig_heatmap


@callback(
    Output(component_id='monthly_temp_heatmap', component_property='figure'),
    Input(component_id='daily_heatmap_dropdown', component_property='value'),
)
def update_monthly_heatmap(metric_chosen):

  data_for_temp_heatmap = monthly_map[monthly_map['metric']==metric_chosen]

  fig = go.Figure(data=go.Heatmap(
      x=data_for_temp_heatmap['year'],
      y=data_for_temp_heatmap['month'],
      z=data_for_temp_heatmap['rank'],
      colorscale=temp_colors,
      zmin=1,
      xgap=1,
      ygap=1,
      zmax=data_for_temp_heatmap['rank'].max(),
      connectgaps=False,
      customdata=data_for_temp_heatmap[['mean',]],
      texttemplate="%{z}",
      hovertemplate=(
        '%{y} %{x}:' +
        '<br>Rank: %{z}' +
        f'<br>Avg({metric_chosen}): %{{customdata[0]:.1f}}°F' +
        '<extra></extra>'  # Removes the default trace info
    ),
  ))
  fig.update_layout(
      # title=f'Monthly Temperature Rank',
      xaxis_title='Year',
      yaxis=dict(title='Month',autorange='reversed'),
      height=600,
      # width=1420,
      paper_bgcolor='#333333',
      plot_bgcolor='#333333',
      margin=dict(l=20, r=20, t=20, b=20),
  )
  
  return fig

# Monthly Hourly Scatter
@callback(
    Output(component_id='hourly_monthly_scatter', component_property='figure'),
    Input(component_id='hourly_monthly_input', component_property='value'),
)
def update_monthly_heatmap(metric_name):
  fig = make_subplots(
    rows=1, cols=len(month_order), subplot_titles=month_order, shared_yaxes=True, horizontal_spacing=0.004,
    y_title=hourly_metrics_pretty[metric_name],
    x_title='Time of Day',
  )

  for i, month in enumerate(month_order):
    month_hour = hourly_all_year[(hourly_all_year['month']==month)&(hourly_all_year['metric_name']==metric_name)]
    month_mean = hourly_all_mean[(hourly_all_mean['month']==month)&(hourly_all_mean['metric_name']==metric_name)]

    fig.add_trace(
        go.Scatter(
            x=month_hour['hour'], 
            y=month_hour['metric_value'], 
            mode='markers', 
            name='Observed Month Avg', 
            marker=dict(size=4, color='red', opacity=0.3),
            # line=dict(width=1, dash='dot', color='rgba(255, 0, 0, 0.25)'),
            customdata=month_hour[['year', 'month']],
            hovertemplate=(
                "<b>%{customdata[1]} %{customdata[0]}</b><br>" +
                "<b>Hour: %{x}:00</b><br>"+
                f"<b>{hourly_metrics_pretty[metric_name]}: %{{y:.2f}}</b><br>"+
                "<extra></extra>"
            ),
            showlegend=(i==0),
            legendgroup='Group A',
        ), row=1, col=i+1
    )
    fig.add_trace(
        go.Scatter(
            x=month_mean['hour'], y=month_mean['metric_value'], mode='lines', name='Mean', line=dict(color='red', width=4),
            hovertemplate=(
            "<b>Hour: %{x}:00</b><br>"+
            f"<b>{hourly_metrics_pretty[metric_name]}: %{{y:.2f}}</b><br>"+
            "<extra></extra>"
            ),
            showlegend=(i==0),
            legendgroup='Group B',
        ), row=1, col=i+1
    )
  fig.update_layout(
      # title=f'Hourly Temperature for {month}',
      paper_bgcolor='#333333', # figure
      plot_bgcolor="#222222", # plot area
      height=800,
      legend=dict(
          orientation="h",
          yanchor="bottom",
          y=-0.10,
          xanchor="center",
          x=0.05,
      ),
  )

  return fig

# Seasonal Hourly Avg
@callback(
    Output(component_id='hourly_season', component_property='figure'),
    Input(component_id='hourly_monthly_input', component_property='value'),
)
def update_hourly_season(metric_name):
  data = hourly_all_mean_season[hourly_all_mean_season['metric_name']==metric_name]
  fig = px.line(
    data,
    x='hour', y='metric_value', color='season', color_discrete_map = {
      'winter': 'royalblue',
      'spring': 'green',
      'summer': 'gold',
      'fall': 'red',
    },
  )
  fig.update_traces(
    hovertemplate=("<b>Season: %{customdata[0]}</b><br>" +
      "<b>Hour:</b> %{x}<br>" +
      f"<b>{hourly_metrics_pretty[metric_name]}:</b> %{{y:.6f}}<extra></extra>"
    ),
    customdata=data[['season']],
  )
  fig.update_layout(
    paper_bgcolor='#333333', # figure
    plot_bgcolor="#222222", # plot area
    height=500,
    xaxis_title='Time of Day',
    yaxis_title=hourly_metrics_pretty[metric_name],
)
  return fig

# Seasonal avg adjusted
@callback(
    Output(component_id='hourly_season_deviation', component_property='figure'),
    Input(component_id='hourly_monthly_input', component_property='value'),
)
def update_hourly_season(metric_name):
  fig = px.line(
    hourly_all_mean_season[hourly_all_mean_season['metric_name']==metric_name],
    x='hour', y='avg_deviation', color='season', color_discrete_map = {
      'winter': 'royalblue',
      'spring': 'green',
      'summer': 'gold',
      'fall': 'red',
    },
  )
  fig.update_layout(
    paper_bgcolor='#333333', # figure
    plot_bgcolor="#222222", # plot area
    height=500,
    xaxis_title='Time of Day',
    yaxis_title=hourly_metrics_pretty[metric_name] + " Deviation from Avg",
)
  return fig

# Hourly heatmap
@callback(
    Output(component_id='hourly_heatmap', component_property='figure'),
    Input(component_id='hourly_monthly_input', component_property='value'),
)
def update_hourly_heatmap(metric_chosen):
  data_for_temp_heatmap = (
    hourly_heatmap[hourly_heatmap['metric_name']==metric_chosen]
  ).query("day_of_year != 59.5")

  fig = go.Figure(data=go.Heatmap(
      x=data_for_temp_heatmap['day_of_year'],
      y=data_for_temp_heatmap['hour'],
      z=data_for_temp_heatmap['metric_value'],
      customdata=data_for_temp_heatmap[['DoY_label']],
      colorscale='turbo',
      connectgaps=False,
      hovertemplate=(
        '<br>Day of Year: %{customdata[0]}' +
        '<br>Hour: %{y}:00' +
        f'<br>Avg({metric_chosen})=%{{z}}' +
        '<extra></extra>'
    ),
  ))
  fig.add_trace(
    go.Scatter(
        x=sunrise_sunset_clean['day_of_year'], 
        y=sunrise_sunset_clean['sunset_hr'], 
        mode='lines', 
        line=dict(color='black', width=0.5, dash='dot'), 
        showlegend=False,
        hovertemplate=(
          '<br>sunset: %{y:.2f}' +
          '<extra></extra>'
        )        
    )
  )
  fig.add_trace(
    go.Scatter(
        x=sunrise_sunset_clean['day_of_year'], 
        y=sunrise_sunset_clean['sunrise_hr'], 
        mode='lines',
        line=dict(color='black', width=0.5, dash='dot'),
        showlegend=False,
        hovertemplate=(
          '<br>sunrise: %{y:.2f}' +
          '<extra></extra>'
        )
    )    
  )
  fig.update_layout(
      # title=f'Monthly Temperature Rank',
      xaxis_title='Day of Year',
      yaxis_title='Time of Day',
      height=500,
      # width=1420,
      paper_bgcolor='#333333',
      plot_bgcolor='#333333',
      margin=dict(l=20, r=20, t=20, b=20),
  )
  return fig

@callback(
    Output(component_id='hourly_2d', component_property='figure'),
    Input(component_id='hourly_2d_xaxis', component_property='value'),
    Input(component_id='hourly_2d_yaxis', component_property='value'),
    Input(component_id='hourly_month_dropdown', component_property='value'),
)
def hourly_2d(metric1, metric2, months):
  
  min_year = (max_date.year-1) - 29 # 10 years
  fig = px.density_heatmap(
    hourly_temp.query(f"(year >= {min_year}) & (year < {max_date.year}) & (month in {months})"), 
    x=metric1, 
    y=metric2,
    marginal_x='histogram', 
    marginal_y='histogram',
    color_continuous_scale=px.colors.sequential.Viridis,
    nbinsx=100,
    nbinsy=100,
  )

  fig.update_layout(
    xaxis_title = hourly_metrics_pretty[metric1],
    yaxis_title = hourly_metrics_pretty[metric2],
    height=850,
  )

  return fig

# Precip Dash
@callback(
    Output(component_id='ytd_precip_chart', component_property='figure'),
    Input(component_id='water_calendar_dropdown', component_property='value'),
    Input(component_id='precip_metric_dropdown', component_property='value'),
)
def precip_ytd_chart(calendar_type, metric):
  ytd_dash = (
      ytd
      .query(f"(metric_name == '{metric}') & (year_type == '{calendar_type}')")
  )

  colors = snow_colors if metric == 'snow' else non_snow_colors

  current_year_ytd = ytd_dash.query("year_for_dash == current_year")
  not_current_ytd = ytd_dash.query(f"(year_for_dash != current_year)")
  chart_label = current_year_ytd['current_year'].head().iloc[0]

  ytd_avg = (
    ytd_normals
    .query(f"(metric_name == '{metric}') & (year_type == '{calendar_type}')")
    .sort_values(by=['year_type', 'metric_name', 'day_of_year_dash'])
  )

  labels_ytd = (
    ytd_avg[['day_of_year_dash', 'dashboard_date']]
    [ytd_avg['dashboard_date'].str.endswith('01')] 
  )

  fig = px.line(
    not_current_ytd, 
    x='day_of_year_dash', 
    y='year_to_date_precip', 
    color='year_for_dash',
  )
  fig.update_traces(
      line=dict(color=colors['year_lines'], width=0.75, dash='dot'),
      selector=dict(mode='lines'),  # Ensure it applies only to line traces
      customdata = not_current_ytd[['dashboard_date', 'year']],
      hovertemplate=(
        '<b>Date:</b> %{customdata[0]} %{customdata[1]}<br>' +
        f'<b>Year to Date {metric}:</b> %{{y}}<br>' +
        '<extra></extra>'
      ),
      showlegend=False,
  )
  fig.add_scatter(
    x=ytd_avg['day_of_year_dash'], 
    y=ytd_avg['avg_precip_ytd'], 
    mode='lines', 
    name='Average', 
    line=dict(color=colors['avg_line'], width=4),
    customdata = not_current_ytd[['dashboard_date', 'year']],
    hovertemplate=(
      '<b>Date:</b> %{customdata[0]} %{customdata[1]}<br>' +
      f'<b>{N_years}-year avg: %{{y}}<br>' +
      '<extra></extra>'
    )
  )
  fig.add_scatter(
    x=current_year_ytd['day_of_year_dash'], 
    y=current_year_ytd['year_to_date_precip'], 
    mode='lines', 
    name=chart_label, 
    line=dict(color=colors['current_year_line'], width=2),
    customdata = current_year_ytd[['dashboard_date', 'year']],
    hovertemplate=(
      '<b>Date:</b> %{customdata[0]} %{customdata[1]}<br>' +
      f'<b>{chart_label} {metric}</b>: %{{y}}<br>' +
      '<extra></extra>'
    )

  )
  fig.add_traces([
      go.Scatter(
          x=list(ytd_avg['day_of_year_dash']) + list(ytd_avg['day_of_year_dash'])[::-1],
          y=list(ytd_avg['p75_precip_ytd']) + list(ytd_avg['p25_precip_ytd'])[::-1],
          fill='toself',
          fillcolor=colors['fillcolor1'],
          line=dict(color='rgba(255,255,255,0)'),
          hoverinfo="skip",
          name='25th-75th Percentile',
          showlegend=False,
      )
  ])
  fig.add_traces([
      go.Scatter(
          x=list(ytd_avg['day_of_year_dash']) + list(ytd_avg['day_of_year_dash'])[::-1],
          y=list(ytd_avg['max_precip_ytd']) + list(ytd_avg['min_precip_ytd'])[::-1],
          fill='toself',
          fillcolor=colors['fillcolor2'],
          line=dict(color='rgba(255,255,255,0)'),
          hoverinfo="skip",
          name='Min/Max Percentile',
          showlegend=False
      )
  ])
  # fig.add_traces([
  #     go.Scatter(
  #         x=list(ytd_avg['day_of_year_dash']) + list(ytd_avg['day_of_year_dash'])[::-1],
  #         y=list(ytd_avg['p90_precip_ytd']) + list(ytd_avg['p10_precip_ytd'])[::-1],
  #         fill='toself',
  #         fillcolor='rgba(0, 150, 0, 0.1)',
  #         line=dict(color='rgba(255,255,255,0)'),
  #         hoverinfo="skip",
  #         name='10th-90th Percentile',
  #         showlegend=False
  #   )
  # ])
  fig.update_layout(
    height=800,
    xaxis=dict(
      title='Day of Year',
      tickvals=labels_ytd['day_of_year_dash'],
      ticktext=labels_ytd['dashboard_date'],
      tickangle=0,
      showgrid=True
    ), 
    yaxis_title=metric, 
    paper_bgcolor='#333333', 
    plot_bgcolor="#222222",
    legend_title=None,
  )
  return fig

@callback(
    Output(component_id='mtd_precip_chart', component_property='figure'),
    Input(component_id='precip_metric_dropdown', component_property='value'),
)
def mtd_precip_chart(metric):

  current_year_mtd = (
    mtd.query(f"year_for_dash == current_year & metric_name == '{metric}' & month == '{current_precip_month}'")
    .sort_values(['day_of_month'])
  )
  current_year_label = current_year_mtd['current_year'].head().iloc[0]
  mtd_avg_metric = mtd_avg.query(f"metric_name == '{metric}' & month == '{current_precip_month}'").dropna().sort_values(['day_of_month'])
  colors = snow_colors if metric == 'snow' else non_snow_colors

  fig = go.Figure()
  fig.add_scatter(
    x=mtd_avg_metric['day_of_month'], 
    y=mtd_avg_metric['avg_precip_mtd'], 
    mode='lines', 
    name='Average', 
    line=dict(color=colors['avg_line'], width=4),
    hovertemplate=(
      '<b>Day Of Month:</b> %{x}<br>' +
      f'<b>Avg {metric}: %{{y:.2f}} in<br>' +
      '<extra></extra>'
    )
  )
  fig.add_scatter(
    x=mtd_avg_metric['day_of_month'], 
    y=mtd_avg_metric['max_precip_mtd'], 
    mode='lines', 
    name='Maximum', 
    line=dict(color=colors['year_lines'], width=1, dash='dot')
  )
  fig.add_scatter(
    x=mtd_avg_metric['day_of_month'], 
    y=mtd_avg_metric['min_precip_mtd'], 
    mode='lines', 
    name='Minimum', 
    line=dict(color=colors['year_lines'], width=1, dash='dot')
  )
  fig.add_scatter(
    x=current_year_mtd['day_of_month'], 
    y=current_year_mtd['month_to_date_precip'], 
    mode='lines', name=current_precip_month + ' ' + str(current_year_label), 
    line=dict(color=colors['current_year_line'], width=2),
      hovertemplate=(
      f"<b>Date:</b> %{{x}} {current_precip_month + ' ' + str(current_year_label)}<br>" +
      f'<b>Cumulative {metric}: %{{y:.2f}} in<br>' +
      '<extra></extra>'
    )
  )
  fig.add_traces([
      go.Scatter(
          x=list(mtd_avg_metric['day_of_month']) + list(mtd_avg_metric['day_of_month'])[::-1],
          y=list(mtd_avg_metric['p75_precip_mtd']) + list(mtd_avg_metric['p25_precip_mtd'])[::-1],
          fill='toself',
          fillcolor=colors['fillcolor1'],
          line=dict(color='rgba(255,255,255,0)'),
          hoverinfo="skip",
          name='25th-75th Percentile',
          showlegend=False,
      )
  ])
  fig.add_traces([
      go.Scatter(
          x=list(mtd_avg_metric['day_of_month']) + list(mtd_avg_metric['day_of_month'])[::-1],
          y=list(mtd_avg_metric['max_precip_mtd']) + list(mtd_avg_metric['min_precip_mtd'])[::-1],
          fill='toself',
          fillcolor=colors['fillcolor2'],
          line=dict(color='rgba(255,255,255,0)'),
          hoverinfo="skip",
          name='Min/Max Percentile',
          showlegend=False
      )
  ])
  fig.add_traces([
      go.Scatter(
          x=list(mtd_avg_metric['day_of_month']) + list(mtd_avg_metric['day_of_month'])[::-1],
          y=list(mtd_avg_metric['p90_precip_mtd']) + list(mtd_avg_metric['p10_precip_mtd'])[::-1],
          fill='toself',
          fillcolor=colors['fillcolor2'],
          line=dict(color='rgba(255,255,255,0)'),
          hoverinfo="skip",
          name='10th-90th Percentile',
          showlegend=False
      )
  ])
  fig.update_layout(
    height=800, 
    xaxis_title=f'Day Of Month ({current_precip_month})', 
    yaxis_title=metric, 
    paper_bgcolor='#333333', 
    plot_bgcolor="#222222"
  )
  return fig

@callback(
    Output(component_id='monthly_precip_heatmap', component_property='figure'),
    Input(component_id='precip_metric_dropdown', component_property='value'),
    Input(component_id='water_calendar_dropdown', component_property='value'),
)
def update_monthly_precip_heatmap(metric_chosen, calendar_chosen):

  heatmap_color_scale = precip_colors
  data_for_precip_heatmap = (
    precip_rank
    .query(f"metric_name == '{metric_chosen}'")
    .query(f"year_type == '{calendar_chosen}'")
    .rename({'year_for_dash': 'year'}, axis=1)
  )

  fig = go.Figure(data=go.Heatmap(
      x=data_for_precip_heatmap['year'],
      y=data_for_precip_heatmap['month'],
      z=data_for_precip_heatmap['rank'],
      colorscale=heatmap_color_scale,
      zmin=1,
      xgap=1,
      ygap=1,
      zmax=data_for_precip_heatmap['rank'].max(),
      connectgaps=False,
      customdata=data_for_precip_heatmap[['total',]],
      texttemplate="%{z}",
      hovertemplate=(
        '%{y} %{x}:' +
        '<br>Rank: %{z}' +
        f'<br>Total ({metric_chosen}): %{{customdata[0]:.1f}} in' +
        '<extra></extra>'  # Removes the default trace info
    ),
  ))
  fig.update_layout(
      xaxis_title=calendar_chosen,
      yaxis=dict(title='Month',autorange='reversed'),
      height=600,
      # width=1420,
      paper_bgcolor='#333333',
      plot_bgcolor='#333333',
      margin=dict(l=20, r=20, t=20, b=20),
  )
  
  return fig

@callback(
    Output(component_id='monthly_precip_total', component_property='figure'),
    Input(component_id='precip_metric_dropdown', component_property='value'),
)
def update_monthly_precip_total(metric_chosen):

  colors = snow_colors if metric_chosen == 'snow' else non_snow_colors
  monthly_totals_for_graph = (
    monthly_totals
    .query("year_for_dash == current_year")
    .query(f"metric_name == '{metric_chosen}'")
    .rename({'year_for_dash': 'year'}, axis=1)
    .sort_values(['month', 'type'])
  )

  fig = px.histogram(
    monthly_totals_for_graph, 
    x="month", 
    y="metric_value", 
    color='type',
    barmode='group',
    height=550,
    color_discrete_sequence=colors['monthly_total'],
    text_auto=True,
  )

  fig.update_traces(
    texttemplate='%{y:.2f}',
    textposition='outside',
    customdata=monthly_totals_for_graph[['year', 'type']],
    hovertemplate=(
      '%{x} %{customdata[0]}:' +
      f'<br>%{{customdata[1]}} {metric_chosen}: %{{y:.1f}} in' +
      '<extra></extra>'  # Removes the default trace info
  ),
)

  fig.update_layout(
    xaxis_title="Month",
    yaxis_title=f"{metric_chosen} (in)",
    height=600,
    paper_bgcolor='#333333', 
    plot_bgcolor="#222222"
  )

  return fig

@callback(
    Output(component_id='yearly_trend', component_property='figure'),
    Input(component_id='yearly_trend_dropdown', component_property='value'),
    Input(component_id='yearly_trend_slider', component_property='value'),
)
def yearly_trend(metric, start_year):
  to_display = yearly_trend_metrics.query(f"metric_name == '{metric}'").query(f"year >= {start_year}").copy()
  to_display['year'] = pd.to_numeric(to_display['year'], errors='coerce')
  avg_line = to_display['total'].mean()
  fig=go.Figure()
  fig.add_trace(
    go.Bar(
      x=to_display['year'],
      y=to_display['total'],
      name=metric,
      text=to_display['total'],
      textposition='outside',
      texttemplate='%{text:.1f}'
    )
  )
  fig.add_hline(
      y=avg_line,
      line=dict(color='red', width=1.5, dash='dash'),  # Customize the line color, width, and style
      annotation_text=f"Avg: {avg_line:.2f}",  # Add a label for the line
      annotation_position="top left",  # Position the label
  )
  fig.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title='Year',
    yaxis_title=yearly_trend_labels[metric],
  )
  return fig


@callback(
    Output(component_id='yearly_trend_from_norm', component_property='figure'),
    Input(component_id='yearly_trend_dropdown', component_property='value'),
    Input(component_id='yearly_trend_slider', component_property='value'),
)
def yearly_trend_from_norm(metric, start_year):
  max_year = yearly_trend_metrics['year'].max()
  to_display = yearly_trend_metrics.query(f"metric_name == '{metric}'").query(f"year >= {start_year}").copy()
  to_display['year'] = pd.to_numeric(to_display['year'], errors='coerce')
  to_display['departure_from_normal'] = to_display['total'] - to_display.query(f"year >= {max_year - 30}")['total'].mean()
  
  fig=go.Figure()
  fig.add_trace(
    go.Bar(
      x=to_display['year'],
      y=to_display['departure_from_normal'],
      name=metric,
    )
  )
  fig.add_hline(
      y=0,
      line=dict(color='white', width=1.5, dash='dash'),  # Customize the line color, width, and style
  )
  fig.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title='Year',
    yaxis_title=yearly_trend_labels[metric],
  )
  return fig


@callback(
    Output(component_id='yearly_scatter', component_property='figure'),
    Input(component_id='yearly_trend_dropdown', component_property='value'),
    Input(component_id='yearly_trend_slider', component_property='value'),
)
def yearly_scatter(metric, start_year):
  to_display = yearly_trend_metrics.query(f"metric_name == '{metric}'").query(f"year >= {start_year}").copy()
  to_display['year'] = pd.to_numeric(to_display['year'], errors='coerce')
  
  # Fit the model
  X = sm.add_constant(to_display['year'] - to_display['year'].min())  # Add intercept
  y = to_display['total']
  model = sm.OLS(y, X)
  model_res = model.fit()
  lm_fit = model_res.predict(X)
  slope = model_res.params['year']
  slope_pvalue = model_res.pvalues['year']
  line_text = f'Slope: {10*slope:.2f}/decade, p-value: {slope_pvalue:.4f}{"*" if slope_pvalue < 0.01 else ""}'

  fig = go.Figure()
  fig.add_trace(
    go.Scatter(
      x=to_display['year'],
      y=to_display['total'],
      mode='markers',
      marker=dict(size=10),
      text=to_display['total'],  # Add the 'total' column as text
      textposition='top center',  # Position the text on top of the markers
      name=f'Yearly {metric.title()}',
    )
  )
  fig.add_trace(
    go.Scatter(
      x=to_display['year'],
      y=lm_fit,
      mode='lines',
      name='Linear Trend',    
    )
  )
  fig.add_annotation(
      x=to_display['year'].mean(),  # Place the label at the end of the trend line
      y=y.max()*1.10,  # Corresponding y-value at the end of the trend line
      text=line_text,  # Text to display
      showarrow=False,  # Show an arrow pointing to the trend line
      font=dict(color='red', size=22),  # Customize font color and size
      align="left",  # Align text to the left
      bgcolor="rgba(255, 255, 255, 0.8)",  # Add a semi-transparent background to the text
      bordercolor='red',
  )
  fig.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title='Year',
    yaxis_title=yearly_trend_labels[metric],
  )
  return fig


@callback(
    Output(component_id='records_yearly', component_property='figure'),
    Input(component_id='records_yearly_dropdown', component_property='value'),
)
def records_dash(metric):
  records_dash_disagg_rank = records_dash_disagg.copy()
  records_dash_disagg_rank['high_rank'] = records_dash_disagg_rank.groupby('day_of_year')[metric].rank(method='min', ascending=False)
  records_dash_disagg_rank['low_rank'] = records_dash_disagg_rank.groupby('day_of_year')[metric].rank(method='min', ascending=True)
  if metric in ['rain', 'snow', 'precip', 'precip_day']:
    records_dash_disagg_rank = records_dash_disagg_rank.query(f"{metric} > 0") # Too many 0 days

  records_dash = (
    records_dash_disagg_rank
    .assign(record_high=lambda x: x['high_rank'] == 1.0, record_low=lambda x: -1*(x['low_rank'] == 1.0))
    .groupby('year')
    .agg(
      record_high_days=('record_high', 'sum'),
      record_low_days=('record_low', 'sum'),
    )
    .reset_index()
  )

  fig = px.bar(records_dash, x='year', y=['record_high_days', 'record_low_days'], color_discrete_sequence=['red', '#6e6df7'])
  fig.update_layout(
    legend_title_text='',
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title='Year',
    yaxis_title=records_dash_labels[metric] + " Days",
  )
  return fig


@callback(
    Output(component_id='wind_direction', component_property='figure'),
    Input(component_id='hourly_month_dropdown', component_property='value'),
)
def wind_dir_graph(months):
  wind_chart = (
    wind
    .query(f"month in {months}")
    .groupby(['wind_direction_cat', 'wind_speed_cat'])
    .agg(frequency=('frequency', 'sum'))
    .reset_index()
  )
  
  wind_chart['percent'] = wind_chart['frequency'] / wind_chart['frequency'].sum()


  fig = px.bar_polar(wind_chart, r="percent", theta="wind_direction_cat",
                    color="wind_speed_cat", template="plotly_dark",
                    color_discrete_sequence= px.colors.sequential.Plasma_r)
  return fig

# On This day view
@callback(
    Output(component_id='on_this_day_min_temp', component_property='figure'),
    Input(component_id='datepicker_day_of_year', component_property='value'),
)
def on_this_day_min_temp(date_value):

  date_formatted = datetime.strptime(date_value+" 2024", "%b %d %Y")
  day_of_year = gen_DoY_index(pd.to_datetime(date_formatted))
  df = temps.query(f"day_of_year == {day_of_year}")

  fig = px.histogram(df, x="min_temp", histnorm='percent')
  fig.update_layout(xaxis_title="Minimum Temperature", yaxis_title="Frequency", bargap=0.1)
  return fig


@callback(
    Output(component_id='on_this_day_max_temp', component_property='figure'),
    Input(component_id='datepicker_day_of_year', component_property='value'),
)
def on_this_day_max_temp(date_value):

  date_formatted = datetime.strptime(date_value+" 2024", "%b %d %Y")
  day_of_year = gen_DoY_index(pd.to_datetime(date_formatted))
  df = temps.query(f"day_of_year == {day_of_year}")

  fig = px.histogram(df, x="max_temp", color_discrete_sequence=['firebrick'], histnorm='percent')
  fig.update_layout(xaxis_title="Maximum Temperature", yaxis_title="Frequency", bargap=0.1)
  return fig

@callback(
    Output(component_id='on_this_day_cloud_cover', component_property='figure'),
    Input(component_id='datepicker_day_of_year', component_property='value'),
)
def on_this_day_cloud_cover(date_value):

  date_formatted = datetime.strptime(date_value+" 2024", "%b %d %Y")
  day_of_year = gen_DoY_index(pd.to_datetime(date_formatted))
  df = temps.query(f"day_of_year == {day_of_year}")

  fig = px.histogram(df, x="cloud_cover", color_discrete_sequence=['cornflowerblue'], histnorm='percent')
  fig.update_traces(xbins={'start': 0, 'end': 100, 'size': 10})
  fig.update_layout(xaxis_title="Cloud Cover", yaxis_title="Frequency", bargap=0.1)

  return fig

@callback(
    Output(component_id='on_this_day_dew_point', component_property='figure'),
    Input(component_id='datepicker_day_of_year', component_property='value'),
)
def on_this_day_dew_point(date_value):

  date_formatted = datetime.strptime(date_value+" 2024", "%b %d %Y")
  day_of_year = gen_DoY_index(pd.to_datetime(date_formatted))
  df = temps.query(f"day_of_year == {day_of_year}")

  fig = px.histogram(df, x="dew_point", color_discrete_sequence=['darkgreen'], histnorm='percent')
  fig.update_layout(xaxis_title="Dew Point", yaxis_title="Frequency", bargap=0.1)

  return fig

@callback(
    Output(component_id='on_this_day_snow', component_property='figure'),
    Input(component_id='datepicker_day_of_year', component_property='value'),
)
def on_this_day_snow(date_value):

  date_formatted = datetime.strptime(date_value+" 2024", "%b %d %Y")
  day_of_year = gen_DoY_index(pd.to_datetime(date_formatted))
  df = precip.query(f"day_of_year == {day_of_year}").copy()
  snow_labels = ['0in', '0-1in', '1-3in', '3-5in', '5-8in', '8-12in', '12-18in', '18+in']
  snow_bins = lambda x: create_bins(
    x,
    [0, 1, 3, 5, 8, 12, 18], 
    snow_labels,
    snow_labels[-1],
  )
  df['snow'] = df['snow'].apply(snow_bins)

  fig = px.histogram(df, x="snow", color_discrete_sequence=['cyan'], histnorm='percent')
  fig.update_xaxes(categoryorder='array', categoryarray=snow_labels)
  fig.update_layout(xaxis_title="Precipitation", yaxis_title="Frequency", bargap=0.1)

  return fig


@callback(
    Output(component_id='on_this_day_rain', component_property='figure'),
    Input(component_id='datepicker_day_of_year', component_property='value'),
)
def on_this_day_rain(date_value):

  date_formatted = datetime.strptime(date_value+" 2024", "%b %d %Y")
  day_of_year = gen_DoY_index(pd.to_datetime(date_formatted))
  df = precip.query(f"day_of_year == {day_of_year}").copy()
  rain_labels = ['0in', '0-0.25in', '0.25-0.5in', '0.5-0.75in', '0.75-1in', '1-2in', '2+in']
  rain_bins = lambda x: create_bins(
    x,
    [0, 0.25, 0.5, 0.75, 1, 2], 
    rain_labels,
    rain_labels[-1],
  )
  df['rain'] = df['rain'].apply(rain_bins)

  fig = px.histogram(df, x="rain", color_discrete_sequence=['limegreen'], histnorm='percent')
  fig.update_xaxes(categoryorder='array', categoryarray=rain_labels)
  fig.update_layout(xaxis_title="Precipitation", yaxis_title="Frequency", bargap=0.1)

  return fig

@callback(
    Output(component_id='on_this_day_min_temp_trend', component_property='figure'),
    Input(component_id='datepicker_day_of_year', component_property='value'),
)
def on_this_day_min_temp_trend(date_value):

  date_formatted = datetime.strptime(date_value+" 2024", "%b %d %Y")
  day_of_year = gen_DoY_index(pd.to_datetime(date_formatted))
  df = temps.query(f"day_of_year == {day_of_year}")

  fig = px.bar(df, x="year", y='min_temp', text_auto=True)
  fig.update_layout(xaxis_title="Year", yaxis_title=f"{date_value} Minimum Temperature", bargap=0.15)
  return fig


@callback(
    Output(component_id='on_this_day_max_temp_trend', component_property='figure'),
    Input(component_id='datepicker_day_of_year', component_property='value'),
)
def on_this_day_max_temp_trend(date_value):

  date_formatted = datetime.strptime(date_value+" 2024", "%b %d %Y")
  day_of_year = gen_DoY_index(pd.to_datetime(date_formatted))
  df = temps.query(f"day_of_year == {day_of_year}")

  fig = px.bar(df, x="year", y='max_temp', color_discrete_sequence=['firebrick'], text_auto=True)
  fig.update_layout(xaxis_title="Year", yaxis_title=f"{date_value} Maximum Temperature", bargap=0.15)
  return fig


@callback(
    Output(component_id='on_this_day_cloud_cover_trend', component_property='figure'),
    Input(component_id='datepicker_day_of_year', component_property='value'),
)
def on_this_day_cloud_cover_trend(date_value):

  date_formatted = datetime.strptime(date_value+" 2024", "%b %d %Y")
  day_of_year = gen_DoY_index(pd.to_datetime(date_formatted))
  df = temps.query(f"day_of_year == {day_of_year}")

  fig = px.bar(df, x='year', y="cloud_cover", color_discrete_sequence=['cornflowerblue'], text_auto=True)
  fig.update_layout(xaxis_title="Year", yaxis_title=f"{date_value} Cloud Cover", bargap=0.15)

  return fig


@callback(
    Output(component_id='on_this_day_precip_trend', component_property='figure'),
    Input(component_id='datepicker_day_of_year', component_property='value'),
)
def on_this_day_precip_trend(date_value):

  date_formatted = datetime.strptime(date_value+" 2024", "%b %d %Y")
  day_of_year = gen_DoY_index(pd.to_datetime(date_formatted))
  df = precip.query(f"day_of_year == {day_of_year}").copy()

  fig = px.bar(df, x='year', y=['rain', 'snow'], color_discrete_sequence=['limegreen', 'cyan'], text_auto=True)
  fig.update_layout(xaxis_title="Year", yaxis_title=f"{date_value} Precipitation", bargap=0.15)

  return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8053)


# On this day view
# Normal High
# Normal Low
# Record High (year)
# Record Low (year)
# % cloud cover
# Most Rain
# Most Snow

# Trend view over time for min, max, cloud cover, precip

