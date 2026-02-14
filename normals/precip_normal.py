import plotly.express as px
import plotly.graph_objects as go
from patsy import dmatrix
from statsmodels import api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import numpy as np
from patsy import dmatrix
import statsmodels.api as sm

# STEP 1: Clean table and engineer features-----------------------------------

# Subset relevant data
precip_raw = pd.read_csv('weather/data_sources/noaa_precip.csv', index_col=False, parse_dates=['date'])
precip = precip_raw[['date','precip' ,'snow', 'snow_depth', 'tempmin', 'tempmax']].copy()
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
precip['month'] = pd.Categorical(precip['date'].dt.strftime('%b'), categories=month_order, ordered=True)
precip['date'] = pd.to_datetime(precip['date'])
precip['year'] = precip['date'].dt.year
precip['precip_day'] = (precip['precip'] > 0).astype(int)

# Define snow season as Aug - July (I know water season is Oct - Sep, but we have had snow in Sep and I would rather count that in next season)
precip['snow_season'] = np.where(precip['month'].isin(['Aug', 'Sep', 'Oct', 'Nov', 'Dec']), precip['year'], precip['year'] - 1)
precip['snow_season'] = precip['snow_season'].astype(str) + '-' + (precip['snow_season'] + 1).astype(str)

precip['water_year'] = np.where(precip['month'].isin(['Oct', 'Nov', 'Dec']), precip['year']+1, precip['year'])

def gen_DoY_index(x, year_type='calendar_year'):
  excess = {
      'calendar_year': 0,
      'water_year': 92,
      'snow_season': 153,
  }[year_type]
  
  mar1_day = 60
  return_value = x.dayofyear

  if x.is_leap_year:
    if x.dayofyear > mar1_day:
      return_value -= 1 
    elif x.dayofyear == mar1_day:
      return_value -= 0.5

  return (return_value + excess - 1) % 365 + 1


def dayofyear_to_month_day(doy):
  dt = pd.Timestamp(f"2025-01-01") + pd.Timedelta(days=doy-1)
  if doy==59.5:
    return "Feb 29"
  return dt.strftime('%b %d')

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

precip.to_csv('weather/output_sources/precip_table.csv', index=False)

# STEP 2: Calculate Normals --------------------------------------
# Calculate daily normals and monthly normals.
N_years = 30
current_year = precip['year'].max()
max_water_year = precip['water_year'].max()
max_winter_year = precip['snow_season'].max()

def offset_season(s, offset):
  return str(int(s.split('-')[0]) + offset) + '-' + str(int(s.split('-')[1]) + offset)

precip = pd.read_csv('weather/output_sources/precip_table.csv', index_col=False, parse_dates=['date'])
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

precip_data_for_norm = (
  precip_data_unpivot#.query("(norm_range) | (current_year == year_for_dash)")
)


monthly_normals = (
    precip[precip['year'].between(current_year - N_years, current_year-1)]
    .groupby(['year', 'month'])
    .agg({
        'precip': 'sum',
        'snow': 'sum',
        'rain': 'sum',
        'precip_day': 'mean',
    })
    .reset_index()
    .groupby('month')
    .agg(
        norm_precip=('precip', 'mean'),
        norm_snow=('snow', 'mean'),
        norm_rain=('rain', 'mean'),
        norm_precip_perc=('precip_day', 'mean'),
    )
    .reset_index()
)

monthly_normals.to_csv('weather/output_sources/monthly_precip_normals.csv', index=False)


# Precip rank
max_date = precip['date'].max()
if max_date != max_date + pd.offsets.MonthEnd(0):
  most_recent_month = max_date.replace(day=1) - pd.Timedelta(days=1)
else:
  most_recent_month = max_date
# Create monthly_totals

precip_data_unpivot['current_year'] = precip_data_unpivot['current_year'].astype(str)
precip_data_unpivot['year_for_dash'] = precip_data_unpivot['year_for_dash'].astype(str)

monthly_totals = (
    precip_data_unpivot
    .query(f"date <= '{most_recent_month}'")
    .groupby(['year_type', 'metric_name', 'year_for_dash', 'month'], observed=True)
    .agg(
        total=('metric_value', 'sum'),
    )
    .reset_index()
)

monthly_totals['precip_rank'] = (
    monthly_totals
    .groupby(['year_type', 'metric_name', 'month'])['total']
    .rank(ascending=False, method='min').astype(int)
)

# Month to date
precip_data_for_norm = precip_data_for_norm.assign(
  day_of_month=precip_data_for_norm['date'].dt.day, 
)

mtd = (
  precip_data_for_norm.query("(year_type == 'calendar_year')")\
    [['date', 'month', 'year_for_dash', 'current_year', 'metric_name', 'metric_value', 'day_of_month', 'norm_range']]
  .fillna({'metric_value': 0})
  .sort_values(by=['metric_name', 'year_for_dash' ,'date'])
  .groupby(['metric_name', 'year_for_dash', 'norm_range', 'month'], observed=True)
  .apply(lambda x: x.assign(month_to_date_precip=x['metric_value'].cumsum()))
  .reset_index(drop=True)
  # .sort_values(by=['year_type', 'metric_name', 'calendar_year', 'date'])
)


mtd_avg = (
  mtd
  .query("year_for_dash != current_year")
  .assign(month_to_date_precip_norm=lambda x: np.where(x['norm_range'], x['month_to_date_precip'], np.nan))
  .groupby(['metric_name', 'month', 'day_of_month'], observed=True)
  .agg(
      avg_precip_mtd=('month_to_date_precip_norm', 'mean'),
      min_precip_mtd=('month_to_date_precip', 'min'),
      max_precip_mtd=('month_to_date_precip', 'max'),
      p10_precip_mtd=('month_to_date_precip_norm', lambda x: np.nanpercentile (x, 10)),
      p25_precip_mtd=('month_to_date_precip_norm', lambda x: np.nanpercentile (x, 25)),
      p50_precip_mtd=('month_to_date_precip_norm', lambda x: np.nanpercentile (x, 50)),
      p75_precip_mtd=('month_to_date_precip_norm', lambda x: np.nanpercentile (x, 75)),
      p90_precip_mtd=('month_to_date_precip_norm', lambda x: np.nanpercentile (x, 90)),
  )
  .reset_index()
  # .query(f"year_type == '{calendar_type}'") # Uncomment to run
)

mtd_avg.sort_values(by=['metric_name', 'month' ,'day_of_month'], inplace=True)
mtd_avg.to_csv("weather/output_sources/mtd_precip_normals.csv", index=False)


ytd = (
  precip_data_for_norm
  .fillna({'metric_value': 0})
  .sort_values(by=['year_type', 'metric_name', 'year_for_dash', 'day_of_year_dash'])
  .groupby(['year_type', 'metric_name', 'norm_range' ,'year_for_dash'], observed=True)
  .apply(lambda x: x.assign(year_to_date_precip=x['metric_value'].cumsum()))
  .reset_index(drop=True)
  .sort_values(by=['year_type', 'metric_name', 'year_for_dash', 'day_of_year_dash'])
)

current_year_ytd = ytd.query("year_for_dash == current_year")
# chart_label = current_year_ytd['current_year'].head().iloc[0]
incomplete_years = (
  precip_data_for_norm
  .groupby(['year_type', 'year_for_dash'])
  .agg(min_index=('day_of_year_dash', 'min'))
  .reset_index().query("min_index!=1")
  [['year_type', 'year_for_dash']]
  .values.tolist()
)


ytd_avg = (
  ytd
  [~(ytd['year_type'] + ytd['year_for_dash'].astype(str)).isin(["".join(item) for item in incomplete_years])]
  .query("year_for_dash != current_year")
  .assign(
    year_to_date_precip_norm=lambda x: np.where(x['norm_range'], x['year_to_date_precip'], np.nan)
  )
  .groupby(['year_type', 'metric_name', 'day_of_year', 'day_of_year_dash'])
  .agg(
      avg_precip_ytd=('year_to_date_precip_norm', 'mean'),
      min_precip_ytd=('year_to_date_precip', 'min'),
      max_precip_ytd=('year_to_date_precip', 'max'),
      p10_precip_ytd=('year_to_date_precip_norm', lambda x: np.nanpercentile (x, 10)),
      p25_precip_ytd=('year_to_date_precip_norm', lambda x: np.nanpercentile(x, 25)),
      p50_precip_ytd=('year_to_date_precip_norm', lambda x: np.nanpercentile(x, 50)),
      p75_precip_ytd=('year_to_date_precip_norm', lambda x: np.nanpercentile(x, 75)),
      p90_precip_ytd=('year_to_date_precip_norm', lambda x: np.nanpercentile(x, 90)),
  )
  .reset_index()
  .query("day_of_year % 1 == 0") # get rid if leap day
  # .query(f"year_type == '{calendar_type}'") # Uncomment to run
)

ytd_avg['dashboard_date'] = ytd_avg['day_of_year'].apply(dayofyear_to_month_day)
ytd_avg.sort_values(by=['year_type', 'metric_name', 'day_of_year'], inplace=True)
ytd_avg.to_csv("weather/output_sources/ytd_precip_normals.csv", index=False)

# Chart
fig = px.line(ytd.query(f"(year != current_year) & (year_type == '{calendar_type}')"), x='day_of_year', y='year_to_date_precip', color='year', title='Year to Date Precipitation by Year')
fig.update_traces(
    line=dict(color='rgb(0, 100, 0, 0.5)', width=0.75, dash='dot'),
    selector=dict(mode='lines')  # Ensure it applies only to line traces
)
fig.add_scatter(x=current_year_ytd['day_of_year'], y=current_year_ytd['year_to_date_precip'], mode='lines', name=chart_label, line=dict(color='black', width=2))
fig.add_scatter(x=ytd_avg['day_of_year'], y=ytd_avg['avg_precip_ytd'], mode='lines', name='Average', line=dict(color='green', width=4))
fig.add_traces([
    go.Scatter(
        x=list(ytd_avg['day_of_year']) + list(ytd_avg['day_of_year'])[::-1],
        y=list(ytd_avg['p75_precip_ytd']) + list(ytd_avg['p25_precip_ytd'])[::-1],
        fill='toself',
        fillcolor='rgba(0, 150, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='25th-75th Percentile',
    )
])

fig.show(renderer="browser")

metric = 'precip'  # Change as needed
month = 'Oct'  # Change as needed

current_year_mtd = mtd.query(f"year_for_dash == current_year & metric_name == '{metric}' & month == '{month}'")
current_year_label = current_year_mtd['current_year'].head().iloc[0]
mtd_avg_metric = mtd_avg.query(f"metric_name == '{metric}' & month == '{month}'")

fig = go.Figure()
fig.add_scatter(
  x=current_year_mtd['day_of_month'], 
  y=current_year_mtd['month_to_date_precip'], 
  mode='lines', name=month + ' ' + str(current_year_label), 
  line=dict(color='black', width=2)
)
fig.add_scatter(
  x=mtd_avg_metric['day_of_month'], 
  y=mtd_avg_metric['avg_precip_mtd'], 
  mode='lines', 
  name='Average', 
  line=dict(color='green', width=4)
)
fig.add_scatter(
  x=mtd_avg_metric['day_of_month'], 
  y=mtd_avg_metric['max_precip_mtd'], 
  mode='lines', 
  name='Maximum', 
  line=dict(color='green', width=1, dash='dot')
)
fig.add_scatter(
  x=mtd_avg_metric['day_of_month'], 
  y=mtd_avg_metric['min_precip_mtd'], 
  mode='lines', 
  name='Minimum', 
  line=dict(color='green', width=1, dash='dot')
)
fig.add_traces([
    go.Scatter(
        x=list(mtd_avg_metric['day_of_month']) + list(mtd_avg_metric['day_of_month'])[::-1],
        y=list(mtd_avg_metric['p75_precip_mtd']) + list(mtd_avg_metric['p25_precip_mtd'])[::-1],
        fill='toself',
        fillcolor='rgba(0, 150, 0, 0.2)',
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
        fillcolor='rgba(0, 150, 0, 0.1)',
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
        fillcolor='rgba(0, 150, 0, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='10th-90th Percentile',
        showlegend=False
    )
])
fig.show(renderer="browser")