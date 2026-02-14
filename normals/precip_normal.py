import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.shared_functions import *

precip = pd.read_csv('output_sources/precip_table.csv', index_col=False, parse_dates=['date'])
precip['month'] = pd.Categorical(precip['date'].dt.strftime('%b'), categories=month_order, ordered=True)

N_years = 30
current_year = precip['year'].max()
max_water_year = precip['water_year'].max()
max_winter_year = precip['snow_season'].max()
max_date = precip['date'].max()

# Calculate normals from the last complete decade
normal_start_year, normal_end_year = calc_normal_years(current_year, N_years)
normal_start_water_year, normal_end_water_year = calc_normal_years(max_water_year, N_years)
print(f"Calculating normals from {normal_start_year} to {normal_end_year}")

# Unpivot different precip year types
precip_data_unpivot = pd.concat([
  precip\
    .assign(
      year_type='calendar_year', 
      current_year=str(current_year), 
      year_for_dash=precip['year'],
      norm_range=precip['year'].between(normal_start_year, normal_end_year),
    ),
  precip\
    .assign(
      year_type='water_year', 
      current_year=str(max_water_year), 
      year_for_dash=precip['water_year'],
      norm_range=precip['water_year'].between(normal_start_water_year, normal_end_water_year),
    ),
  precip\
    .assign(
      year_type='snow_season', 
      current_year=max_winter_year, 
      year_for_dash=precip['snow_season'],
      norm_range=precip['snow_season'].between(
        offset_season(max_winter_year, - (current_year - normal_start_year)), 
        offset_season(max_winter_year, -(current_year - normal_end_year))
      ),
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
  precip_data_unpivot
)

# Monthly Precip normals
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

monthly_normals.to_csv('output_sources/precip_monthly_normals.csv', index=False)

# Compute Monthly Rank
if max_date != max_date + pd.offsets.MonthEnd(0):
  most_recent_month = max_date.replace(day=1) - pd.Timedelta(days=1)
else:
  most_recent_month = max_date

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

# Month to date Precip
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
mtd_avg.to_csv("output_sources/precip_mtd_normals.csv", index=False)

# Year to date precip
ytd = (
  precip_data_for_norm
  .fillna({'metric_value': 0})
  .sort_values(by=['year_type', 'metric_name', 'year_for_dash', 'day_of_year_dash'])
  .groupby(['year_type', 'metric_name', 'norm_range' ,'year_for_dash'], observed=True)
  .apply(lambda x: x.assign(year_to_date_precip=x['metric_value'].cumsum()))
  .reset_index(drop=True)
  .sort_values(by=['year_type', 'metric_name', 'year_for_dash', 'day_of_year_dash'])
)

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
ytd_avg.sort_values(by=['year_type', 'metric_name', 'day_of_year_dash'], inplace=True)
ytd_avg.to_csv("output_sources/precip_ytd_normals.csv", index=False)

# END OF TABLE GENERATION

# Example Chart QA
calendar_type = 'water_year'
metric_name = 'precip'

current_year_ytd = ytd.query("year_for_dash == current_year").query(f"year_type == '{calendar_type}'").query(f"metric_name == '{metric_name}'").sort_values(by='date')
ytd_data = ytd.query(f"(year_for_dash != current_year) & (year_type == '{calendar_type}') & (metric_name == '{metric_name}')").sort_values(by=['year_for_dash','date'])
ytd_avg_data = ytd_avg.query(f"year_type == '{calendar_type}' & metric_name == '{metric_name}'").sort_values(by='day_of_year_dash')
chart_label = current_year_ytd['current_year'].head().iloc[0]

fig = px.line(ytd_data, x='day_of_year_dash', y='year_to_date_precip', color='year_for_dash', title='Year to Date Precipitation by Year')
fig.update_traces(
    line=dict(color='rgb(100, 200, 100, 0.1)', width=0.75, dash='dot'),
    selector=dict(mode='lines')  # Ensure it applies only to line traces
)
fig.add_scatter(x=current_year_ytd['day_of_year_dash'], y=current_year_ytd['year_to_date_precip'], mode='lines', name=chart_label, line=dict(color='cyan', width=2))
fig.add_scatter(x=ytd_avg_data['day_of_year_dash'], y=ytd_avg_data['avg_precip_ytd'], mode='lines', name='Average', line=dict(color='green', width=4))
fig.add_traces([
    go.Scatter(
        x=list(ytd_avg_data['day_of_year_dash']) + list(ytd_avg_data['day_of_year_dash'])[::-1],
        y=list(ytd_avg_data['p75_precip_ytd']) + list(ytd_avg_data['p25_precip_ytd'])[::-1],
        fill='toself',
        fillcolor='rgba(0, 150, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='25th-75th Percentile',
    )
])
fig.add_traces([
    go.Scatter(
        x=list(ytd_avg_data['day_of_year_dash']) + list(ytd_avg_data['day_of_year_dash'])[::-1],
        y=list(ytd_avg_data['p90_precip_ytd']) + list(ytd_avg_data['p10_precip_ytd'])[::-1],
        fill='toself',
        fillcolor='rgba(0, 150, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='10th-90th Percentile',
    )
])
fig.update_layout(template="plotly_dark")
fig.show(renderer="browser")