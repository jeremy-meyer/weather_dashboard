import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from patsy import dmatrix
import statsmodels.api as sm

hourly_raw = pd.read_csv('weather/data_sources/hourly_weather.csv')
hourly_raw['timestamp'] = pd.to_datetime(hourly_raw['timestamp'])
hourly_raw['year'] = pd.to_datetime(hourly_raw['timestamp']).dt.year
hourly_raw['month'] = pd.to_datetime(hourly_raw['timestamp']).dt.strftime('%b')
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
hourly_raw['month'] = pd.Categorical(hourly_raw['month'], categories=month_order, ordered=True)

season_map = {
    'winter': ['Dec', 'Jan', 'Feb'],
    'spring': ['Mar', 'Apr', 'May'],
    'summer': ['Jun', 'Jul', 'Aug'],
    'fall': ['Sep', 'Oct', 'Nov']
}
# Create new season_map column in hourly_raw
hourly_raw['season'] = hourly_raw['month'].map({v: k for k, vs in season_map.items() for v in vs})

hourly_model = hourly_raw[['timestamp', 'temp', 'day', 'year', 'month', 'season']].copy()
hourly_model['time_index'] = (
  (hourly_model['timestamp'] - hourly_model['timestamp'].min()).dt.total_seconds() / (60*60*24)
)
hourly_model['year_index'] = (
  hourly_model['time_index'] % 365.25
)
hourly_model['hourly_index'] = (
  hourly_model['timestamp'].dt.hour
)


basis_hourly = dmatrix(
    f"1 + time_index + cc(year_index, df=6, constraints='center', upper_bound=365.25) + month:cc(hourly_index, df=9, constraints='center', upper_bound=24)",
    {"time_index": hourly_model['time_index'], "year_index": hourly_model['year_index'], "hourly_index": hourly_model['hourly_index'], 'month': hourly_model['month']},
    return_type='dataframe'
)

# basis_hourly2 = dmatrix(
#     f"1 + time_index + cc(year_index, df=6, constraints='center', upper_bound=365.25) + month:C(hourly_index)",
#     {"time_index": hourly_model['time_index'], "year_index": hourly_model['year_index'], "hourly_index": hourly_model['hourly_index'], 'month': hourly_model['month']},
#     return_type='dataframe'
# )


model = sm.OLS(hourly_model['temp'], basis_hourly).fit()
model.summary()
hourly_model['pred_temp'] = model.predict(basis_hourly)

# model2 = sm.OLS(hourly_model['temp'], basis_hourly2).fit()
# model2.summary()
# hourly_model['pred_temp'] = model.predict(basis_hourly)

data = hourly_model[hourly_model['day'].between('2025-07-01', '2025-08-15')]
sns.lineplot(
  data=data,
  x='timestamp',
  y='temp',
)
sns.lineplot(
  data=data,
  x='timestamp',
  y='pred_temp',
)

plt.xticks(rotation=45)
plt.show()

sns.lineplot(
  data=hourly_raw[hourly_raw['day'].between('2025-07-01', '2025-07-31')],
  x='timestamp',
  y='dew_point',
)
plt.show()

# Compare time of day seasonality for each month
# Additive decomposition
# Multiply each term in the design matrix by its coefficient
coefficients = model.params
hourly_model['time_trend'] = basis_hourly['time_index'] * coefficients['time_index']
hourly_model['year_seasonality'] = (
    basis_hourly.filter(like='cc(year_index').dot(coefficients.filter(like='cc(year_index'))
)

hourly_model['hourly_seasonality'] = (
    basis_hourly.filter(like='(hourly_index').dot(coefficients.filter(like='(hourly_index'))
)

hourly_model['intercept'] = coefficients['Intercept']

# Calculate the total predicted temperature as the sum of all components
hourly_model['pred_temp_decomposed'] = (
    hourly_model['intercept'] +
    hourly_model['time_trend'] +
    hourly_model['year_seasonality'] +
    hourly_model['hourly_seasonality']
)

# Verify that the decomposed prediction matches the model's prediction
assert np.allclose(hourly_model['pred_temp'], hourly_model['pred_temp_decomposed'])

# Compare hourly effects across month
hourly_effects = hourly_model.groupby(['month', 'hourly_index']).agg(
    hourly_seasonality=('hourly_seasonality', 'mean')
).reset_index()

hottest_hours = hourly_effects.loc[hourly_effects.groupby('month')['hourly_seasonality'].idxmax()]
coldest_hours = hourly_effects.loc[hourly_effects.groupby('month')['hourly_seasonality'].idxmin()]

# Plot the hourly_effects for each month as separate lines
plt.figure(figsize=(12, 6))
sns.lineplot(data=hourly_effects, x='hourly_index', y='hourly_seasonality', hue='month')
sns.scatterplot(
    data=hottest_hours,
    x='hourly_index',
    y='hourly_seasonality',
    color='red',
    legend=False,
    s=25,  # Size of the dots
    marker='o'
)
sns.scatterplot(
    data=coldest_hours,
    x='hourly_index',
    y='hourly_seasonality',
    color='blue',
    legend=False,
    s=25,  # Size of the dots
    marker='o'
)
plt.title('Average Hourly Temperature by Month')
plt.xlabel('Hour of Day')
plt.ylabel('Temperature Deviation (F)')
plt.xticks(range(24))
plt.legend(title='Month', bbox_to_anchor=(1.115, 1), loc='upper right')
plt.show()

# Super interesting! Winter months have a less pronounced hourly seasonality effect compared to summer months
# Coldest time of day is around 7am in Summertime, warmest is around 5pm
# In Winter, the coldest time can be as late as 8am, warmest can be around 3pm
hottest_hours
coldest_hours
# Timezone is GMT-6 - 1hr offset
# Why does the .fit() produce a large condition number?

hourly_model[hourly_model['day']=='2024-03-10']


# Todo: Find params of best fit
# Other hourly patterns: rain, snow, humidity
hourly_means_overall = (
  hourly_raw
  .assign(hour=hourly_raw['timestamp'].dt.hour)
  .groupby(['hour'])
  .agg(
    precip=('precip', 'mean'),
    snow=('snow', 'mean'),
    humidity=('humidity', 'mean'),
    wind=('wind_speed', 'mean'),
    clouds=('cloud_cover', 'mean'),
    pressure=('pressure', 'mean'),
    temp=('temp', 'mean'),
    dew=('dew_point', 'mean')
  )
  .reset_index()
)

hourly_means_seasons = (
  hourly_raw
  .assign(hour=hourly_raw['timestamp'].dt.hour)
  .groupby(['hour', 'season'])
  .agg(
    precip=('precip', 'mean'),
    snow=('snow', 'mean'),
    humidity=('humidity', 'mean'),
    wind=('wind_speed', 'mean'),
    clouds=('cloud_cover', 'mean'),
    pressure=('pressure', 'mean'),
    temp=('temp', 'mean'),
    dew=('dew_point', 'mean')
  )
  .reset_index()
)


# Create a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# Flatten the axes array for easier iteration
axes = axes.flatten()
colors = {
    'precip': 'darkgreen',
    'snow': 'lightblue',
    'dew': 'limegreen',
    'wind': 'purple',
    'clouds': 'gray',
    'pressure': 'red',
}
# Plot each metric in a separate subplot
for i, metric in enumerate(colors):
    sns.barplot(data=hourly_means_overall, x='hour', y=metric, ax=axes[i], color=colors[metric])
    ax_temp = axes[i].twinx()
    ax_temp.set_ylim(bottom=0, top=hourly_means_overall['temp'].max() * 1.1)
    sns.lineplot(data=hourly_means_overall, x='hour', y='temp', ax=ax_temp, color='black', linewidth=1, linestyle='--')
    axes[i].set_title(f'Average {metric.capitalize()} by Hour')
    axes[i].set_xlabel('Hour of Day')
    axes[i].set_ylabel(metric.capitalize())
plt.tight_layout()
plt.show()


# Create a 2x3 grid of subplots for seasons
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# Flatten the axes array for easier iteration
axes = axes.flatten()

# Plot each metric in a separate subplot
for i, metric in enumerate(colors):
    sns.lineplot(data=hourly_means_seasons, x='hour', y=metric, ax=axes[i], hue='season', legend=False)
    axes[i].set_title(f'Average {metric.capitalize()} by Hour')
    axes[i].set_xlabel('Hour of Day')
    axes[i].set_ylabel(metric.capitalize())

# Create dummy plot to extract legend info
dummy_fig, dummy_ax = plt.subplots()
handles, labels = sns.lineplot(
    data=hourly_means_seasons,
    x='hour',
    y=metric,
    hue='season',
    ax=dummy_ax
).get_legend_handles_labels() # Get handles and labels from a dummy plot
plt.close(dummy_fig)

fig.legend(
    handles,
    labels,
    title='Season',
    loc='upper right',
    bbox_to_anchor=(0.99, 0.975),
)

# Adjust layout
plt.tight_layout(rect=[0,0,0.92,1])  # Leave space for the shared legend
plt.show()

# Todo: Find optimal df of hourly model
# Adjust for DST?

