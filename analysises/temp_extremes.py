import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

csv_subdir = 'weather/data_sources/'
full_data = pd.read_csv(csv_subdir+'/daily_weather.csv', index_col=False)


# Daily Plots
def generate_plot(df, x, y, color, title=''):
  plt.figure(figsize=(14, 7))
  sns.barplot(x=x, y=y, data=df, color=color, edgecolor='black')
  plt.title(title)
  plt.xticks(rotation=45, ha='right')
  overall_mean = df[y].mean()
  plt.axhline(overall_mean, color='black', linestyle='--', linewidth=0.5)
  # Add label directly on the graph near the mean line
  ax = plt.gca()
  # Place label at the right edge of the plot, aligned with the mean line
  ax.text(
      x=ax.get_xlim()[0]*0.9,
      y=overall_mean,
      s=f'Average: {overall_mean:.1f}',
      color='black',
      va='bottom',
      ha='left',
      fontsize=9,
      bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
  )
  plt.tight_layout()
  plt.show()

temperatures = full_data[['date', 'min_temp', 'max_temp', 'avg_temp']].copy()
temperatures = temperatures[temperatures['date']>='1990-01-01'] # At least a 30 year avg
temperatures['date'] = pd.to_datetime(temperatures['date'])
temperatures['year'] = temperatures['date'].dt.year
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
temperatures['month'] = pd.Categorical(temperatures['date'].dt.strftime('%b'), categories=month_order, ordered=True)

# Coldest days
temperatures.sort_values(by='min_temp')[['date', 'min_temp']].head(10)

# Warmest days
temperatures.sort_values(by='max_temp', ascending=False)[['date', 'max_temp']].head(10)

# Greatest minimum
temperatures.sort_values(by='min_temp', ascending=False)[['date', 'min_temp']].head(10)

# Coldest Maximum
temperatures.sort_values(by='max_temp')[['date', 'max_temp']].head(10)

# Warmest and coldest by year
yearly_extremes = (
    temperatures
    .groupby('year')
    .agg({'max_temp': 'max', 'min_temp': 'min'})
    .reset_index()
)

# Find the date for each year's max and min temperature
yearly_extremes['date_max_temp'] = yearly_extremes.apply(
    lambda row: temperatures[(temperatures['year'] == row['year']) & (temperatures['max_temp'] == row['max_temp'])]['date'].iloc[0],
    axis=1
)
yearly_extremes['date_min_temp'] = yearly_extremes.apply(
    lambda row: temperatures[(temperatures['year'] == row['year']) & (temperatures['min_temp'] == row['min_temp'])]['date'].iloc[0],
    axis=1
)
yearly_extremes

# Warmest/coldest by month
monthly_extremes = (
    temperatures
    .groupby('month')
    .agg({'max_temp': 'max', 'min_temp': 'min'})
    .reset_index()
    .sort_values(by='month')
)
monthly_extremes

# Greatest/Least temperature range in a single day
temp_swing = (
    temperatures
    .assign(temp_swing=lambda x: x['max_temp'] - x['min_temp'])
    .sort_values(by='temp_swing', ascending=False)
    .drop(columns=['year', 'month', 'avg_temp'])
)
temp_swing.head(10)
temp_swing.sort_values(by='temp_swing', ascending=True).head(10)

# Number of 100+ degree days by year
deg100 = (
    temperatures
    .assign(above_100=lambda x: x['max_temp'] >= 100.0)
    .groupby('year')
    .agg({'above_100': 'sum'})
    .reset_index()
    .sort_values(by='year')
)
generate_plot(deg100, x='year', y='above_100', color='red', title='100 degree days')

# Number of days below 10 in a year
below_10 = (
    temperatures
    .assign(below_10=lambda x: x['min_temp'] < 10)
    .groupby('year')
    .agg({'below_10': 'sum'})
    .reset_index()
    .sort_values(by='year')
)
generate_plot(below_10, x='year', y='below_10', color='royalblue', title='Days Below 10 Degrees')

# Avg high temp per year
avg_temp_per_year = (
    temperatures[temperatures['date'] <= '2024-12-31']
    .groupby('year')
    .agg({'max_temp': 'mean', 'min_temp': 'mean', 'avg_temp': 'mean'})
    .reset_index()
)

generate_plot(avg_temp_per_year, x='year', y='max_temp', color='red', title='Average Temperature Per Year')


# Create a 1x3 subplot
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=False)

# Define the metrics to plot
metrics = ['max_temp', 'min_temp', 'avg_temp']

for i, metric in enumerate(metrics):
  X = sm.add_constant(avg_temp_per_year['year']-min(avg_temp_per_year['year']))
  y = avg_temp_per_year[metric]
  model = sm.OLS(y, X).fit()
  slope = model.params.year
  signif = model.pvalues.year

  print(model.summary())

  # Plot with OLS trend line and slope label
  ax = sns.regplot(
      data=avg_temp_per_year,
      x='year',
      y=metric,
      scatter=True,
      color='black',
      line_kws={'label': f"Trend (slope={slope:.3f}°F/yr, {signif:.3f})", 'color': 'red', 'lw': 1.5},
      ci=None,
      ax=axes[i],
  )
  axes[i].legend()
  axes[i].set_title(f'Salt Lake City {metric.replace("_", " ").capitalize()}')
  axes[i].set_xlabel('Year')
  axes[i].set_ylabel(f'{metric.replace("_", " ").capitalize()} (°F)')
plt.tight_layout()
plt.show()

# Very interesting in that the minimums are increasing at a faster rate than the maximums
# min temp is around 1.3°F/decade
# max temp is around 0.7°F/decade
# Overall avg is increasing around 1°F/decade