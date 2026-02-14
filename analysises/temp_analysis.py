from patsy import dmatrix
import statsmodels.api as sm
import pandas as pd
from numpy import mean
from matplotlib import pyplot as plt
from datetime import date
import seaborn as sns
import calplot
from matplotlib.colors import LinearSegmentedColormap


# Config
N_years = 30
current_year = 2025

# Subset relevant data
full_data = pd.read_csv('weather/data_sources/'+'daily_weather.csv', index_col=False)
temperatures = full_data[['date', 'avg_temp', 'min_temp', 'max_temp']].copy()
temperatures['date'] = pd.to_datetime(temperatures['date'])
temperatures['year'] = temperatures['date'].dt.year
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
temperatures['month'] = pd.Categorical(temperatures['date'].dt.strftime('%b'), categories=month_order, ordered=True)

# Create 1/2 index for leap year so we can intepret "hottest day of year" correctly
# This will put Feb 29th between 2-28 and 3-1 in the model
def gen_DoY_index(x):
  if x.is_leap_year:
    if x.dayofyear > 60:
      return x.dayofyear - 1
    elif x.dayofyear==60:
      return x.dayofyear - 0.5
  return x.dayofyear

# Inverse DoY function (year is not a leap year)
def dayofyear_to_month_day(doy):
  dt = pd.Timestamp(f"2025-01-01") + pd.Timedelta(days=doy-1)
  if doy==59.5:
    return "Feb 29"
  return dt.strftime('%b %d')

# Create Day of Year Index Column
temperatures['day_of_year'] = temperatures['date'].apply(gen_DoY_index)
data_for_normal = temperatures.loc[temperatures['year'].between(current_year - N_years, current_year-1)].copy()
data_for_normal['date_formatted'] = data_for_normal['date'].dt.strftime('%b-%d')

# MODEL SECTION --------------------------------------------------------------
# Use cyclic spline so endpoints line up and are continous
# Set upper bound to 366 so Jan 1 and Dec 31 have different predicted values
basis_normal = dmatrix(
    "1 + cc(day_of_year, df=5, constraints='center', upper_bound=366)",
    {"day_of_year": data_for_normal['day_of_year'],},
    return_type='dataframe'
)

# Predict fitted values
model_norm = sm.OLS(data_for_normal['max_temp'], basis_normal).fit()
model_norm.summary()
data_for_normal['initial_prediction'] = round(model_norm.predict(basis_normal),2)

# Initial Fit
sns.scatterplot(data_for_normal, x='day_of_year', y='max_temp', legend=None, alpha=0.4, color='red', size=3)
sns.lineplot(data_for_normal[data_for_normal['year']==2024], x='day_of_year', y='initial_prediction', color='darkred', linewidth=2)
plt.title('Average High Temperatures')
plt.show()

# Model with all data
basis_normal = dmatrix(
    "1 + cc(day_of_year, df=5, constraints='center', upper_bound=366)",
    {"day_of_year": data_for_normal['day_of_year'],},
    return_type='dataframe'
)

# CV: Find optimal knots for cyclic cubic spline
# To ensure equal number of points per day, Plan: 30-fold CV by year
# Goal: Pick smallest df where the error is minimized
# Errors tend to taper off after df=5, so this is likley limit before overfitting

cv_metric = 'min_temp'
dfs = range(2,25)
df_errors = dict()

for df in dfs:
  errors_all_years = []
  for y in data_for_normal['year'].unique():
    cv_data = data_for_normal[data_for_normal['year']!=y].copy()
    test_data = data_for_normal[data_for_normal['year']==y].copy()

    basis_cv = dmatrix(
      f"1 + cc(day_of_year, df={df}, constraints='center', upper_bound=366)",
      {"day_of_year": cv_data['day_of_year'],},
      return_type='dataframe'
    )

    basis_test = dmatrix(
      f"1 + cc(day_of_year, df={df}, constraints='center', upper_bound=366)",
      {"day_of_year": test_data['day_of_year'],},
      return_type='dataframe'
    )
      
    model_cv = sm.OLS(cv_data['max_temp'], basis_cv).fit()
    test_data['test_prediction'] = round(model_cv.predict(basis_test,2))
    test_data['error'] = test_data['test_prediction'] - test_data[cv_metric]
    mae = mean(abs(test_data['error']))
    # mse = mean(test_data['error']**2)
    errors_all_years.append(mae)
  
  df_errors[df] = mean(errors_all_years)
  print(f"FInished {df}")
pd.DataFrame(df_errors.items(), columns=['df', 'error']).sort_values('error')

plt.plot(df_errors.keys(), df_errors.values(), color='skyblue')
plt.xlabel('Degrees of Freedom (df)')
plt.ylabel('Mean Squared Error')
plt.title(f'Spline Degrees of Freedom vs. Mean Squared Error {cv_metric}')
plt.tight_layout()
plt.show()


# Check fits with other knots
# Note: Fit will be the same as with full data, but it easier to see with N=1 instead of 30
alt_model_input = (
  data_for_normal
  .groupby(['day_of_year'])
  .agg(
    max_temp=('max_temp', 'mean'),
  )
  .reset_index()
)

knots_to_display = [5,13]
for k in knots_to_display:
  basis = dmatrix(
    f"1 + cc(day_of_year, df={k}, constraints='center', upper_bound=366)",
    {"day_of_year": alt_model_input['day_of_year'],},
    return_type='dataframe'
  )
  model = sm.OLS(alt_model_input[cv_metric], basis).fit()
  alt_model_input[f'prediction_{k}'] = round(model.predict(basis),2)


alt_model_input.sort_values('prediction_5', ascending=False)
colors = ['darkred', 'orange', 'gold', 'darkyellow', 'green']
sns.scatterplot(alt_model_input, x='day_of_year', y=cv_metric, legend=None, alpha=0.5, color='red', size=3)
for i, k in enumerate(knots_to_display):
  sns.lineplot(alt_model_input, x='day_of_year', y=f'prediction_{k}', color=colors[i], linewidth=2, label=f'df={k}')
plt.title(f'Average {cv_metric}')
plt.show()

# Results:
# DF=13 for max_temp
# DF=6 for min_temp
# DF=6 for avg_temp

# "Normals" Model inference -------------------------
def create_model(metric, df):
  basis_normal = dmatrix(
      f"1 + cc(day_of_year, df={df}, constraints='center', upper_bound=366)",
      {"day_of_year": data_for_normal['day_of_year'],},
      return_type='dataframe'
  )

  # Predict fitted values
  model_norm = sm.OLS(data_for_normal[metric], basis_normal).fit()
  predictions = round(model_norm.predict(basis_normal),2)
  return {"basis": basis_normal, "model": model_norm, "predictions": predictions}

high_temp_model = create_model("max_temp", 13)
low_temp_model = create_model("min_temp", 6)
avg_temp_model = create_model("avg_temp", 6)
# data_for_normal.drop(['initial_prediction'], inplace=True, axis=1)
data_for_normal['normal_high'] = high_temp_model['predictions']
data_for_normal['normal_low'] = low_temp_model['predictions']
data_for_normal['normal_temp'] = avg_temp_model['predictions']

# What is avg hottest / coldest day of year?

(
  data_for_normal[data_for_normal['year']==2024]
  [['day_of_year','date_formatted','normal_high']]
  .sort_values(by='normal_high', ascending=False)
)
(
  data_for_normal[data_for_normal['year']==2024]
  [['day_of_year','date_formatted','normal_low']]
  .sort_values(by='normal_low', ascending=True)
)

# Jan 4th is the coldest day of the year
# July 20th is hottest day of the year

# Sanity Check work
(
  data_for_normal
  .groupby('date_formatted')
  .agg({'max_temp': 'mean'})
  .reset_index()
  .sort_values(by='max_temp', ascending=False)
  .head(10)
)

# check that endpoints are different
data_for_normal[data_for_normal['year']==2024]


# 10/90 Bands
# Calculate from data, then interpolate with spline

def compute_bands(metric, p_bands=[0.1, 0.9], spline_df=6, plot=True):
  # Need to specify p in lambda due to late binding
  band_aggs = {
    f"{metric}_p{str(p*100).replace('.', '_')}": lambda row, p=p: row.quantile(p)
    for p in p_bands
  }
  
  band_names = band_aggs.keys()
  percentiles = (
      data_for_normal
      .groupby('day_of_year')[metric]
      .agg(**band_aggs)
      .reset_index()
  )

  # Create basis for bands
  basis_bands = dmatrix(
      f"1 + cc(day_of_year, df={spline_df}, constraints='center', upper_bound=366)",
      {"day_of_year": percentiles['day_of_year'],},
      return_type='dataframe'
  )

  models = {p: sm.OLS(percentiles[p], basis_bands).fit() for p in band_names}
  for p in band_names:
    percentiles[p+'_fit'] = round(models[p].predict(basis_bands),2)


  # Fit a spline to this
  if plot:
    colors = ['blue', 'red', 'green', 'orange']
    for i, p in enumerate(band_names):
      col = colors[i % 4]
      sns.scatterplot(percentiles, x='day_of_year', y=p, label=str(p.split(f"{metric}_p")[1]).replace('_', '.') + " percentile", color=col, alpha=0.4)
      sns.lineplot(percentiles, x='day_of_year', y=p+"_fit", color='dark'+col, linewidth=2)
    plt.show()

  return percentiles

high_quantiles = compute_bands('max_temp', [0.1, 0.9], spline_df=6)
low_quantiles = compute_bands('min_temp', [0.1, 0.9])

# Put everything together
combined = (
  data_for_normal
  .merge(high_quantiles, on='day_of_year', how='left')
  .merge(low_quantiles, on='day_of_year', how='left')
)
one_year = combined[combined['year']==2024]

# Write out normals to CSV
(
  one_year
  .rename(columns={'max_temp_p10_0_fit': 'max_temp_p10', 'max_temp_p90_0_fit': 'max_temp_p90', 'min_temp_p10_0_fit': 'min_temp_p10', 'min_temp_p90_0_fit': 'min_temp_p90'})
  [['date_formatted', 'day_of_year', 'month', 'day_of_year', 'normal_high', 'normal_low', 'normal_temp', 'max_temp_p10', 'max_temp_p90', 'min_temp_p10', 'min_temp_p90']]
  .to_csv('weather/output_sources/temp_normals.csv', index=False)
)

# high Temperatures
plt.figure(figsize=(8, 6))
sns.scatterplot(combined, x='day_of_year', y='max_temp', legend=None, alpha=0.4, color='darkred', size=2)
sns.lineplot(one_year, x='day_of_year', y='normal_high', color='darkred', linewidth=2)
# Plot translucent band for 10th-90th percentile fit
plt.fill_between(
    one_year['day_of_year'],
    one_year['max_temp_p10_0_fit'],
    one_year['max_temp_p90_0_fit'],
    color='red',
    alpha=0.35,
    label='10th-90th Percentile Band'
)
plt.title('Average High Temperatures')
plt.show()

# Low Temperatures
plt.figure(figsize=(8, 6))
sns.scatterplot(combined, x='day_of_year', y='min_temp', legend=None, alpha=0.4, color='darkblue', size=2)
sns.lineplot(one_year, x='day_of_year', y='normal_low', color='blue', linewidth=2)
# Plot translucent band for 10th-90th percentile fit
plt.fill_between(
    one_year['day_of_year'],
    one_year['min_temp_p10_0_fit'],
    one_year['min_temp_p90_0_fit'],
    color='royalblue',
    alpha=0.35,
    label='10th-90th Percentile Band'
)
plt.title('Average Low Temperatures')
plt.show()

# Combined
plt.figure(figsize=(10, 7))
sns.scatterplot(combined, x='day_of_year', y='max_temp', legend=None, alpha=0.25, color='firebrick', size=1)
sns.scatterplot(combined, x='day_of_year', y='min_temp', legend=None, alpha=0.25, color='darkblue', size=1)
sns.lineplot(one_year, x='day_of_year', y='normal_high', color='darkred', linewidth=2)
sns.lineplot(one_year, x='day_of_year', y='normal_low', color='blue', linewidth=2)
# Plot translucent band for 10th-90th percentile fit
plt.fill_between(
    one_year['day_of_year'],
    one_year['max_temp_p10_0_fit'],
    one_year['max_temp_p90_0_fit'],
    color='tomato',
    alpha=0.35,
    label='10th-90th High Percentile Band'
)
plt.fill_between(
    one_year['day_of_year'],
    one_year['min_temp_p10_0_fit'],
    one_year['min_temp_p90_0_fit'],
    color='royalblue',
    alpha=0.35,
    label='10th-90th Low Percentile Band'
)
plt.title('Average High/Low Temperatures')
plt.show()

# Monthly "normals"
month_avgs = (
  one_year
  .groupby('month')
  .agg({'normal_high': lambda x: round(mean(x), 1), 'normal_low': lambda x: round(mean(x), 1)})
  .reset_index()
)

month_records = (
  combined
  .groupby('month')
  .agg({'max_temp':'max', 'min_temp': 'min'})
  .reset_index()
)

(
  month_avgs
  .merge(month_records, on='month', how='left')
)

# When is the average last frost?

temperatures['frost'] = temperatures.apply(
    lambda row: row['day_of_year'] if row['min_temp'] < 32.5 else pd.NaT,
    axis=1
)

frost_dates = (
  temperatures[temperatures['day_of_year'] < 200]
  .groupby('year')
  .agg({"frost": "max"})
  .reset_index()
)

frost_dates['date'] = frost_dates['frost'].apply(dayofyear_to_month_day)
frost_dates

sns.histplot(frost_dates['frost'].dropna(), bins=15, color='skyblue', edgecolor='black', kde=True, stat='density')
plt.title('Avg First Frost')
plt.show()

# Quantiles
frost_dates['frost'].quantile([0, 0.1, 0.5, 0.75, 0.9, 1]).apply(dayofyear_to_month_day)

# Earliest Frost
frost_dates_early = (
  temperatures[temperatures['day_of_year'] > 200]
  .groupby('year')
  .agg({"frost": "min"})
  .reset_index()
  .dropna()
)
frost_dates_early['date'] = frost_dates_early['frost'].apply(dayofyear_to_month_day)
frost_dates_early

sns.histplot(frost_dates_early['frost'].dropna(), bins=15, color='orange', edgecolor='black', kde=True, stat='density')
plt.title('Avg Last Frost')
plt.show()

# Quantiles
frost_dates_early['frost'].quantile([0, 0.1, 0.5, 0.75, 0.9, 1]).apply(dayofyear_to_month_day)

max_date = temperatures['date'].max()
if max_date != max_date + pd.offsets.MonthEnd(0):
  most_recent_month = temperatures['date'].max().replace(day=1) - pd.Timedelta(days=1)
else:
  most_recent_month = max_date


# Heatmap of actual temp relative to avg
normals = data_for_normal[data_for_normal['year']==2024][['day_of_year', 'normal_high', 'normal_low', 'normal_temp']]
normals = (
  temperatures
  .merge(normals, on='day_of_year', how='left')
).copy()
normals["departure"] = normals['avg_temp'] - normals['normal_temp']
normals["departure_high"] = normals['max_temp'] - normals['normal_high']
normals["departure_low"] = normals['min_temp'] - normals['normal_low']

heatmap = normals[normals['year'].between(2021, 2025)].copy()
heatmap = heatmap.set_index('date')



# Daily Mean Temp
calplot.calplot(
    heatmap['departure'],
    cmap='coolwarm',
    colorbar=True,
    figsize=(15, 12),
    suptitle='Temperature Departure from Normal',
    vmin=-22,
    vmax=22,
)
plt.show()

# Todo: See this on a monthly level
monthly_map = (
  normals[temperatures['date'] <= most_recent_month]
  .groupby(['year', 'month'])
  .agg({'departure': "mean"})
  .reset_index()
).copy()

monthly_map_year = (
  monthly_map
  .groupby(['year'])
  .agg(
    departure=('departure', 'mean'),
    count=('departure', 'count'),
  )
  .reset_index()
)

monthly_map_year = monthly_map_year[monthly_map_year['count']==12]
monthly_map_year = monthly_map_year.drop(['count'], axis=1)
monthly_map_year['month'] = 'Year'
monthly_map = pd.concat([monthly_map, monthly_map_year])
monthly_map['month'] = pd.Categorical(monthly_map['month'], categories=month_order+["Year"], ordered=True)


monthly_map['rank'] = (
    monthly_map.groupby('month')['departure']
    .rank(method='min', ascending=True)
)

monthly_pivot_value = monthly_map.pivot(index='month', columns='year', values='departure')
monthly_pivot_rank = monthly_map.pivot(index='month', columns='year', values='rank')

custom_colors = [
  '#3b4cc0', '#528ecb', '#7fb8da', '#b5d5e6', 
  '#e0e0e0',"#e0e0e0", '#e0e0e0','#e0e0e0',
  '#f6bfa6', '#ea7b60', '#c63c36', '#962d20',
]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)

plt.figure(figsize=(12, 6))
sns.heatmap(
    monthly_pivot_value,
    cmap=custom_cmap,
    # annot=True,
    fmt=".1f",
    linewidths=0.05,
    cbar_kws={'label': 'Departure (°F)'}
)
plt.title('Monthly Temperature Departure from 1995-2024 Normal')
plt.xlabel('Year')
plt.ylabel('Month')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))
sns.heatmap(
    monthly_pivot_rank,
    cmap=custom_cmap,
    vmax=max(monthly_map['rank']),
    vmin=1,
    center=(max(monthly_map['rank'])+1)/2,
    annot=True,
    fmt=".0f",
    linewidths=0.005,
    cbar_kws={'label': 'Rank'},
    annot_kws={"size": 8}
)
plt.title('Monthly Departure Rank from 1995-2024 Normal')
plt.xlabel('Year')
plt.ylabel('Month')
plt.tight_layout()
plt.show()

# Dashboard Ideas ----------------------
# Heatmap over last month showing departure from normal
# Show daily normal + record high/low + percentiles