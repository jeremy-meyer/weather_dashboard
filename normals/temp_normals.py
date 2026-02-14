import pandas as pd
from numpy import mean
from matplotlib import pyplot as plt
from datetime import date
import seaborn as sns
import calplot
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
from patsy import dmatrix
from utils.shared_functions import *


# Subset relevant data
full_data = pd.read_csv('raw_sources/'+'daily_weather.csv', index_col=False, parse_dates=['date'])
temperatures = full_data[['date', 'avg_temp', 'min_temp', 'max_temp']].copy()
temperatures['year'] = temperatures['date'].dt.year
temperatures['month'] = pd.Categorical(temperatures['date'].dt.strftime('%b'), categories=month_order, ordered=True)
temperatures['day_of_year'] = temperatures['date'].apply(gen_DoY_index)

# Calculate 30 year normal based on last complete decade
current_year = temperatures['year'].max()
N_years = 30
normal_start_year, normal_end_year = calc_normal_years(current_year, N_years)
data_for_normal = temperatures.loc[temperatures['year'].between(normal_start_year, normal_end_year)].copy()
data_for_normal['date_formatted'] = data_for_normal['date'].dt.strftime('%b-%d')
print(f"Calculating normals from {normal_start_year} to {normal_end_year}")


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

cv_metric = 'avg_temp'
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
  print(f"Finished {df}")
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
    min_temp=('min_temp', 'mean'),
    avg_temp=('avg_temp', 'mean'),
  )
  .reset_index()
)

knots_to_display = [5,9]
for k in knots_to_display:
  basis = dmatrix(
    f"1 + cc(day_of_year, df={k}, constraints='center', upper_bound=366)",
    {"day_of_year": alt_model_input['day_of_year'],},
    return_type='dataframe'
  )
  model = sm.OLS(alt_model_input[cv_metric], basis).fit()
  alt_model_input[f'prediction_{k}'] = round(model.predict(basis),2)


alt_model_input.sort_values('day_of_year', ascending=False)
colors = ['darkred', 'orange', 'gold', 'darkyellow', 'green']
sns.scatterplot(alt_model_input, x='day_of_year', y=cv_metric, legend=None, alpha=0.5, color='red', size=3)
for i, k in enumerate(knots_to_display):
  sns.lineplot(alt_model_input, x='day_of_year', y=f'prediction_{k}', color=colors[i], linewidth=2, label=f'df={k}')
plt.title(f'Average {cv_metric}')
plt.show()

# Results:
# DF=7 for max_temp (less noisy)
# DF=6 for min_temp
# DF=5 for avg_temp

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

high_temp_model = create_model("max_temp", 7)
low_temp_model = create_model("min_temp", 6)
avg_temp_model = create_model("avg_temp", 5)
# data_for_normal.drop(['initial_prediction'], inplace=True, axis=1)
data_for_normal['normal_high'] = high_temp_model['predictions']
data_for_normal['normal_low'] = low_temp_model['predictions']
data_for_normal['normal_temp'] = avg_temp_model['predictions']

# What is avg hottest / coldest day of year on avg?

(
  data_for_normal[data_for_normal['year']==normal_end_year]
  [['day_of_year','date_formatted','normal_high']]
  .sort_values(by='normal_high', ascending=False)
)
(
  data_for_normal[data_for_normal['year']==normal_end_year]
  [['day_of_year','date_formatted','normal_low']]
  .sort_values(by='normal_low', ascending=True)
)

# Jan 4th is the coldest day of the year
# July 26th is hottest day of the year

# Sanity Check work
(
  data_for_normal
  .groupby('date_formatted')
  .agg({'max_temp': 'mean'})
  .reset_index()
  .sort_values(by='max_temp', ascending=False)
  .head(10)
)


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
one_year = combined[combined['year']==normal_end_year - (normal_end_year % 4)] # make sure to get a leap year

# Write out normals to CSV
(
  one_year
  .rename(columns={'max_temp_p10_0_fit': 'max_temp_p10', 'max_temp_p90_0_fit': 'max_temp_p90', 'min_temp_p10_0_fit': 'min_temp_p10', 'min_temp_p90_0_fit': 'min_temp_p90'})
  [['date_formatted', 'day_of_year', 'month', 'day_of_year', 'normal_high', 'normal_low', 'normal_temp', 'max_temp_p10', 'max_temp_p90', 'min_temp_p10', 'min_temp_p90']]
  .to_csv('output_sources/temp_daily_normals.csv', index=False)
)