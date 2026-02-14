import pandas as pd
import numpy as np

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Generates day of year index, with options for calendar year, water year, or snow season. 
# Accounts for leap years by assigning Feb 29 a value of 59.5 and shifting subsequent days back by 1.
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

# Converts day of year index back to month and day for a non-leap year. Inverse of above
def dayofyear_to_month_day(doy):
  dt = pd.Timestamp(f"2025-01-01") + pd.Timedelta(days=doy-1)
  if doy==59.5:
    return "Feb 29"
  return dt.strftime('%b %d')

# Adds/subtracts years from a snow season string in format "YYYY-YYYY"
def offset_season(s, offset):
  return str(int(s.split('-')[0]) + offset) + '-' + str(int(s.split('-')[1]) + offset)

# Calculates start and end years for normals based on the last complete decade or the most recent year. Defaults to 30 year normals.
def calc_normal_years(current_year, N_years=30, last_complete_decade=True):
  if last_complete_decade:
    end_year = int(np.floor(current_year / 10) * 10)
  else:
    end_year = current_year-1
  start_year = end_year - N_years + 1
  return (start_year, end_year)