
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% data

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
covid_df = pd.read_csv(url, index_col=0)

# columns that are NOT dates
meta_cols = [
    'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
    'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population'
]

# all date columns (wide format)
date_cols = [c for c in covid_df.columns if c not in meta_cols]
dates = pd.to_datetime(date_cols)

#%% Instructions
'''
Overall instructions:
As described in the homework description, each graphic you make must:
   1. Have a thoughtful title
   2. Have clearly labelled axes 
   3. Be legible
   4. Not be a pie chart
I should be able to run your .py file and recreate the graphics without error.
As per usual, any helper variables or columns you create should be thoughtfully
named.
'''


#%% viz 1

# Filter to Utah counties
utah_df = covid_df[covid_df['Province_State'] == 'Utah'].copy()

# Choose a county to highlight: Salt Lake County
highlight_county = 'Salt Lake'
highlight_row = utah_df[utah_df['Admin2'] == highlight_county].iloc[0]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot all Utah counties in light gray
for _, row in utah_df.iterrows():
    ax.plot(dates, row[date_cols].values,
            color='lightgrey', linewidth=1, alpha=0.7)

# Plot highlight county in contrasting color
ax.plot(dates, highlight_row[date_cols].values,
        color='tab:red', linewidth=2.5, label=f'{highlight_county} County')

# Format axes and title
ax.set_title('COVID-19 Cumulative Confirmed Cases Over Time in Utah Counties')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative confirmed cases')

# Date formatting
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()

ax.legend()
plt.tight_layout()
plt.show()


#%% viz 2

# Filter to Utah and Florida
fl_df = covid_df[covid_df['Province_State'] == 'Florida'].copy()

# Use the latest date column as "to date" total
latest_date_col = date_cols[-1]

# Find county with most cases in each state
utah_top_idx = utah_df[latest_date_col].idxmax()
fl_top_idx = fl_df[latest_date_col].idxmax()

utah_top = utah_df.loc[utah_top_idx]
fl_top = fl_df.loc[fl_top_idx]

utah_name = f"{utah_top['Admin2']} County, Utah"
fl_name = f"{fl_top['Admin2']} County, Florida"

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(dates, utah_top[date_cols].values,
        label=utah_name, linewidth=2, linestyle='-')
ax.plot(dates, fl_top[date_cols].values,
        label=fl_name, linewidth=2, linestyle='--')

ax.set_title('Contrast of COVID-19 Cumulative Cases:\nTop Utah vs Top Florida County')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative confirmed cases')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()

ax.legend()
plt.tight_layout()
plt.show()


#%% viz 3 

# Reuse the top Utah county (utah_top) for this example
county_series = utah_top[date_cols].astype(int)
running_total = county_series
daily_new = county_series.diff().fillna(0)

fig, ax1 = plt.subplots(figsize=(10, 6))

color_total = 'tab:blue'
color_daily = 'tab:orange'

# Left y-axis: cumulative cases
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative confirmed cases', color=color_total)
ax1.plot(dates, running_total.values,
         color=color_total, linewidth=2, label='Cumulative cases')
ax1.tick_params(axis='y', labelcolor=color_total)

# Right y-axis: daily new cases
ax2 = ax1.twinx()
ax2.set_ylabel('Daily new confirmed cases', color=color_daily)
ax2.plot(dates, daily_new.values,
         color=color_daily, linewidth=1.5, linestyle='--', label='Daily new cases')
ax2.tick_params(axis='y', labelcolor=color_daily)

# Title and x-axis formatting
county_label = f"{utah_top['Admin2']} County, Utah"
plt.title(f'Cumulative vs Daily New COVID-19 Cases\n{county_label}')

ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

fig.tight_layout()
plt.show()
 

#%% viz 4
"""
Stacked bar chart: Utah's county contributions to total COVID-19 cases.
Styled to look like the penguin example (clean bar, no grid, plain numbers).
"""

from matplotlib.ticker import FuncFormatter

state_for_stack = "Utah"
top_n = 3  # show 3 biggest counties + Other counties

# Filter to chosen state
state_df = covid_df[covid_df["Province_State"] == state_for_stack].copy()

# Total cases per county on latest date
county_totals = (
    state_df.groupby("Admin2")[latest_date_col]
            .sum()
            .sort_values(ascending=False)
)

# Drop obvious non-county labels if present
drop_labels = ["Unassigned", "Out of UT", "Out of State"]
county_totals = county_totals[~county_totals.index.isin(drop_labels)]

top_counties = county_totals.head(top_n)
other_total = county_totals.iloc[top_n:].sum()

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 4))

x_pos = [0]       # single bar position
bar_width = 0.6
bottom = 0

# Stack top counties
for county, cases in top_counties.items():
    ax.bar(x_pos, cases, width=bar_width, bottom=bottom, label=county)
    bottom += cases

# Stack "Other counties"
if other_total > 0:
    ax.bar(x_pos, other_total, width=bar_width, bottom=bottom, label="Other counties")

# X-axis: one tick, labeled with the state
ax.set_xticks(x_pos)
ax.set_xticklabels([state_for_stack])

# Y-axis formatting: plain numbers with commas, no scientific notation
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

# Labels & title 
ax.set_ylabel("Number of confirmed cases")
ax.set_title("County Contributions to Total COVID-19 Cases in Utah")

# Simple legend in upper right, like penguin chart
ax.legend(title="Counties", loc="upper right")

# No grid lines 
ax.grid(False)

plt.tight_layout()
plt.show()
