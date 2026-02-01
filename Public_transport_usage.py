'''
Public Transport Usage Analysis
Student ID: 24127116
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scrfft import scrfft

# Read the 2019 and 2022 Dataset Using PANDAS
data_2019 = pd.read_csv('2019data1.csv')
data_2022 = pd.read_csv('2022data1.csv')

# Convert the date columns in the dataset to datetime
data_2019['Date'] = pd.to_datetime(data_2019['Date'])
data_2022['Date and time'] = pd.to_datetime(data_2022['Date and time'])




# Solution [FIGURE 1]: Daily Passenger Numbers with Fourier Smoothing

# 2019 Daily Total
data_2019['Total_passengers'] = (data_2019['Bus pax number peak'] + 
                                  data_2019['Bus pax number offpeak'] + 
                                  data_2019['Metro pax number peak'] + 
                                  data_2019['Metro pax number offpeak'])

daily_2019 = data_2019.groupby('Date')['Total_passengers'].sum().reset_index()
daily_2019['Day_number'] = (daily_2019['Date'] - daily_2019['Date'].min()).dt.days

# 2022 Daily Total
daily_2022_counts = data_2022.groupby(data_2022['Date and time'].dt.date).size().reset_index()
daily_2022_counts.columns = ['Date', 'Count']
daily_2022_counts['Date'] = pd.to_datetime(daily_2022_counts['Date'])

# Scale factor: 2022 Total passengers / total sample size
total_2022_passengers = 97440752 + 119844941  # Bus + Metro
total_2022_sample = len(data_2022)
scale_factor = total_2022_passengers / total_2022_sample

daily_2022_counts['Total_passengers'] = daily_2022_counts['Count'] * scale_factor
daily_2022_counts['Day_number'] = (daily_2022_counts['Date'] - daily_2022_counts['Date'].min()).dt.days

# Fourier Smoothing function
def fourier_smooth_scrfft(x, y, n_terms=8):
    # Compute FFT-based Fourier coefficients
    f, a, b = scrfft(x, y)

    # Reconstruct the smoothed function
    period = x.max() - x.min()
    y_smooth = np.zeros_like(x, dtype=float)

    # Add DC component (a0)
    y_smooth += a[0]

    # Add n Fourier terms
    for n in range(1, n_terms + 1):
        y_smooth += (
            a[n] * np.cos(2 * np.pi * n * x / period) +
            b[n] * np.sin(2 * np.pi * n * x / period)
        )

    return y_smooth


# ---- Apply to the data ----

y_smooth_2019 = fourier_smooth_scrfft(
    daily_2019["Day_number"].values,
    daily_2019["Total_passengers"].values,
    n_terms=8
)

y_smooth_2022 = fourier_smooth_scrfft(
    daily_2022_counts["Day_number"].values,
    daily_2022_counts["Total_passengers"].values,
    n_terms=8
)

#Figure 1
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.scatter(daily_2019['Day_number'], daily_2019['Total_passengers'], 
           alpha=0.3, s=10, label='2019 Daily Data', color='blue')
ax1.scatter(daily_2022_counts['Day_number'], daily_2022_counts['Total_passengers'], 
           alpha=0.3, s=10, label='2022 Daily Data', color='red')
ax1.plot(daily_2019['Day_number'], y_smooth_2019, 
        linewidth=2, label='2019 Fourier Smoothed', color='darkblue')
ax1.plot(daily_2022_counts['Day_number'], y_smooth_2022, 
        linewidth=2, label='2022 Fourier Smoothed', color='darkred')

ax1.ticklabel_format(axis='y', style='plain')
ax1.set_xlabel('Day of the Year', fontsize=12)
ax1.set_ylabel('Number of Passengers', fontsize=12)
ax1.set_title('2019 and 2022 Daily Public Transport Passenger Numbers (Student ID: 24127116)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1.png', dpi=300, bbox_inches='tight')
plt.show()


# Solution [FIGURE 2]: Average passengers by day of week

# 2019 - day of week analysis
data_2019['Dayofweek'] = data_2019['Date'].dt.dayofweek
weekly_2019 = data_2019.groupby('Dayofweek')['Total_passengers'].mean()

# 2022 - day of week analysis
data_2022['Dayofweek'] = pd.to_datetime(data_2022['Date and time']).dt.dayofweek
daily_counts_2022 = data_2022.groupby(data_2022['Date and time'].dt.date).agg({
    'Date and time': 'first'
}).reset_index(drop=True)
daily_counts_2022['Date'] = pd.to_datetime(data_2022['Date and time'].dt.date.unique())
daily_counts_2022['Count'] = data_2022.groupby(data_2022['Date and time'].dt.date).size().values
daily_counts_2022['Dayofweek'] = daily_counts_2022['Date'].dt.dayofweek
daily_counts_2022['Total_passengers'] = daily_counts_2022['Count'] * scale_factor
weekly_2022 = daily_counts_2022.groupby('Dayofweek')['Total_passengers'].mean()

# X, Y, Z values (seasonal fractions)
data_2022['Month'] = pd.to_datetime(data_2022['Date and time']).dt.month

# Spring Months: March, April, May
# Summer Months: June, July, August 
# Autumn Months: September, October, November

spring_journeys = len(data_2022[data_2022['Month'].isin([3, 4, 5])])
summer_journeys = len(data_2022[data_2022['Month'].isin([6, 7, 8])])
autumn_journeys = len(data_2022[data_2022['Month'].isin([9, 10, 11])])
total_journeys = len(data_2022)

X = (spring_journeys / total_journeys) * 100
Y = (summer_journeys / total_journeys) * 100
Z = (autumn_journeys / total_journeys) * 100

# Figure 2
fig2, ax2 = plt.subplots(figsize=(12, 6))
days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
x_pos = np.arange(len(days))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, weekly_2019.values, width, label='2019', alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, weekly_2022.values, width, label='2022', alpha=0.8)

ax2.set_xlabel('Day of Week', fontsize=12)
ax2.set_ylabel('Average Number of Passengers', fontsize=12)
ax2.set_title(f'Average Passengers by Day of Week (Student ID: 24127116)\nX={X:.1f}%, Y={Y:.1f}%, Z={Z:.1f}%', 
             fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(days, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figure2.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSeasonal Journey Fractions (2022):")
print(f"X (Spring - Mar, Apr, May): {X:.2f}%")
print(f"Y (Summer - Jun, Jul, Aug): {Y:.2f}%")
print(f"Z (Autumn - Sep, Oct, Nov): {Z:.2f}%")


# Solution [FIGURE 3]: Metro price vs distance with linear regression

metro_data = data_2022[data_2022['Mode'] == 'Metro'].copy()

if len(metro_data) > 0:
    # Prepare data for regression
    X_metro = metro_data['Distance'].values
    y_metro = metro_data['Price'].values
    
    # Linear regression using NumPy polyfit
    # polyfit returns [slope, intercept] for degree 1
    coeffs = np.polyfit(X_metro, y_metro, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Predictions for plotting
    X_line = np.linspace(X_metro.min(), X_metro.max(), 100)
    y_line = slope * X_line + intercept
    
    # Create Figure 3
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(X_metro, y_metro, alpha=0.3, s=10, label='Metro Journeys')
    ax3.plot(X_line, y_line, color='red', linewidth=2, 
            label=f'Linear Fit: y = {slope:.4f}x + {intercept:.4f}')
    
    ax3.set_xlabel('Trip Length (km)', fontsize=12)
    ax3.set_ylabel('Price (Euros)', fontsize=12)
    ax3.set_title('Metro Journey Price vs Distance (Student ID: 24127116)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure3.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No metro data available for 2022")


# Solution [FIGURE 4]: Mode share comparison

# 2019 mode share - from the table
bus_2019 = data_2019['Bus pax number peak'].sum() + data_2019['Bus pax number offpeak'].sum()
metro_2019 = data_2019['Metro pax number peak'].sum() + data_2019['Metro pax number offpeak'].sum()
total_2019 = 228689197  # From the data description

bus_pct_2019 = (bus_2019 / total_2019) * 100
metro_pct_2019 = (metro_2019 / total_2019) * 100
tram_pct_2019 = 0  # No tram data in 2019

# 2022 mode share
mode_counts_2022 = data_2022['Mode'].value_counts()
bus_count_2022 = mode_counts_2022.get('Bus', 0)
metro_count_2022 = mode_counts_2022.get('Metro', 0)
tram_count_2022 = mode_counts_2022.get('Tram', 0)

total_count_2022 = len(data_2022)
bus_pct_2022 = (bus_count_2022 / total_count_2022) * 100
metro_pct_2022 = (metro_count_2022 / total_count_2022) * 100
tram_pct_2022 = (tram_count_2022 / total_count_2022) * 100

# Figure 4
fig4, ax4 = plt.subplots(figsize=(10, 6))
modes = ['Bus', 'Tram', 'Metro']
x_pos = np.arange(len(modes))
width = 0.35

pct_2019 = [bus_pct_2019, tram_pct_2019, metro_pct_2019]
pct_2022 = [bus_pct_2022, tram_pct_2022, metro_pct_2022]

bars1 = ax4.bar(x_pos - width/2, pct_2019, width, label='2019', alpha=0.8)
bars2 = ax4.bar(x_pos + width/2, pct_2022, width, label='2022', alpha=0.8)

ax4.set_xlabel('Mode of Transport', fontsize=12)
ax4.set_ylabel('Fraction of Journeys (%)', fontsize=12)
ax4.set_title('Mode Share of Public Transport Journeys (Student ID: 24127116)', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(modes)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure4.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nMode Share 2019: Bus={bus_pct_2019:.1f}%, Tram={tram_pct_2019:.1f}%, Metro={metro_pct_2019:.1f}%")
print(f"Mode Share 2022: Bus={bus_pct_2022:.1f}%, Tram={tram_pct_2022:.1f}%, Metro={metro_pct_2022:.1f}%")
