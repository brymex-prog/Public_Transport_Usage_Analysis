# Public Transport Usage Analysis
This project analyses public transport usage patterns using datasets from 2019 and 2022, with the aim of understanding seasonal trends, passenger behaviour, and pricing relationships across different modes of transport.
The analysis applies time-series techniques, regression analysis, and comparative visualisation to explore changes in transport demand over time.

## Dataset Description
Two datasets were used:

### 2019 Dataset

350 records × 9 features

Daily passenger counts and revenue for:

Bus (peak & off-peak)

Metro (peak & off-peak)

### 2022 Dataset

2,000 records × 5 features

Journey-level data including:

Date & time

Mode of transport

Distance travelled

Journey duration

Ticket prices


## Tools & Technologies

Python

Pandas, NumPy

Matplotlib

Jupyter Notebook


## Analysis Approach

Data cleaning and preparation for both datasets

Aggregation of daily passenger counts

Application of Fourier smoothing to analyse seasonal patterns

Comparison of weekday vs weekend passenger behaviour

Regression analysis of journey distance vs ticket price

Mode share analysis (Bus vs Metro)


## Key Insights

Passenger volumes in 2019 showed more stable seasonal patterns compared to 2022.

2022 data revealed lower mid-year usage with recovery toward the end of the year.

Weekday travel consistently exceeded weekend travel, though weekend usage increased in 2022.

A positive linear relationship exists between metro journey distance and ticket price.

Bus usage increased slightly between 2019 and 2022, indicating a shift in travel behaviour.

## Conclusion

This project demonstrates how time-series analysis, regression techniques, and data visualisation can be used to extract meaningful insights from transport data. The findings provide valuable context for transport planning, demand forecasting, and service optimisation.
