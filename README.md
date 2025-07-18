# CAPSTONE 2 Project - Share price predictive analytics


Gradio app deployment using Linear Regressor model: Machine Learning model adopted with 3 visualization chart for:


The single-day price prediction as text.

A Forecast Trajectory chart showing the predicted path from the last known price to the future date.

A Historical Context chart showing the forecast appended to the stock's full price history.

A Model Feature Importance chart, which reveals the most influential factors in the model's prediction.



<img width="937" height="454" alt="image" src="https://github.com/user-attachments/assets/aeabb195-c15c-4a60-a8ca-e5e78efef710" />



<img width="914" height="378" alt="image" src="https://github.com/user-attachments/assets/01773bfb-0e3c-4e5e-be01-bcffe4b367e6" />



<img width="966" height="501" alt="image" src="https://github.com/user-attachments/assets/298aaded-4506-424b-91c0-4c9b1c1227a1" />





Linear Regressor model: Machine Learning model adopted.
Accuracy Results reported:


| Company   | Mean Absolute Error (MAE)   | Root Mean Squared Error (RMSE)   |   R-squared (R²) |
|:----------|:----------------------------|:---------------------------------|-----------------:|
| NVDA      | $2.93                       | $3.99                            |             0.98 |
| AMD       | $3.23                       | $4.50                            |             0.97 |
| Qualcomm  | $3.10                       | $4.23                            |             0.95 |
| Intel     | $0.63                       | $0.97                            |             0.99 |
| Telsa     | $8.08                       | $11.47                           |             0.98 |


====================================================================================================================

Basic Steps from Data Preparation, Data Cleaning, Conduct Exploratory Data Analysis (EDA), Machine Learning model test, validation and accuracy matrix checks
Advance Steps: accuracy matrix comparison, adopt Machine learning model, model deployment to gradio app and visualization to validate deployment test results.

====================================================================================================================

from IPython.display import HTML
HTML('<div style="font-size: 30px; font-weight: bold;">'
     'CAPSTONE 2 Project - Share price predictive analytics, Preparation data set and Data Cleaning'
     '</div>')

# CAPSTONE 2 Project - Share price predictive analytics #
# Preparation data set and Data Cleaning # 

# Dataset References:

https://www.nasdaq.com/market-activity/stocks | NVDA, AMD, INTC, Qualcomm, Telsa Share prices https://fred.stlouisfed.org/series/FEDFUNDS | Fed Interest Rates https://fred.stlouisfed.org/series/GDP | GDP https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL | CPI data

import pandas as pd

# Load and inspect each of the four data files.

# 1. CPI.csv
print("Inspecting CPI.csv")
cpi_df = pd.read_csv('CPI.csv')
print(cpi_df.info())
print(cpi_df.head())
print("-" * 50)

# 2. Federal Interest Rates.csv
print("Inspecting Federal Interest Rates.csv")
interest_rates_df = pd.read_csv('Federal Interest Rates.csv')
print(interest_rates_df.info())
print(interest_rates_df.head())
print("-" * 50)

# 3. GDP.csv
print("Inspecting GDP.csv")
gdp_df = pd.read_csv('GDP.csv')
print(gdp_df.info())
print(gdp_df.head())
print("-" * 50)

# 4. stockdata.csv
print("Inspecting stockdata.csv")
stockdata_df = pd.read_csv('stockdata.csv')
print(stockdata_df.info())
print(stockdata_df.head())
print("-" * 50)


Inspecting CPI.csv
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 65 entries, 0 to 64
Data columns (total 2 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   cpi date  65 non-null     object 
 1   CPI       65 non-null     float64
dtypes: float64(1), object(1)
memory usage: 1.1+ KB
None
   cpi date       CPI
0  1/1/2020  3.192467
1  2/1/2020  2.268274
2  3/1/2020  0.127169
3  4/1/2020 -2.158071
4  5/1/2020 -0.191344
--------------------------------------------------
Inspecting Federal Interest Rates.csv
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 61 entries, 0 to 60
Data columns (total 2 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   interest rate date  61 non-null     object 
 1   interest rates      61 non-null     float64
dtypes: float64(1), object(1)
memory usage: 1.1+ KB
None
  interest rate date  interest rates
0           6/1/2020            0.08
1           7/1/2020            0.09
2           8/1/2020            0.10
3           9/1/2020            0.09
4          10/1/2020            0.09
--------------------------------------------------
Inspecting GDP.csv
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 21 entries, 0 to 20
Data columns (total 2 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   GDP date  21 non-null     object 
 1   GDP       21 non-null     float64
dtypes: float64(1), object(1)
memory usage: 468.0+ bytes
None
    GDP date        GDP
0   1/1/2020  21727.657
1   4/1/2020  19935.444
2   7/1/2020  21684.551
3  10/1/2020  22068.767
4   1/1/2021  22656.793
--------------------------------------------------
Inspecting stockdata.csv
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6280 entries, 0 to 6279
Data columns (total 7 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   Company        6280 non-null   object
 1   Date           6280 non-null   object
 2   Close price    6280 non-null   object
 3   Volume         6280 non-null   int64 
 4   Open price     6280 non-null   object
 5   Highest price  6280 non-null   object
 6   Lowest price   6280 non-null   object
dtypes: int64(1), object(6)
memory usage: 343.6+ KB
None
    Company       Date Close price     Volume Open price Highest price  \
0       AMD  7/13/2020     $53.59    57741820    $56.68        $58.35    
1     Intel  7/13/2020     $58.58    19082940    $59.84        $60.62    
2      NVDA  7/13/2020     $10.05   457074800    $10.60        $10.79    
3  Qualcomm  7/13/2020     $91.33     7606351    $93.30        $94.13    
4     Telsa  7/13/2020     $99.80   584780108   $110.60       $119.67    

  Lowest price  
0      $53.38   
1      $58.39   
2      $10.03   
3      $91.20   
4      $98.07  


import pandas as pd

# Load the datasets
cpi_df = pd.read_csv('CPI.csv')
interest_rates_df = pd.read_csv('Federal Interest Rates.csv')
gdp_df = pd.read_csv('GDP.csv')
stockdata_df = pd.read_csv('stockdata.csv')

# --- Data Cleaning and Preparation ---

# 1. Clean CPI data
cpi_df['Date'] = pd.to_datetime(cpi_df['cpi date'])
cpi_df = cpi_df.drop('cpi date', axis=1)
cpi_df = cpi_df.set_index('Date')

# 2. Clean Interest Rates data
interest_rates_df['Date'] = pd.to_datetime(interest_rates_df['interest rate date'])
interest_rates_df = interest_rates_df.drop('interest rate date', axis=1)
interest_rates_df = interest_rates_df.rename(columns={'interest rates': 'Interest_Rate'})
interest_rates_df = interest_rates_df.set_index('Date')


# 3. Clean GDP data
gdp_df['Date'] = pd.to_datetime(gdp_df['GDP date'])
gdp_df = gdp_df.drop('GDP date', axis=1)
gdp_df = gdp_df.set_index('Date')


# 4. Clean Stock data
stockdata_df['Date'] = pd.to_datetime(stockdata_df['Date'])
for col in ['Close price', 'Open price', 'Highest price', 'Lowest price']:
    stockdata_df[col] = stockdata_df[col].replace({'\$': ''}, regex=True).astype(float)
stockdata_df = stockdata_df.rename(columns={
    'Close price': 'Close_Price',
    'Open price': 'Open_Price',
    'Highest price': 'Highest_Price',
    'Lowest price': 'Lowest_Price'
})
stockdata_df = stockdata_df.set_index(['Date', 'Company'])


# --- Merge DataFrames ---

# Merge stock data with economic indicators
# Start by merging all economic dataframes.
merged_df = cpi_df.join(interest_rates_df, how='outer')
merged_df = merged_df.join(gdp_df, how='outer')


# Forward fill the economic data to fill missing values.
# This is appropriate for this kind of time-series data where the values are reported periodically
# and can be assumed to be constant between reporting dates.
merged_df = merged_df.ffill()

# Now, join with stock data.
# We need to reset index for merging and then set it back.
stockdata_df = stockdata_df.reset_index()
merged_df = merged_df.reset_index()

# Perform the merge on the 'Date' column
final_df = pd.merge(stockdata_df, merged_df, on='Date', how='left')

# Set the multi-index back
final_df = final_df.set_index(['Date', 'Company'])

# Display the cleaned and merged data
print("Cleaned and Merged Data:")
print(final_df.info())
print(final_df.head())
final_df.to_csv("cleaned_merged_data.csv")


Cleaned and Merged Data:
<class 'pandas.core.frame.DataFrame'>
MultiIndex: 6280 entries, (Timestamp('2020-07-13 00:00:00'), 'AMD') to (Timestamp('2025-07-11 00:00:00'), 'Telsa')
Data columns (total 8 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Close_Price    6280 non-null   float64
 1   Volume         6280 non-null   int64  
 2   Open_Price     6280 non-null   float64
 3   Highest_Price  6280 non-null   float64
 4   Lowest_Price   6280 non-null   float64
 5   CPI            195 non-null    float64
 6   Interest_Rate  195 non-null    float64
 7   GDP            195 non-null    float64
dtypes: float64(7), int64(1)
memory usage: 453.3+ KB
None
                     Close_Price     Volume  Open_Price  Highest_Price  \
Date       Company                                                       
2020-07-13 AMD             53.59   57741820       56.68          58.35   
           Intel           58.58   19082940       59.84          60.62   
           NVDA            10.05  457074800       10.60          10.79   
           Qualcomm        91.33    7606351       93.30          94.13   
           Telsa           99.80  584780108      110.60         119.67   

                     Lowest_Price  CPI  Interest_Rate  GDP  
Date       Company                                          
2020-07-13 AMD              53.38  NaN            NaN  NaN  
           Intel            58.39  NaN            NaN  NaN  
           NVDA             10.03  NaN            NaN  NaN  
           Qualcomm         91.20  NaN            NaN  NaN  
           Telsa            98.07  NaN            NaN  NaN  

pip install pandas matplotlib seaborn

Requirement already satisfied: pandas in c:\users\user\anaconda3\lib\site-packages (2.2.2)Note: you may need to restart the kernel to use updated packages.

Requirement already satisfied: matplotlib in c:\users\user\anaconda3\lib\site-packages (3.9.2)
Requirement already satisfied: seaborn in c:\users\user\anaconda3\lib\site-packages (0.13.2)
Requirement already satisfied: numpy>=1.26.0 in c:\users\user\anaconda3\lib\site-packages (from pandas) (1.26.4)
Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\user\appdata\roaming\python\python312\site-packages (from pandas) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in c:\users\user\anaconda3\lib\site-packages (from pandas) (2024.1)
Requirement already satisfied: tzdata>=2022.7 in c:\users\user\anaconda3\lib\site-packages (from pandas) (2023.3)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib) (1.2.0)
Requirement already satisfied: cycler>=0.10 in c:\users\user\anaconda3\lib\site-packages (from matplotlib) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\user\anaconda3\lib\site-packages (from matplotlib) (4.51.0)
Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib) (1.4.4)
Requirement already satisfied: packaging>=20.0 in c:\users\user\appdata\roaming\python\python312\site-packages (from matplotlib) (24.2)
Requirement already satisfied: pillow>=8 in c:\users\user\anaconda3\lib\site-packages (from matplotlib) (10.4.0)
Requirement already satisfied: pyparsing>=2.3.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib) (3.1.2)
Requirement already satisfied: six>=1.5 in c:\users\user\appdata\roaming\python\python312\site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned and merged data
# Make sure 'cleaned_merged_data.csv' is in the same directory as your notebook
try:
    final_df = pd.read_csv('cleaned_merged_data.csv', index_col=['Date', 'Company'], parse_dates=['Date'])
except FileNotFoundError:
    print("Error: 'cleaned_merged_data.csv' not found.")
    print("Please make sure the data file is in the same directory as your notebook.")
    # As a fallback, here is the code to generate the cleaned data again from the original files.
    # You would need CPI.csv, Federal Interest Rates.csv, GDP.csv, and stockdata.csv

    # Load the datasets
    cpi_df = pd.read_csv('CPI.csv')
    interest_rates_df = pd.read_csv('Federal Interest Rates.csv')
    gdp_df = pd.read_csv('GDP.csv')
    stockdata_df = pd.read_csv('stockdata.csv')

    # --- Data Cleaning and Preparation ---
    cpi_df['Date'] = pd.to_datetime(cpi_df['cpi date'])
    cpi_df = cpi_df.drop('cpi date', axis=1)
    cpi_df = cpi_df.set_index('Date')

    interest_rates_df['Date'] = pd.to_datetime(interest_rates_df['interest rate date'])
    interest_rates_df = interest_rates_df.drop('interest rate date', axis=1)
    interest_rates_df = interest_rates_df.rename(columns={'interest rates': 'Interest_Rate'})
    interest_rates_df = interest_rates_df.set_index('Date')

    gdp_df['Date'] = pd.to_datetime(gdp_df['GDP date'])
    gdp_df = gdp_df.drop('GDP date', axis=1)
    gdp_df = gdp_df.set_index('Date')

    stockdata_df['Date'] = pd.to_datetime(stockdata_df['Date'])
    for col in ['Close price', 'Open price', 'Highest price', 'Lowest price']:
        stockdata_df[col] = stockdata_df[col].replace({'\$': ''}, regex=True).astype(float)
    stockdata_df = stockdata_df.rename(columns={
        'Close price': 'Close_Price', 'Open price': 'Open_Price',
        'Highest price': 'Highest_Price', 'Lowest price': 'Lowest_Price'
    })
    stockdata_df = stockdata_df.set_index(['Date', 'Company'])

    # --- Merge DataFrames ---
    merged_df = cpi_df.join(interest_rates_df, how='outer').join(gdp_df, how='outer')
    merged_df = merged_df.ffill()

    stockdata_df = stockdata_df.reset_index()
    merged_df = merged_df.reset_index()
    final_df = pd.merge(stockdata_df, merged_df, on='Date', how='left')
    final_df = final_df.set_index(['Date', 'Company'])
    final_df.to_csv("cleaned_merged_data.csv")


# --- Handle Missing Values ---
final_df['CPI'] = final_df['CPI'].ffill().bfill()
final_df['Interest_Rate'] = final_df['Interest_Rate'].ffill().bfill()
final_df['GDP'] = final_df['GDP'].ffill().bfill()


# --- Correlation Heatmaps ---
print("Generating Correlation Heatmaps...")

companies = final_df.index.get_level_values('Company').unique()

fig, axes = plt.subplots(nrows=len(companies), ncols=1, figsize=(10, 8 * len(companies)))
fig.suptitle('Correlation Heatmaps of Stock Data and Economic Indicators for Tech Companies', fontsize=16)

for i, company in enumerate(companies):
    company_df = final_df.loc[final_df.index.get_level_values('Company') == company]
    correlation_matrix = company_df.corr()
    ax = axes[i]
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title(f'Correlation Matrix for {company}')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()


# --- Time Series Visualization ---
print("\nGenerating Time Series Plots...")

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 20), sharex=True)
fig.suptitle('Time Series Analysis', fontsize=16)

# Plot 1: Closing Prices
for company in companies:
    close_prices = final_df.loc[final_df.index.get_level_values('Company') == company, 'Close_Price']
    close_prices = close_prices.reset_index(level='Company', drop=True)
    axes[0].plot(close_prices.index, close_prices, label=company)
axes[0].set_title('Stock Closing Prices Over Time')
axes[0].set_ylabel('Close Price (USD)')
axes[0].legend()
axes[0].grid(True)

# Plot 2: CPI
cpi_series = final_df['CPI'].reset_index().drop_duplicates(subset='Date').set_index('Date')['CPI']
axes[1].plot(cpi_series.index, cpi_series, label='CPI', color='orange')
axes[1].set_title('CPI Over Time')
axes[1].set_ylabel('CPI')
axes[1].legend()
axes[1].grid(True)

# Plot 3: Interest Rate
interest_rate_series = final_df['Interest_Rate'].reset_index().drop_duplicates(subset='Date').set_index('Date')['Interest_Rate']
axes[2].plot(interest_rate_series.index, interest_rate_series, label='Interest Rate', color='green')
axes[2].set_title('Federal Interest Rate Over Time')
axes[2].set_ylabel('Interest Rate (%)')
axes[2].legend()
axes[2].grid(True)

# Plot 4: GDP
gdp_series = final_df['GDP'].reset_index().drop_duplicates(subset='Date').set_index('Date')['GDP']
axes[3].plot(gdp_series.index, gdp_series, label='GDP', color='red')
axes[3].set_title('GDP Over Time')
axes[3].set_xlabel('Date')
axes[3].set_ylabel('GDP (in Billions USD)')
axes[3].legend()
axes[3].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()



<img width="730" height="650" alt="image" src="https://github.com/user-attachments/assets/3ae4ebd9-e895-4e35-bba6-2a7f839d0dec" />


<img width="743" height="553" alt="image" src="https://github.com/user-attachments/assets/4e68c0f2-0b05-4663-8aa2-94ec9ee2acd2" />


<img width="703" height="553" alt="image" src="https://github.com/user-attachments/assets/b02bcd15-6f06-47ad-9018-28777cbebf6e" />


<img width="732" height="551" alt="image" src="https://github.com/user-attachments/assets/493e6266-7161-4de1-b9d9-9f7ffd238db8" />


<img width="718" height="563" alt="image" src="https://github.com/user-attachments/assets/062706e2-46c4-46dd-91bd-5cdbdf50ae0c" />


<img width="816" height="543" alt="image" src="https://github.com/user-attachments/assets/f9e9a16b-5e8d-4f13-8fc7-21362197e006" />


<img width="832" height="521" alt="image" src="https://github.com/user-attachments/assets/23eaabb7-4f6b-4251-9a35-1766b04cb0da" />


from IPython.display import HTML
HTML('<div style="font-size: 30px; font-weight: bold;">'
     'Machine Learning Models'
     '</div>')

=============================================================================================================

# MACHINE LEARNING Models

# '1. Linear Regressor model, 2 LSTM (Long Short-Term Memory), 3 Random Forest Regressor, 4 Gradient Boosting (LightGBM) model, 5 Hybrid Deep Learning + Tree Models (XGBoost) model' '
')
MACHINE LEARNING MODELS
1 Linear Regressor model
2 LSTM (Long Short-Term Memory) model
3 Random Forest Regressor model
4 Gradient Boosting (LightGBM) model
5 Hybrid Deep Learning + Tree Models (XGBoost) model


=============================================================================================================



           






























