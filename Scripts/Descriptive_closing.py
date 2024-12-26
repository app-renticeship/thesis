#© 2024 
#Author: Muhammad Fatahillah Dante <mfdantee@gmail.com> 
#Code will be published to Author's public repository: https://github.com/app-renticeship
#Universitas Indonesia, Faculty of Economics and Business 
#Created to process data for bachelor's thesis
#
#------------------------------------------------------------------
#This section imports external libraries to be used in this project
import os                       #Basic computer capabilities to be able to locate csv files inside data folder
import yfinance as data_pull    #API call to yahoo finance for financial data
import pandas as pd             #Data manipulation external library
import numpy as np              #External mathematical operations library to process large arrays and matrices (price data)
from arch import arch_model       #GARCH model
from arch.unitroot import ADF   #Augmented Dickey-Fuller Test from external library 'arch'

#-------------------------------------------------------------------
script_directory = os.path.dirname(os.path.abspath(__file__))
indexdata_location = os.path.join(script_directory, '../Data/snp40_index_return.csv')
output_location = os.path.join(script_directory, '../Output/closing_processed_output.xlsx')
debug_location = os.path.join(script_directory, '../Output/debug_output.xlsx')
raw_data = {}                   #Dictionary to store raw price data
returns_data = {}               #Dictionary to store daily returns of price data
tickers = {                     #Yahoo Finance tickers for each commodity prices
    'oil': 'BZ=F',
    'coal': 'MTFZ24.NYM',
    'gas': 'NG=F',
}
research_period = {
    'start': '2020-01-01',      #Sample start period
    'end': '2023-12-29'         #Sample end period (discounting end of the year holiday)
}
time_interval = '1d'            #Daily time interval for price data
#-------------------------------------------------------------------
#This section reads local csv data that contains S&P Southeast Asia 40 Index returns
try:
    returns_data['index'] = pd.read_csv(indexdata_location, index_col='Date', parse_dates=True)
    returns_data['index'] = returns_data['index'].dropna()
except:
    print("[*] Missing snp40_index_return.csv file inside Data folder!")
    exit()
#------------------------------------------------------------------
#This section pulls data from yahoo finance based on the parameters set above
print("[*] Downloading Commodity Data")
raw_data['oil'] = data_pull.download(
    tickers['oil'],                
    research_period['start'],       
    research_period['end'],         
    time_interval            
    )
raw_data['coal'] = data_pull.download(
    tickers['coal'],                
    research_period['start'],       
    research_period['end'],         
    time_interval            
    )
raw_data['gas'] = data_pull.download(
    tickers['gas'],                
    research_period['start'],       
    research_period['end'],         
    time_interval           
    )

print("[*] Completed Downloading Commodity Data")
#-------------------------------------------------------------------
#This section processes raw price data adj. close, open, etc into daily logarithmic returns
#Iterates over every key (price data oil, coal, and gas) inside of raw_data dictionary
#then calculate daily logarithmic return

for price_type in raw_data:         
    raw_data[price_type] = raw_data[price_type].dropna()
    returns_data[price_type] = raw_data[price_type]['Adj Close']


#-------------------------------------------------------------------
#This section generates the statistical summary of the price data
returns_dataframe = pd.concat([returns_data['index'], returns_data['oil'], returns_data['coal'], returns_data['gas']], axis=1)
returns_dataframe.rename(columns={
    'S&PSEA40INDEX': 'S&P SEA 40 Index',
    'BZ=F': 'Crude Oil',
    'MTFZ24.NYM': 'Coal',
    'NG=F': 'Natural Gas'
}, inplace=True)
returns_dataframe = returns_dataframe.dropna()
raw_summary = returns_dataframe.describe()
sorted_summary = raw_summary.loc[['min', 'max', 'mean', 'std']]
sorted_summary = sorted_summary.transpose() #Transpose the matrix to reverse the data axis in the table
#-------------------------------------------------------------------
#Augmented Dickey-Fuller Test

def is_stationary(pval, sig_lvl=0.05):      #Check if data point is stationary or not (stationary if p-value < 0.05)
    return "Stationary" if pval<sig_lvl else "Non-stationary"

# adf_results = {}
# print(returns_dataframe.var())
# print("test")
# print(returns_dataframe['S&P SEA 40 Index'])
# for column in returns_dataframe.columns:
#     adf = ADF(returns_dataframe[column], trend='ct', method='t-stat')
#     adf_results[column] = {
#         "t-statistic": adf.stat,
#         "p-value": adf.pvalue,
#         "conclusion": is_stationary(adf.pvalue)
#     }
# adf_results_summary = pd.DataFrame(adf_results).T
# print(adf_results_summary)
adf = ADF(raw_data['oil']['Adj Close'])
print(adf.summary().as_text())

#-------------------------------------------------------------------
#This section writes the all the processed data into excel .xlsx file under Output folder
try:
    with pd.ExcelWriter(output_location) as writer:
        sorted_summary.to_excel(writer, sheet_name="Descriptive")  
        print("[*] Descriptive Summary Generated at: Output/processed_output.xlsx")
        adf_results_summary.to_excel(writer, sheet_name="ADF Results")
        print("[*] ADF Test Summary Generated at: Output/processed_output.xlsx")
except:
    print("[*] Something went wrong with writing the excel file")


#-------------------------------------------------------------------
#This section calculates GARCH 1,1

dependent_variable = returns_dataframe['S&P SEA 40 Index']
independent_variable = returns_dataframe[['Crude Oil', 'Coal', 'Natural Gas']]
#independent_variable = returns_dataframe['Coal']
model = arch_model(
    dependent_variable,
    x=independent_variable,
    mean='ARX',       
    vol='Garch',
    p=1,
    q=1,
)
garch_result = model.fit()

#print(dependent_variable.index)
#print(independent_variable.index)
print(garch_result.summary())
#print(garch_result.pvalues)
returns_dataframe['S&P SEA 40 Index'].to_csv('index.csv')  
print("[*] Descriptive Summary Generated at: Output/processed_output.csv")
returns_dataframe['Crude Oil'].to_csv('return_oil.csv')
raw_data['oil'].to_csv('raw_oil.csv')
print("[*] ADF Test Summary Generated at: Output/processed_output.csv")
