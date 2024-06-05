import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm

amd = pd.read_csv('AMD_5YR.csv')
amd.head()
amd.tail()
#5 year data from 6/3/2019 to 5/31/24

#renaming the close/last column to just close
amd = amd.rename(columns = {'Close/Last':'Close'})
amd.head()
print(amd.dtypes)

#I need to change the date to datetime and the objects to float
#To convert to float, we remove the dollar sign 
amd[['Close', 'Open', 'High', 'Low']] = amd[['Close', 'Open', 'High', 'Low']].replace('[\$,]', '', regex=True).astype(float)
print(amd.dtypes)

#I want to have a price diff column on this dataset
amd['PriceDiff'] = amd['Close'].shift(-1) - amd['Close']
amd['DailyReturn'] = amd['PriceDiff']/amd['Close']
amd.dropna(inplace=True)
# Convert 'Date' column to datetime
amd['Date'] = pd.to_datetime(amd['Date'])  
amd.set_index('Date', inplace=True)
#Data needs sorting
amd.sort_values(by='Date', inplace=True)
#Line graph of the closing data 2019-2024 
amd.loc['01-01-2019':'12-31-2019', 'Close'].plot(label='2019')#2019
amd.loc['01-01-2020':'12-31-2020', 'Close'].plot(label='2020') #2020
amd.loc['01-01-2021':'12-31-2021', 'Close'].plot(label='2021') #2021
amd.loc['01-01-2022':'12-31-2022', 'Close'].plot(label='2022') #2022
amd.loc['01-01-2023':'12-31-2023', 'Close'].plot(label='2023') #2023
amd.loc['01-01-2024':'12-31-2024', 'Close'].plot(label='2024') #2024
plt.legend()

#fitting a normal distribution on the daily returns
mu, sigma = stats.norm.fit(amd['DailyReturn'])

plt.figure(figsize=(15, 8))
plt.hist(amd['DailyReturn'], bins=50, density=True, alpha=0.7, color='blue', label='Histogram')

x = np.linspace(amd['DailyReturn'].min(), amd['DailyReturn'].max(), 1000)
#probability density function (PDF) values
pdf = stats.norm.pdf(x, mu, sigma)

# Plot the fitted normal distribution
plt.plot(x, pdf, 'r-', lw=2, label='Fitted Normal Distribution')

plt.title('Histogram and Fitted Normal Distribution of Daily Return')
plt.xlabel('Daily Return')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
#looks great and fits normal

#I am exploring the log of the daily returns and see the nature of this
amd['LogReturn'] = np.log(amd['PriceDiff'])
amd.dropna(inplace=True)
amd.head()

log_return = amd['LogReturn'][np.isfinite(amd['LogReturn'])]

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(log_return, bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title('Log Returns Distribution Histogram')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


#there are values that are infinite. Filter them out to see the distribtuion

log_return = amd['LogReturn'][np.isfinite(amd['LogReturn'])]

# Calculate mean and standard deviation for LogReturn
mu = log_return.mean()
sigma = log_return.std(ddof=1)

# Create density DataFrame for PDF
density = pd.DataFrame()
density['X'] = np.linspace(log_return.min(), log_return.max(), 1000)  
density['PDF'] = norm.pdf(density['X'], mu, sigma)

# Plot histogram with PDF
plt.figure(figsize=(15, 8))
log_return.hist(bins=50, density=True, alpha=0.6, color= 'blue', range = (-4, 4))  
plt.plot(density['X'], density['PDF'], color='red')
plt.xlabel('Log Return')
plt.ylabel('Density')
plt.title('Probability Distribution of Log Returns')
plt.grid(True)
plt.show()

#I need to assess the goodness of fit
#First a qq plot

import statsmodels.api as sm

fig, ax = plt.subplots(figsize=(8, 6))
sm.qqplot(log_return, line='s', ax=ax) 
plt.title('Q-Q Plot of Log Returns')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()
#close to getting this to a normal distribution but I think the norm of daily returns is already a great fit

#the pricediff fits the lognorm distribution
#I will calculate the VaR and will use historical simulation to estimate Var and CVaR at different confidence intervals.
mu = amd['DailyReturn'].mean()
sigma = amd['DailyReturn'].std(ddof=1)
prob_return = norm.cdf(-0.05, mu, sigma)
print(prob_return)
#probability that the AMD drops over 5% in a day is around 6.8%

#I want to explore what some other scenerios.
mu220 = 220*mu
sigma220 = (220**.05) * sigma
prob_dropping_20percent = stats.norm.cdf(-0.2, mu220, sigma220)
print(prob_dropping_20percent)
#AMD has a 45.80% chance of dropping more that 20%. 

VaR = norm.ppf(.05, mu, sigma)
print('Single Day Value At Risk: ', VaR)

#Monte Carlo simulation for the AMD Daily Returns
#mu and sigma is already calculated 
mc_sims = 1000
T = 100

simulated_returns = np.zeros((mc_sims, T))

for i in range(mc_sims): 
    rand_returns = np.random.normal(mu, sigma, T)   
    simulated_returns[i] = np.cumprod(1 + rand_returns)

simulated_returns_df = pd.DataFrame(simulated_returns)

def mcVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError('Expected a pandas data series.')

def mcCVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVar = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVar].mean()
    else:
        raise TypeError('Expected a pandas data series.')
        
#slice the last row of the simulation

portResults = pd.Series(simulated_returns_df.iloc[-1])

VaR = mcVaR(portResults, alpha=5)
CVaR = mcCVaR(portResults, alpha=5)

print("Value at Risk (VaR):", VaR)
print("Conditional Value at Risk (CVaR):", CVaR)

# Plot simulated returns with the VaR line and CVaR line
plt.figure(figsize=(15, 8))
for i in range(mc_sims):
    plt.plot(simulated_returns_df.iloc[i], color='blue', alpha=0.1)  
plt.axhline(y=VaR, color='red', linestyle='--', label='VaR')
plt.axhline(y=CVaR, color='green', linestyle='--', label='CVaR')
plt.title('Monte Carlo Simulation of AMD Cummulative Daily Returns with VaR and CVaR')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()

