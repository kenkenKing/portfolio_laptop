# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 03:01:27 2021

@author: cjh93
"""

import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import os
import BS



def read_ticker(path):
    '''
    Create lists of the list of tickers.
    First one will be a list of tickers as elements. 
    Second one will be a string of tickers that is readable by yfinance
    
    Input:
        String of the path.

    Output:
        Tuple of length 2.
        First element is the list of tickers as elements
        Second will be a string of tickers that is readable by yfinance
    '''
    

    TICKERS_TO_WATCH = path
    f = open(TICKERS_TO_WATCH, "r")
    tickers_to_watch = f.read().split('\n')
    tickers_to_watch_yf = ' '.join(tickers_to_watch)
    f.close()
    return (tickers_to_watch, tickers_to_watch_yf)

#### change directroy
wk_dir = r'C:\Automation_Trading'
os.chdir(wk_dir)

ticker_path = 'TICKERS_TO_WATCH.txt'
IR_path = 'IR.txt'

tickers_to_watch = read_ticker(ticker_path)
IR_to_watch = read_ticker(IR_path)

tickers = tickers_to_watch[0]
tickers_yf = tickers_to_watch[1]

IR = IR_to_watch[0]
IR_yf = IR_to_watch[1]

def data_download(tickers_yf, price_type, underlying_type):
    data = yf.download(tickers_yf)
    if underlying_type == 'IR':
        data[price_type].to_csv('IR Data.csv')
    elif underlying_type == 'Equity':
        data[price_type].to_csv('Equity Data.csv')
    return data[price_type]

def option_data_download(tickers_yf, fwd_days, strike_pct = 0.1):
    '''
    doc string
    '''
    
    tickers = tickers_yf.split(' ')
    last_price = pd.read_csv(r'Equity Data.csv').iloc[-1]
    fwd_date = datetime.date.today() + datetime.timedelta(days = fwd_days)
    
    for ticker in tickers:
        print('getting option data of', ticker)
        try:
            temp_tick = yf.Ticker(ticker)
            temp_tick_exp = temp_tick.options
            opt_dates = [date for date in temp_tick_exp if datetime.datetime.strptime(date, '%Y-%m-%d').date() < fwd_date]
            temp_tick_opt_all = pd.DataFrame()
            
            for exp_date in opt_dates:
                temp_tick_opt = temp_tick.option_chain(exp_date)
                temp_tick_call = temp_tick_opt.calls[['contractSymbol', 'lastTradeDate', 'strike', 'bid', 'ask', 'lastPrice']]
                temp_tick_put = temp_tick_opt.puts[['contractSymbol', 'lastTradeDate','strike', 'bid', 'ask', 'lastPrice']]
                
                last_spot = last_price[ticker]
                temp_tick_call = temp_tick_call.loc[(temp_tick_call['strike'] < (1+strike_pct) * last_spot) &
                                                    (temp_tick_call['strike'] > (1-strike_pct)* last_spot)]
                temp_tick_call['put_call'] = 'call'
                temp_tick_call['exp_date'] = exp_date
                
                temp_tick_put = temp_tick_put.loc[(temp_tick_put['strike'] < (1+strike_pct) * last_spot) &
                                                  (temp_tick_put['strike'] > (1-strike_pct)* last_spot)]
                temp_tick_put['put_call'] = 'put'
                temp_tick_put['exp_date'] = exp_date
                
                temp_tick_opt = pd.concat([temp_tick_call, temp_tick_put])
                temp_tick_opt_all = pd.concat([temp_tick_opt_all, temp_tick_opt])
                
                today = datetime.date.today().isoformat()
                today_folder_exist = os.path.isdir(today)
                if not today_folder_exist:
                    os.mkdir(today)
                    
                today_opt_ticker_folder = os.path.join(today,ticker)
                if not os.path.isdir(today_opt_ticker_folder):
                    os.mkdir(today_opt_ticker_folder)
                
            temp_tick_opt_all.to_csv(os.path.join(today_opt_ticker_folder, ticker + '_opt.csv'),index = False)
   
        except:
            print(ticker, 'does not have options traded at the counter')
            
option_data_download(tickers_yf, 180, strike_pct = 0.1)

def data_statistics(tickers_yf, IR_yf, start_date, end_date, price_type, 
                    bb_rolling_window = 20, bb_n_std = 2):
    
    '''
    Returns a data frame of statistics of tickers on the end date.
    
    Statistics includes:
        mean of log returns
        standard diviation of log returns
        Sharpe ratio of the underlying
        Bolinger Band upper bound and lower bound
    
    Inputs:
        tickers_yf: a string in yfinance form that includes series of equity tickers
        
        IR_yf: a string in yfinance form that includes series of interest rate tickers
        
        start_date: a string of ISO format date, the start date of the data for calculation purpose
        
        end_date: a string of ISO format date, the end date of the data for calculation purpose
        
        price_type: string values, the price type we want to use from the data downloaded from yfinance
            possible values are: 'Close', 'Adj Close', 'High', 'Low', 'Open', 'Volume'. Also note that 
            'Volume' option do not generate meaningful statistic results.
        
        bb_rolling_window: integer, the rolling moving window to calculate the Bolinger Band. It should not be 
            greater or equalt to the date difference from then end_date to the start_date. Default value is 20.
            
        bb_n_std: integer, the number of standard diviations from the mean to derive the Bolinger Band bounds.
            Default value is 2.
    '''
    
    df = pd.DataFrame()
    
    EQUITY = data_download(tickers_yf, price_type, 'Equity')
    IR = data_download(IR_yf, price_type, 'IR')
    
    EQUITY = EQUITY.loc[EQUITY.index > start_date]
    IR = IR.loc[IR.index > start_date]
    
    
#    https://quant.stackexchange.com/questions/33076/how-to-calculate-daily-risk-free-rate-using-13-week-treasury-bill
    nhi = 90
    d = IR['^IRX'][-1]/100
    rf = ( 1.0 / (1 - d * nhi/360.0) )**(1/nhi) - 1
    
    #### get log returns
    log_returns = np.log(EQUITY.iloc[1:].values/ EQUITY.iloc[:-1])
    log_returns.index = EQUITY.index[1:]

    #### annualized mean of return
    log_return_mean = log_returns.mean()*252

    #### annualized std of return
    log_return_std = log_returns.std()*np.sqrt(252)

    #### sharp ratio
    spr = (log_return_mean - rf)/log_return_std
    
    #### Bolinger Band
    BB_MA = EQUITY.rolling(bb_rolling_window).mean()
    BB_STD = EQUITY.rolling(bb_rolling_window).std()
    BB_UBOUND = BB_MA + bb_n_std * BB_STD
    BB_LBOUND = BB_MA - bb_n_std * BB_STD
    
    df['close price'] = EQUITY.iloc[-1]
    df['annualized mean of return'] = log_return_mean
    df['annualized std of return'] = log_return_std
    df['sharpe ratio'] = spr
    df['Bolinger Band Upper'] = BB_UBOUND.iloc[-1]
    df['Bolinger Band Lower'] = BB_LBOUND.iloc[-1]
    
    df['Date'] = EQUITY.index[-1]
    df = df.round(decimals = 5)
    return df

end_date = (datetime.date.today() + datetime.timedelta(days = 1)).isoformat()
start_date = (datetime.date.today() + datetime.timedelta(days = -365)).isoformat()
price_type = 'Close' # 'Close', 'Adj Close', 'High', 'Low', 'Open', 'Volume'
df = data_statistics(tickers_yf, IR_yf, start_date, end_date, price_type)

def generate_daily_results(df):
    
    '''
    Please insert a doc string
    '''
    
    today = datetime.date.today().isoformat()
    today_folder_exist = os.path.isdir(today)
    if not today_folder_exist:
        os.mkdir(today)

    data_name = 'data_stat_' + today+ '.csv'
    if df['Date'].iloc[0].date().isoformat() == today:
        df.to_csv(os.path.join(today , data_name))
    else:
        print('statistic is not generated since data is not from today')
    
generate_daily_results(df)

option = BS.BSEuroCallOption(s = 100, x = 100, t = 1, sigma = 0.25, rf = 0.02, div = 0)
BS.calculate_implied_volatility(option, 10)




#msft = yf.Ticker("MSFT")
#msft_close = yf.download('MSFT')['Close'][-1]
##msft.options #this gives the expiration dates in the counter
#opt = msft.option_chain('2021-03-26')
#print(opt.calls.columns)
#
## impliedVolatility sucks
#opt.calls[['contractSymbol','strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']]
#opt.calls[['strike', 'impliedVolatility']]

#def drawdown(returns):
#    wealth_index = (1+returns).cumprod()
#    previous_peaks = wealth_index.cummax()
#    drawdowns = (wealth_index - previous_peaks)/previous_peaks
#    return pd.DataFrame({
#        "Wealth": wealth_index,
#        "Peaks": previous_peaks,
#        "Drawdown": drawdowns
#        })



#data['Close'][tickers[1]].plot()


#data['Adj Close']['IWM']

#ticker_list = ['TSLA', 'GE', 'BAC', 'IWM']
#ticker_list = ' '.join(ticker_list)
#tickers = yf.Ticker(ticker_list)
#
#data = yf.download(ticker_list, start="2016-01-01", end="2021-02-22")
#data['Adj Close']['IWM']


#msft = yf.Ticker("MSFT")
#
## get stock info
#msft.info
#
## get historical market data
#hist = msft.history(period="max")
#
## show actions (dividends, splits)
#msft.actions
#
## show dividends
#msft.dividends
#
## show splits
#msft.splits
#
## show financials
#msft.financials
#msft.quarterly_financials
#
## show major holders
#msft.major_holders
#
## show institutional holders
#msft.institutional_holders
#
## show balance sheet
#msft.balance_sheet
#msft.quarterly_balance_sheet
#
## show cashflow
#msft.cashflow
#msft.quarterly_cashflow
#
## show earnings
#msft.earnings
#msft.quarterly_earnings
#
## show sustainability
#msft.sustainability
#
## show analysts recommendations
#msft.recommendations
#
## show next event (earnings, etc)
#msft.calendar
#
## show ISIN code - *experimental*
## ISIN = International Securities Identification Number
#msft.isin
#
## show options expirations
#msft.options
#
## get option chain for specific expiration
##opt = msft.option_chain('YYYY-MM-DD')
#opt = msft.option_chain('2023-01-20')
#
#import yfinance as yf
#
#tickers = yf.Tickers('msft aapl goog')
## ^ returns a named tuple of Ticker objects
#
## access each ticker using (example)
#tickers.tickers.MSFT.info
#tickers.tickers.AAPL.history(period="1mo")
#tickers.tickers.GOOG.actions
#
#data = yf.download("SPY AAPL", start="2017-01-01", end="2021-02-22")
#
#data = yf.download(  # or pdr.get_data_yahoo(...
#        # tickers list or string as well
#        tickers = "SPY AAPL MSFT",
#
#        # use "period" instead of start/end
#        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#        # (optional, default is '1mo')
#        period = "ytd",
#
#        # fetch data by interval (including intraday if period < 60 days)
#        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
#        # (optional, default is '1d')
#        interval = "1m",
#
#        # group by ticker (to access via data['SPY'])
#        # (optional, default is 'column')
#        group_by = 'ticker',
#
#        # adjust all OHLC automatically
#        # (optional, default is False)
#        auto_adjust = True,
#
#        # download pre/post regular market hours data
#        # (optional, default is False)
#        prepost = True,
#
#        # use threads for mass downloading? (True/False/Integer)
#        # (optional, default is True)
#        threads = True,
#
#        # proxy URL scheme use use when downloading?
#        # (optional, default is None)
#        proxy = None
#    )