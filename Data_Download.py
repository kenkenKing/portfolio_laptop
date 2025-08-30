import yfinance as yf
import pandas as pd
import datetime
import os
import sys

if datetime.datetime.today().weekday() in [5,6]:
    print('today is weekend, no data will be downloaded')
    sys.exit()

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
# wk_dir = r'C:\Automation_Trading'
# os.chdir(wk_dir)

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
        data = data[price_type].round(3)
        data.to_csv('IR Data.csv')
    elif underlying_type == 'Equity':
        data = data[price_type].round(2)
        data.to_csv('Equity Data.csv')
    return data

def option_data_download(tickers_yf, fwd_days, strike_pct = 0.15, max_contract_per_day = 30):
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
            
            # in case some equities delisted
            last_spot = last_price[ticker]
            if last_spot != last_spot:
                print(ticker, 'is no longer available due to corporate actions, no option data will be downloaded')
                continue
            
            for exp_date in opt_dates:
                temp_tick_opt = temp_tick.option_chain(exp_date)
                
#                ['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 
#                 'volume', 'openInterest','impliedVolatility', 'inTheMoney', 'contractSize', 'currency']
                temp_tick_call = temp_tick_opt.calls[['contractSymbol', 'lastTradeDate', 'volume', 'strike', 'bid', 'ask', 'lastPrice']]
                temp_tick_put = temp_tick_opt.puts[['contractSymbol', 'lastTradeDate', 'volume', 'strike', 'bid', 'ask', 'lastPrice']]
                
                temp_tick_call = temp_tick_call.loc[(temp_tick_call['strike'] < (1+strike_pct) * last_spot) &
                                                    (temp_tick_call['strike'] > (1-strike_pct)* last_spot)]
                temp_tick_call['call_put'] = 'call'
                temp_tick_call['exp_date'] = exp_date
                
                temp_tick_put = temp_tick_put.loc[(temp_tick_put['strike'] < (1+strike_pct) * last_spot) &
                                                  (temp_tick_put['strike'] > (1-strike_pct)* last_spot)]
                temp_tick_put['call_put'] = 'put'
                temp_tick_put['exp_date'] = exp_date
                
                contract_count = temp_tick_call.shape[0]
                if contract_count > int(max_contract_per_day/2):
                    extra = contract_count - max_contract_per_day
                    half_way = int(contract_count/2)
                    
                    temp1 = temp_tick_call.iloc[int(extra/2):half_way,:]
                    temp2 = temp_tick_call.iloc[int(extra/2)+half_way:,:]
                    temp_tick_call = pd.concat([temp1, temp2])
                    
                    temp1 = temp_tick_put.iloc[int(extra/2):half_way,:]
                    temp2 = temp_tick_put.iloc[int(extra/2)+half_way:,:]
                    temp_tick_put = pd.concat([temp1, temp2])
                
                temp_tick_opt = pd.concat([temp_tick_call, temp_tick_put])
                temp_tick_opt_all = pd.concat([temp_tick_opt_all, temp_tick_opt])
                
                today = datetime.date.today().isoformat()
                today_folder = os.path.join('Data', today)
                today_folder_exist = os.path.isdir(today_folder)
                if not today_folder_exist:
                    os.mkdir(today_folder)
                    
                today_opt_ticker_folder = os.path.join(today_folder,ticker)
                if not os.path.isdir(today_opt_ticker_folder):
                    os.mkdir(today_opt_ticker_folder)
                
            temp_tick_opt_all.to_csv(os.path.join(today_opt_ticker_folder, ticker + '_opt.csv'),index = False)
   
        except:
            print(ticker, 'does not have options traded at the counter')


price_type = 'Close' # 'Close', 'Adj Close', 'High', 'Low', 'Open', 'Volume'
data_download(tickers_yf, price_type, 'Equity')
data_download(IR_yf, price_type, 'IR')
option_data_download(tickers_yf, 240, strike_pct = 3, max_contract_per_day = 40)