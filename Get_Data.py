import pandas as pd
import os

EQ = pd.read_csv('Equity Data.csv', index_col = 0)
IR = pd.read_csv('IR Data.csv', index_col = 0)
data_dir = r'Data'

def get_equity_series(tickers, start_date, end_date):
    available_tickers = EQ.columns
    
    if type(tickers) == str:
        if tickers not in available_tickers:
            print(tickers, 'is not in the database')
        else:
            res_ticker = tickers
            return EQ[res_ticker].loc[(EQ.index >= start_date) & (EQ.index <= end_date)]
        
    elif type(tickers) == list:
        res_ticker = []
        NA_ticker = []
        for ticker in tickers:
            if ticker in available_tickers:
                res_ticker += [ticker]
            else:
                NA_ticker += [ticker]
        if NA_ticker != []:
            print(NA_ticker, 'are not available in the database')
        return EQ[res_ticker].loc[(EQ.index >= start_date) & (EQ.index <= end_date)]
    
def get_option_series(ticker, stirke, call_put, exp_date, start_date, end_date):
    available_tickers = EQ.columns
    
    if ticker not in available_tickers:
        print(ticker, 'is not in the database')
    else:
        dates = os.listdir(data_dir)
        folders = [os.path.join(data_dir, date) for date in dates]
        res = pd.DataFrame()
        for index in range(len(folders)):
            try:
                df = pd.read_csv(os.path.join(folders[index],ticker,ticker+'_opt.csv'))
                df = df.loc[(df['strike'] == strike) & (df['exp_date'] == exp_date) &
                            (df['call_put'] == call_put)]
                df['date'] = dates[index]
                res = res.append(df)
            except:
                print(ticker, 'on', dates[index], 'is not available')
        res = res.set_index('date')
        col_order = ['contractSymbol','strike','exp_date', 'call_put','bid','ask',
                     'lastPrice','lastTradeDate','volume','time to maturity','implied volatility']
        res = res[col_order]
        return res


    

if __name__ == '__main__':
    tickers = 'AAPL'
    call_put = 'call'
    strike = 120
    exp_date = '2021-04-01'
    start_date = '2021-02-02'
    end_date = '2021-03-17'
    temp_eq_series = get_equity_series(tickers, start_date, end_date)
    temp_opt_series = get_option_series(tickers, strike, call_put, exp_date, start_date, end_date)
    print(temp_eq_series)
    print(temp_opt_series)