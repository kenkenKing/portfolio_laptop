import pandas as pd
import numpy as np
import datetime
import os
os.chdir('C:\Automation_Trading')
import matplotlib.pyplot as plt
import BS
import Tree
import sys

if datetime.datetime.today().weekday() in [5,6]:
    sys.exit()

def Bolinger_Band(EQ, bb_rolling_window = 20, bb_n_std = 2):
    BB_MA = EQ.rolling(bb_rolling_window, min_periods = 0).mean()
    BB_STD = EQ.rolling(bb_rolling_window, min_periods = 0).std()
    BB_UBOUND = BB_MA + bb_n_std * BB_STD
    BB_LBOUND = BB_MA - bb_n_std * BB_STD
    
    return (BB_LBOUND, BB_UBOUND)


def underlying_plot(df, start_date, end_date, bb_rolling_window = 20, bb_n_std = 2,
                    legend_box_size = 1, legend_label_size = 1,
                    save = True):
    
    BB_LBOUND, BB_UBOUND = Bolinger_Band(df, bb_rolling_window, bb_n_std)
    BB_LBOUND = BB_LBOUND.loc[(BB_LBOUND.index >= start_date) & (BB_LBOUND.index <= end_date)]
    BB_UBOUND = BB_UBOUND.loc[(BB_UBOUND.index >= start_date) & (BB_UBOUND.index <= end_date)]
    df_ = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    
    for ticker in df.columns:
        new_df = pd.DataFrame(columns = ['LBOUND', 'PRICE', 'UBOUND'])
        new_df['LBOUND'] = BB_LBOUND[ticker]
        new_df['PRICE'] = df_[ticker]
        new_df['UBOUND'] = BB_UBOUND[ticker]
        
        plt.figure(figsize=(15, 12))
#        df_[ticker].plot()
        new_df.plot()
        plt.xlabel('Time')
        plt.ylabel('Price over time')
        plt.legend(loc="upper left", 
                   borderpad = legend_box_size, labelspacing = legend_label_size)
        
        ## the date part need to be changed according to end _date adjustments
        if save == True:
            end_date_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
            today_str = (end_date_date + datetime.timedelta(days = -1)).isoformat()
            today_folder = os.path.join('Data', today_str)
            
            folder = os.path.join(today_folder, ticker)
            if not os.path.isdir(folder):
                os.mkdir(folder)
            plt.savefig(os.path.join(folder, ticker + '.png'))

def data_statistics(EQ, IR, start_date, end_date, price_type, 
                    bb_rolling_window = 20, bb_n_std = 2,
                    legend_box_size = 1, legend_label_size = 1,
                    plot_save = True):
    
    '''
    Returns a data frame of statistics of tickers on the end date.
    
    Statistics includes:
        mean of log returns
        standard diviation of log returns
        Sharpe ratio of the underlying
        Bolinger Band upper bound and lower bound
    
    Inputs:
        EQ: dataframe of the equity data of today
        
        IR: dataframe of the IR data of today
        
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
    
    EQ = EQ.loc[(EQ.index >= start_date) & (EQ.index <= end_date)]
    IR = IR.loc[(IR.index >= start_date) & (IR.index <= end_date)]
    
    underlying_plot(EQ, start_date, end_date, legend_box_size = legend_box_size, 
                    legend_label_size = legend_label_size, save = plot_save)
    
#    underlying_plot(IR, start_date, end_date, legend_box_size = legend_box_size, 
#                    legend_label_size = legend_label_size, save = plot_save)
    
#    https://quant.stackexchange.com/questions/33076/how-to-calculate-daily-risk-free-rate-using-13-week-treasury-bill
    nhi = 90
    d = IR['^IRX'][-1]/100
    rf = ( 1.0 / (1 - d * nhi/360.0) )**(1/nhi) - 1
    
    #### get log returns
    log_returns = np.log(EQ.iloc[1:].values/ EQ.iloc[:-1])
    log_returns.index = EQ.index[1:]

    #### annualized mean of return
    log_return_mean = log_returns.mean()*252

    #### annualized std of return
    log_return_std = log_returns.std()*np.sqrt(252)
    
    #### correlation matrix
    correlation_mat = log_returns.corr()
    
    #### covariance matrix
    cov_mat = 252*log_returns.cov()

    #### sharp ratio
    spr = (log_return_mean - rf)/log_return_std
    
    #### Bolinger Band
    BB_LBOUND, BB_UBOUND = Bolinger_Band(EQ, bb_rolling_window = bb_rolling_window, bb_n_std = bb_n_std)

    df['close price'] = EQ.iloc[-1]
    df['annualized mean of return'] = log_return_mean
    df['annualized std of return'] = log_return_std
    df['sharpe ratio'] = spr
    df['Bolinger Band Upper'] = BB_UBOUND.iloc[-1]
    df['Bolinger Band Lower'] = BB_LBOUND.iloc[-1]
    
    df['Date'] = EQ.index[-1]
    df = df.round(decimals = 5)
    return df, correlation_mat, cov_mat

def generate_daily_results(df, valuation_date = datetime.date.today().isoformat()):
    
    '''
    Please insert a doc string
    '''
    
    today_folder = os.path.join('Data', valuation_date)
    today_folder_exist = os.path.isdir(today_folder)
    if not today_folder_exist:
        os.mkdir(today_folder)

    data_name = 'data_stat_' + valuation_date + '.csv'
    corr_name = 'corr_matrix_' + valuation_date + '.csv'
    cov_name = 'cov_matrix_' + valuation_date + '.csv'
    if df[0]['Date'].iloc[0] == valuation_date:
        df[0].to_csv(os.path.join(today_folder, data_name))
        df[1].to_csv(os.path.join(today_folder, corr_name))
        df[2].to_csv(os.path.join(today_folder, cov_name))
    else:
        print('statistic is not generated since data is not from today')


def get_IV(EQ, IR, valuation_date = datetime.date.today().isoformat(),
            IR_idx = '^IRX'):
    
    
    today_folder = os.path.join('Data', valuation_date)
    
    #### get IR
    rf = IR.loc[valuation_date][IR_idx]
    
    tickers = EQ.columns
    
    for ticker in tickers:
        try:
            opt_chain_loc = os.path.join(os.path.join(today_folder, ticker), ticker + '_opt.csv')
            opt_chain = pd.read_csv(opt_chain_loc)
            if opt_chain.empty:
                print(ticker, 'has empty option chain. Calculation of IV terminated')
                continue
            else:
                print('processing option chain data for', ticker)
        except:
            print(ticker, 'does not have option chain')
            continue
        
        ##### get last spot
        spot = EQ.loc[valuation_date][ticker]
        
        #### get div
        div = 0
        
        #### get time to marutrity
        opt_chain['exp_date'] = opt_chain['exp_date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
        valuation_date_dt = datetime.datetime.strptime(valuation_date, '%Y-%m-%d').date()
        opt_chain['time to maturity'] = (opt_chain['exp_date'] - valuation_date_dt).apply(lambda x: x.days) / 365
        
        
        def tree_iv(s, x, t, sigma, rf, div, n, call_put, style, mkt_price, err_thsld = 0.01):
            if style == 'american':
                opt = Tree.TreeOption(s, x, t, sigma, rf, div, n, call_put, style)
                return opt.implied_volatility(mkt_price, err_thsld)
            elif style == 'european':
                opt = BS.BSOption(s, x, t, sigma, rf, div, call_put)
                return opt.implied_volatility(mkt_price, err_thsld)
        n_steps = 20
        opt_chain['implied volatility'] = opt_chain.apply(lambda x: tree_iv(spot, x['strike'], x['time to maturity'], 0.2,
                                                                             rf, div, n_steps, x['call_put'], 'american', 
                                                                             x['lastPrice']), axis = 1)
        opt_chain.to_csv(opt_chain_loc, index = False)
        
        #### implied volatility grid
        expiry = np.sort(opt_chain['exp_date'].unique())
        strikes = np.sort(opt_chain['strike'].unique())
        
        opt_chain_call = opt_chain.loc[opt_chain['call_put'] == 'call']
        opt_chain_put = opt_chain.loc[opt_chain['call_put'] == 'put']

        iv_df_list = [opt_chain_call, opt_chain_put]
        for df in iv_df_list:
            
            iv_grid = pd.DataFrame(index = strikes)
            
            if df.empty:
                continue
    
            if df['call_put'].values[0] == 'call':
                iv_grid_loc = os.path.join(os.path.join(today_folder, ticker), ticker + '_call_iv.csv')
            else:
                iv_grid_loc = os.path.join(os.path.join(today_folder, ticker), ticker + '_put_iv.csv')

                
            for exp_date in expiry:
                iv_df = df.loc[df['exp_date'] == exp_date]
                iv_df = iv_df.drop_duplicates(subset = ['strike'])
                
                if len(iv_df['implied volatility'].unique()) == 1 and np.isnan(iv_df['implied volatility'].unique()[0]):
                    continue
                
                elif len(iv_df['implied volatility'].unique()) == 2:
                    iv_df = iv_df.rename(columns={'implied volatility': exp_date})
                    iv_grid = pd.merge(iv_grid, iv_df[['strike', exp_date]], left_index=True, 
                                            right_on='strike',how='left').set_index('strike')      
                    try:
                        iv_grid = iv_grid.bfill().ffill()
                    except:
                        print(ticker, 'call options on', exp_date.isoformat(), 
                              'do not have missing IV to interpolate/extrapolate')
                        continue
                    
                else:
                    iv_df = iv_df.rename(columns={'implied volatility': exp_date})
                    iv_grid = pd.merge(iv_grid, iv_df[['strike', exp_date]], left_index=True, 
                                            right_on='strike',how='left').set_index('strike')      
                    try:
                        iv_grid = iv_grid.interpolate(method='linear', axis=0).bfill().ffill()
                    except:
                        print(ticker, 'call options on', exp_date.isoformat(), 
                              'do not have missing IV to interpolate/extrapolate')
                        continue

            iv_grid.to_csv(iv_grid_loc)
        

#### execuction

wk_dir = r'C:\Automation_Trading'
os.chdir(wk_dir)
today_str = datetime.date.today().isoformat()
EQ = pd.read_csv('Equity Data.csv', index_col = 0)
IR = pd.read_csv('IR Data.csv', index_col = 0)
price_type = 'Close' # 'Close', 'Adj Close', 'High', 'Low', 'Open', 'Volume'
end_date = (datetime.date.today() + datetime.timedelta(days = 1)).isoformat()
start_date = (datetime.date.today() + datetime.timedelta(days = -365)).isoformat()
plot_save = True

dfs = data_statistics(EQ, IR, start_date, end_date, price_type, plot_save = plot_save)
generate_daily_results(dfs, today_str)
get_IV(EQ, IR, valuation_date = today_str)
