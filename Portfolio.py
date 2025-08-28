import pandas as pd
import os
import numpy as np
import datetime
from scipy.optimize import minimize
import matplotlib.pyplot as plt


data_path = 'Data'
EQportfolio_path = 'Equity_Portfolio.txt'
f = open(EQportfolio_path, "r")
EQportfolio = f.read().split('\n')
f.close()

# to do: to add Black Litterman model

def get_portfolio_stats(portfolio, stat_date = datetime.date.today().isoformat(), IR_ticker = '^IRX'):
    
    #### get the last available date of data stats
    avai_date = stat_date
    while os.path.isdir(os.path.join(data_path,avai_date)) == False:
        avai_date = datetime.datetime.strptime(avai_date, '%Y-%m-%d').date() - datetime.timedelta(days = 1)
        avai_date = avai_date.isoformat()
        
    data_stats_path = os.path.join(data_path,avai_date,'data_stat_' + avai_date + '.csv')
    correlation_path = os.path.join(data_path,avai_date,'corr_matrix_' + avai_date + '.csv')
    cov_path = os.path.join(data_path,avai_date,'cov_matrix_' + avai_date + '.csv')
    
    data_stats = pd.read_csv(data_stats_path, index_col = 0)
    corr_mat = pd.read_csv(correlation_path, index_col = 0)
    IR = pd.read_csv('IR Data.csv', index_col = 0).loc[avai_date][IR_ticker]
    IR = 0 if pd.isnull(IR) else IR
    cov_mat = pd.read_csv(cov_path, index_col = 0)
    
    close_price = data_stats.loc[EQportfolio]['close price']
    mean = data_stats.loc[EQportfolio]['annualized mean of return']
    std = data_stats.loc[EQportfolio]['annualized std of return']
    corr = corr_mat.loc[EQportfolio][EQportfolio]
    cov = cov_mat.loc[EQportfolio][EQportfolio]
    return close_price, mean, std, corr, cov, IR

close_price, tick_mean, tick_std, tick_corr, tick_cov, IR = get_portfolio_stats(EQportfolio, stat_date = '2021-05-21')
nhi = 90
rf = ( 1.0 / (1 - IR/100 * nhi/360.0) )**(1/nhi) - 1

def portfolio_df(portfolio, weight):
    return pd.Series(data=weight,index=portfolio)

def portfolio_mean(weight):
    
    if type(weight) == np.ndarray:
        return np.dot(weight, np.array(tick_mean))
    
    weight_np = weight.to_numpy().reshape(len(weight),1)
    return weight_np.transpose().dot(tick_mean)[0]

def portfolio_variance(weight):
    
    if type(weight) == np.ndarray:
        return np.dot(np.dot(weight, np.array(tick_cov)), weight)
    
    weight_np = weight.to_numpy().reshape(len(weight),1)
    return weight_np.transpose().dot(tick_cov.to_numpy()).dot(weight_np)[0,0]

def portfolio_std(weight):
    return np.sqrt(portfolio_variance(weight))

def portfolio_sr(weight):
    port_mean = portfolio_mean(weight)
    port_std = np.sqrt(portfolio_variance(weight))
    return (port_mean - rf) / port_std

def indicator_vector(portfolio):
    length = len(portfolio)
    weight = [1 for i in range(length)]
    target_df = pd.Series(data = weight, index = portfolio)
    return target_df

def cholesky_decomp(cov):
    cov_mat = np.matrix(cov)
    return np.linalg.cholesky(cov_mat), np.linalg.cholesky(cov_mat).transpose()

##### portfolio functions
def artifitial_portfolio(weight_list, portfolio):
    '''
    this portfolio inputs w/e the weight you want to
    '''
    target_df = pd.Series(data = weight_list, index = portfolio)
    
    mean = portfolio_mean(target_df)
    var = portfolio_variance(target_df)
    std = portfolio_std(target_df)
    sr = portfolio_sr(target_df)
    return target_df, mean, var, std, sr

def equally_weighted_portfolio(portfolio):
    length = len(portfolio)
    weight = [1/length for i in range(length)]
    target_df = pd.Series(data = weight, index = portfolio)
    
    mean = portfolio_mean(target_df)
    var = portfolio_variance(target_df)
    std = portfolio_std(target_df)
    sr = portfolio_sr(target_df)
    return target_df, mean, var, std, sr

def minimum_variance_portfolio(portfolio, 
                               method = 'closed form', 
                               short_sell = True,
                               upper_bound = 1.0,
                               lower_bound = 0.0):
    if method == 'closed form':
        cov_inv = np.linalg.inv(tick_cov.to_numpy())
        indicator = indicator_vector(portfolio).to_numpy().reshape(len(portfolio),1)
        numerator = cov_inv.dot(indicator)
        denominator = indicator.transpose().dot(cov_inv).dot(indicator)
        
        weight = (numerator/denominator).reshape(len(portfolio),).tolist()
        target_df = pd.Series(data = weight, index = portfolio)

        mean = portfolio_mean(target_df)
        var = portfolio_variance(target_df)
        std = portfolio_std(target_df)
        sr = portfolio_sr(target_df)
        return target_df, mean, var, std, sr
    
    elif method == 'calibration':
        
        def obj_func(weight_list):
            weight_np = np.array(weight_list).reshape(len(weight_list),1)
            return weight_np.transpose().dot(tick_cov.to_numpy()).dot(weight_np)[0,0]
        
        Ini = equally_weighted_portfolio(portfolio)[0].tolist()
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        bounds = ((lower_bound, upper_bound),)*len(portfolio)
        if short_sell:
            a = minimize(obj_func, Ini,
                         method='SLSQP',
                         options={'maxiter': 1000},
                         constraints=(weights_sum_to_1),
                         )
        else:
            a = minimize(obj_func, Ini,
                         method='SLSQP',
                         options={'maxiter': 1000},
                         constraints=(weights_sum_to_1),
                         bounds = bounds
                         )
        weight = a.x.tolist()
        target_df = pd.Series(data = weight, index = portfolio)

        mean = portfolio_mean(target_df)
        var = portfolio_variance(target_df)
        std = portfolio_std(target_df)
        sr = portfolio_sr(target_df)
        return target_df, mean, var, std, sr

def mean_variance_portfolio(portfolio):
    mv_port, mv_mean, mv_var, mv_std, mv_sr = minimum_variance_portfolio(portfolio)
    l_cholesky, u_cholesky = cholesky_decomp(tick_cov)
    mean_diff_np = (tick_mean - indicator_vector(portfolio) * mv_mean).to_numpy().reshape(len(portfolio),1)
    MPoR = np.linalg.inv(l_cholesky).dot(mean_diff_np) # market price of risk
    weight = np.array(np.linalg.inv(u_cholesky).dot(MPoR)) #convert back to ndarray
    
    weight = weight.reshape(len(portfolio),).tolist()
    target_df = pd.Series(data = weight, index = portfolio)
    mean = portfolio_mean(target_df)
    var = portfolio_variance(target_df)
    std = portfolio_std(target_df)
    sr = portfolio_sr(target_df)
    return target_df, mean, var, std, sr

def frontier_portfolio(portfolio, 
                       target_mean = portfolio_mean(minimum_variance_portfolio(EQportfolio)[0]), 
                       method = 'closed form',
                       short_sell = True,
                       upper_bound = 1.0,
                       lower_bound = 0.0):
    if method == 'closed form':
        mv_port, mv_mean, mv_var, mv_std, mv_sr = minimum_variance_portfolio(portfolio)
        mean_var_port, mean_var_mean, mean_var_var, mean_var_std, mean_var_sr = mean_variance_portfolio(portfolio)
        l_cholesky, u_cholesky = cholesky_decomp(tick_cov)
        
        mean_diff_np = (tick_mean - indicator_vector(portfolio) * mv_mean).to_numpy().reshape(len(portfolio),1)
        MPoR_mv = np.linalg.inv(l_cholesky).dot(mean_diff_np)
        MPoR_mv = np.array(MPoR_mv)
        MPoR_mv_norm = np.linalg.norm(MPoR_mv)
        multiplier = (target_mean - mv_mean)/MPoR_mv_norm
        # this is not yet finished
        target_df = mv_port + multiplier*mean_var_port
        mean = portfolio_mean(target_df)
        var = portfolio_variance(target_df)
        std = portfolio_std(target_df)
        sr = portfolio_sr(target_df)
        return target_df, mean, var, std, sr
    
    elif method == 'calibration':
        
        def obj_func(weight_list):
            weight_np = np.array(weight_list).reshape(len(weight_list),1)
            return weight_np.transpose().dot(tick_cov.to_numpy()).dot(weight_np)[0,0]
        
        Ini = equally_weighted_portfolio(portfolio)[0].tolist()
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        mean_is_target = {'type': 'eq',
                          'fun': lambda weights: np.array(weights).dot(tick_mean.to_numpy()) - target_mean
                          }
        
        bounds = ((lower_bound, upper_bound),)*len(portfolio)
        if not short_sell:
            a = minimize(obj_func, Ini,
                         method='SLSQP',
                         options={'maxiter': 1000},
                         constraints=(weights_sum_to_1, mean_is_target),
                         bounds = bounds
             )
        else:
            a = minimize(obj_func, Ini,
                         method='SLSQP',
                         options={'maxiter': 1000},
                         constraints=(weights_sum_to_1, mean_is_target)
                         )
        weight = a.x.tolist()
        target_df = pd.Series(data = weight, index = portfolio)

        mean = portfolio_mean(target_df)
        var = portfolio_variance(target_df)
        std = portfolio_std(target_df)
        sr = portfolio_sr(target_df)
        return target_df, mean, var, std, sr

def frontier_portfolio_plot(portfolio,
                            method = 'closed form',
                            short_sell = True,
                            upper_bound = 1.0,
                            lower_bound = 0.0):
    min_mean = portfolio_mean(minimum_variance_portfolio(portfolio)[0])
    mean_grid = np.linspace(min_mean, 3*min_mean, num = 100, endpoint = True)
    std_grid_func = np.vectorize(lambda m: frontier_portfolio(
        portfolio=portfolio, target_mean=m, method=method, short_sell=short_sell, upper_bound=upper_bound, lower_bound=lower_bound)[3]
        )
    std_grid = std_grid_func(mean_grid)
    
    plt.plot(std_grid, mean_grid, label="Frontier Portfolio")
    plt.xlabel("standard deviation")
    plt.ylabel("target_mean")
    plt.title("Frontier Portfolio")
    plt.legend()
    plt.show()
    print(mean_grid)
    print(std_grid)
    return 

def maximum_sharpe_ratio_portfolio(portfolio,
                                   short_sell = True,
                                   upper_bound = 1.0,
                                   lower_bound = 0.0):
    
    Ini = equally_weighted_portfolio(portfolio)[0].tolist()    
    bounds = ((lower_bound, upper_bound),)*len(portfolio)
    
    nhi = 90
    rf = ( 1.0 / (1 - IR/100 * nhi/360.0) )**(1/nhi) - 1
    
    weights_sum_to_1 = {'type': 'eq',
                    'fun': lambda weights: np.sum(weights) - 1
                    }
    
    def neg_sharpe(weights, rf):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        temp_df = pd.Series(data = weights, index = portfolio)
        r = portfolio_mean(temp_df)
        vol = portfolio_std(temp_df)
        return -(r - rf)/vol
    
    if not short_sell:
        weights = minimize(neg_sharpe, Ini,
                       args=(rf), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds
                       )
    else:
        weights = minimize(neg_sharpe, Ini,
               args=(rf), method='SLSQP',
               options={'disp': False},
               constraints=(weights_sum_to_1,)
               )
    
    weight = weights.x.tolist()
    target_df = pd.Series(data = weight, index = portfolio)
    
    mean = portfolio_mean(target_df)
    var = portfolio_variance(target_df)
    std = portfolio_std(target_df)
    sr = portfolio_sr(target_df)
    return target_df, mean, var, std, sr


#### show all
def show_all_portfolio_stats(portfolio,
                             method = 'closed form',
                             short_sell = True,
                             upper_bound = 1.0,
                             lower_bound = 0.0):
    
    # to do: to plot mean and variance onto the frontier portfolio plot
    
    try:
        shares = [1,27.522935,9.839372,0.407249]
        shares = pd.Series(data = shares, index = portfolio)
        weight_list = shares*close_price/(shares*close_price).sum()
        arti_port, arti_mean, arti_var, arti_std, arti_sr = artifitial_portfolio(weight_list, portfolio)
        arti_port_df = pd.DataFrame(arti_port).transpose()
        print('artifitial portfolio weights:', arti_port_df, sep = '\n')
        print('artifitial portfolio mean:', arti_mean)
        print('artifitial portfolio variance:', arti_var)
        print('artifitial portfolio std:', arti_std)
        print('artifitial portfolio sr:', arti_sr)
        print()
    except:
        print('artifitial porfolio cannot be ran, please double check')
        print()
        pass
    
    eq_wght_port, eq_wght_mean, eq_wght_var, eq_wght_std, eq_wght_sr = equally_weighted_portfolio(portfolio)
    eq_wght_port_df = pd.DataFrame(eq_wght_port).transpose()
    print('equally weighted portfolio weights:', eq_wght_port_df, sep = '\n')
    print('equally weighted portfolio mean:', eq_wght_mean)
    print('equally weighted portfolio variance:', eq_wght_var)
    print('equally weighted portfolio std:', eq_wght_std)
    print('equally weighted portfolio sr:', eq_wght_sr)
    print()
    
    mv_port, mv_mean, mv_var, mv_std, mv_sr = minimum_variance_portfolio(portfolio)
    mv_port_df = pd.DataFrame(mv_port).transpose()
    print('minimum variance portfolio weights:', mv_port_df, sep = '\n')
    print('minimum variance portfolio mean:', mv_mean)
    print('minimum variance portfolio variance:', mv_var)
    print('minimum variance portfolio std:', mv_std)
    print('minimum variance portfolio sr:', mv_sr)
    print()
    
    mv_cali_s_port, mv_cali_s_mean, mv_cali_s_var, mv_cali_s_std, mv_cali_s_sr = minimum_variance_portfolio(portfolio, method = 'calibration')
    mv_cali_s_df = pd.DataFrame(mv_cali_s_port).transpose()
    print('minimum variance portfolio using calibration method w/ short selling weights:', mv_cali_s_df, sep = '\n')
    print('minimum variance portfolio using calibration method w/ short selling mean:', mv_cali_s_mean)
    print('minimum variance portfolio using calibration method w/ short selling variance:', mv_cali_s_var)
    print('minimum variance portfolio using calibration method w/ short selling std:', mv_cali_s_std)
    print('minimum variance portfolio using calibration method w/ short selling sr:', mv_cali_s_sr)
    print() 

    mv_cali_ns_port, mv_cali_ns_mean, mv_cali_ns_var, mv_cali_ns_std, mv_cali_ns_sr = minimum_variance_portfolio(portfolio, method = 'calibration', short_sell = False)
    mv_cali_ns_df = pd.DataFrame(mv_cali_ns_port).transpose()
    print('minimum variance portfolio using calibration method w/o short selling weights:', mv_cali_ns_df, sep = '\n')
    print('minimum variance portfolio using calibration method w/o short selling mean:', mv_cali_ns_mean)
    print('minimum variance portfolio using calibration method w/o short selling variance:', mv_cali_ns_var)
    print('minimum variance portfolio using calibration method w/o short selling std:', mv_cali_ns_std)
    print('minimum variance portfolio using calibration method w/o short selling sr:', mv_cali_ns_sr)
    print() 

    mean_var_port, mean_var_mean, mean_var_var, mean_var_std, mean_var_sr = mean_variance_portfolio(portfolio)
    mean_var_port_df = pd.DataFrame(mean_var_port).transpose()
    print('mean variance portfolio weights:', mean_var_port_df, sep = '\n')
    print('mean variance portfolio mean:', mean_var_mean)
    print('mean variance portfolio variance:', mean_var_var)
    print('mean variance portfolio std:', mean_var_std)
    print('mean variance portfolio sr:', mean_var_sr)
    print()
    
    frontier_port, frontier_mean, frontier_var, frontier_std, frontier_sr = frontier_portfolio(portfolio, mv_mean)
    frontier_port_df = pd.DataFrame(frontier_port).transpose()
    print('frontier portfolio with mv portfolio mean weights:', frontier_port_df, sep = '\n')
    print('frontier portfolio with mv portfolio mean mean:', frontier_mean)
    print('frontier portfolio with mv portfolio mean variance:', frontier_var)
    print('frontier portfolio with mv portfolio mean std:', frontier_std)
    print('frontier portfolio with mv portfolio mean sr:', frontier_sr)
    print()
    
    target_mean1 = 0.75
    frontier_port, frontier_mean, frontier_var, frontier_std, frontier_sr = frontier_portfolio(portfolio, 
                                                                                              target_mean = target_mean1, 
                                                                                              method = 'calibration',
                                                                                              short_sell = False)
    frontier_port_df = pd.DataFrame(frontier_port).transpose()
    print(f'frontier portfolio with target mean {target_mean1} w/o short selling weights:', frontier_port_df, sep = '\n')
    print(f'frontier portfolio with target mean {target_mean1} w/o short selling mean:', frontier_mean)
    print(f'frontier portfolio with target mean {target_mean1} w/o short selling variance:', frontier_var)
    print(f'frontier portfolio with target mean {target_mean1} w/o short selling std:', frontier_std)
    print(f'frontier portfolio with target mean {target_mean1} w/o short selling sr:', frontier_sr)
    print()
    
    frontier_port, frontier_mean, frontier_var, frontier_std, frontier_sr = frontier_portfolio(portfolio, 
                                                                                              target_mean = target_mean1, 
                                                                                              method = 'calibration',
                                                                                              short_sell = True)
    frontier_port_df = pd.DataFrame(frontier_port).transpose()
    print(f'frontier portfolio with target mean {target_mean1} w/ short selling weights:', frontier_port_df, sep = '\n')
    print(f'frontier portfolio with target mean {target_mean1} w/ short selling mean:', frontier_mean)
    print(f'frontier portfolio with target mean {target_mean1} w/ short selling variance:', frontier_var)
    print(f'frontier portfolio with target mean {target_mean1} w/ short selling std:', frontier_std)
    print(f'frontier portfolio with target mean {target_mean1} w/ short selling sr:', frontier_sr)
    print()
    
    msr_port, msr_mean, msr_var, msr_std, msr_sr = maximum_sharpe_ratio_portfolio(portfolio, short_sell = True)
    msr_port_df = pd.DataFrame(msr_port).transpose()
    print('maximum sharpe ratio portfolio w/ short sell weights:', msr_port_df, sep = '\n')
    print('maximum sharpe ratio portfolio w/ short sell mean:', msr_mean)
    print('maximum sharpe ratio portfolio w/ short sell variance:', msr_var)
    print('maximum sharpe ratio portfolio w/ short sell std:', msr_std)
    print('maximum sharpe ratio portfolio w/ short sell sr:', msr_sr)
    print()
    
    msr_port, msr_mean, msr_var, msr_std, msr_sr = maximum_sharpe_ratio_portfolio(portfolio, short_sell = False)
    msr_port_df = pd.DataFrame(msr_port).transpose()
    print('maximum sharpe ratio portfolio w/o short sell weights:', msr_port_df, sep = '\n')
    print('maximum sharpe ratio portfolio w/o short sell mean:', msr_mean)
    print('maximum sharpe ratio portfolio w/o short sell variance:', msr_var)
    print('maximum sharpe ratio portfolio w/o short sell std:', msr_std)
    print('maximum sharpe ratio portfolio w/o short sell sr:', msr_sr)
    print()

    min_mean = portfolio_mean(minimum_variance_portfolio(portfolio)[0])
    mean_grid = np.linspace(min_mean, max(eq_wght_mean,mv_mean,mean_var_mean,msr_mean), num = 1000, endpoint = True)
    std_grid_func = np.vectorize(lambda m: frontier_portfolio(
        portfolio=portfolio, target_mean=m, method=method, short_sell=short_sell, upper_bound=upper_bound, lower_bound=lower_bound)[3]
        )
    std_grid = std_grid_func(mean_grid)
    
    plt.plot(std_grid, mean_grid, label="Frontier Portfolio")
    
    points_x = [eq_wght_std, mv_std , mean_var_std, msr_std]
    points_y = [eq_wght_mean,mv_mean, mean_var_mean,msr_mean]
    labels = ["Equally Weighted", "Min Var", "Mean Var", "Max Sharpe"]
    colors = ['red', 'blue', 'green', 'yellow']
    plt.scatter(points_x, points_y, color=colors, s=30, zorder=5)
    
    for px, py, label, c in zip(points_x, points_y, labels, colors):
        plt.text(px, py, label, fontsize=10, ha="right", va="bottom")
    
    plt.xlabel("standard deviation")
    plt.ylabel("target_mean")
    plt.title("Frontier Portfolio")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    show_all_portfolio_stats(EQportfolio)
    # frontier_portfolio_plot(EQportfolio)

