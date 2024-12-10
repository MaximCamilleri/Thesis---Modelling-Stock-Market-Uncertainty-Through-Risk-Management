import numpy as np
import pandas as pd
import pyfolio
from pyfolio import timeseries
from tensorboard import program
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import norm, kurtosis, skew 
import statistics
import os

class algorithm_return: 
    def __init__(self):
        self.annual_return = np.array([])
        self.cum_return = np.array([])
        self.annual_volatility = np.array([])
        self.sortino = np.array([])
        self.drawdown = np.array([])
        self.sharpe = np.array([])
        self.account_value = []

class backtesting:
    def __init__(self, model_name, output = 2, retraining = True, file_path = 'working_files/account_value_trade_{}_{}.csv'):
        self.model_name = model_name
        benchmark = self.benchmark()
        self.retraining = retraining
        self.file_path = file_path

        self.avg_agent = self.algorithmic_approach_avg()
        self.output(output, benchmark, self.avg_agent)

    def benchmark(self):
        dji = pd.read_csv("../data/^DJI.csv")
        test_dji=dji[(dji['Date']>='2016-01-01') & (dji['Date']<='2020-06-30')]
        test_dji = test_dji.reset_index(drop=True)
        test_dji['daily_return']=test_dji['Adj Close'].pct_change(1)
        self.test_dji = test_dji

        dow_strat = self._backtest_strat(test_dji)

        return dow_strat
    
    def algorithmic_approach_avg(self):
        avg_agent = algorithm_return()

        for n in self.model_name:
            df=pd.read_csv('../data/dow_30_2009_2020.csv')
            unique_trade_date = df[(df.datadate > 20151001)&(df.datadate <= 20200707)].datadate.unique()
            df_trade_date = pd.DataFrame({'datadate':unique_trade_date})

            if self.retraining:
                rebalance_window = 63
                validation_window = 63
                account_value = self._get_account_value_retraining(n, rebalance_window, validation_window, unique_trade_date, df_trade_date)
            else: 
                account_value = self._get_account_value(n)

            account_value = self._get_daily_return(account_value)
            account_value['Date'] = self.test_dji['Date']
            avg_agent.account_value.append(account_value)
            strat = self._backtest_strat(account_value)

            perf_stats_all = timeseries.perf_stats(
                returns=strat,
                positions=None,
                transactions=None,
                turnover_denom="AGB",
            )

            for index, row in perf_stats_all.items():
                if index == "Annual return":
                    avg_agent.annual_return = np.append(avg_agent.annual_return, row)
                if index == "Cumulative returns":
                    avg_agent.cum_return = np.append(avg_agent.cum_return, row)
                if index == "Sharpe ratio":
                    avg_agent.sharpe = np.append(avg_agent.sharpe, row)
                if index == "Annual volatility":
                    avg_agent.annual_volatility = np.append(avg_agent.annual_volatility, row)
                if index == "Sortino ratio":
                    avg_agent.sortino = np.append(avg_agent.sortino, row)
                if index == "Max drawdown":
                    avg_agent.drawdown = np.append(avg_agent.drawdown, row)
    
        return avg_agent
    
    def output(self, output, base, avg_agent):
        if output == 0: 
            pass 
        elif output == 1:
            with pyfolio.plotting.plotting_context(font_scale=1.1):
                pyfolio.create_full_tear_sheet(returns = avg_agent.annual_return, benchmark_rets=base, set_context=False)
        elif output == 2: 
            print(tabulate([
                ['Annual Return', avg_agent.annual_return.mean(), np.median(avg_agent.annual_return)], 
                ['Cumulative Return', avg_agent.cum_return.mean(), np.median(avg_agent.cum_return)],
                ['Annual Volatility', avg_agent.annual_volatility.mean(), np.median(avg_agent.annual_volatility)],
                ['Sharpe Ratio', avg_agent.sharpe.mean(), np.median(avg_agent.sharpe)],
                ['Sortino Ratio', avg_agent.sortino.mean(), np.median(avg_agent.sortino)],
                ['Max Drawdown', avg_agent.drawdown.mean(), np.median(avg_agent.drawdown)]
            ], headers=['Attribute', 'Mean', "Median"]))

            print(f"Sharpe STD: {statistics.stdev(avg_agent.sharpe)}")

    # ================
    # Helper Functions 
    # ================

    def _get_daily_return(self, df):
        df['daily_return']=df.account_value.pct_change(1)
        # print('Sharpe: ',(252**0.5)*df['daily_return'].mean()/ df['daily_return'].std())
        return df

    def _backtest_strat(self, df):
        strategy_ret= df.copy()
        strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'])
        strategy_ret.set_index('Date', drop = False, inplace = True)
        strategy_ret.index = strategy_ret.index.tz_localize('UTC')
        del strategy_ret['Date']
        ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
        return ts

    def _get_account_value_retraining(self, model_name, rebalance_window, validation_window, unique_trade_date, df_trade_date):
        df_account_value=pd.DataFrame()
        for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
            temp = pd.read_csv(self.file_path.format(i, model_name))
            df_account_value = pd.concat([df_account_value, temp])
        df_account_value = pd.DataFrame({'account_value':df_account_value['0']})
        df_account_value = df_account_value.reset_index(drop=True)
        df_account_value=df_account_value.join(df_trade_date[63:].reset_index(drop=True))
        return df_account_value
    
    def _get_account_value(self, model_name):
        df_account_value=pd.DataFrame()
        temp = pd.read_csv('working_files/account_value_trade_{}_{}.csv'.format(model_name, 1))

        df_account_value = df_account_value.append(temp,ignore_index=True)
        df_account_value = pd.DataFrame({'account_value':df_account_value['0']})
        return df_account_value

    def get_best_account_value(self):
        return self.best_account_value
    
    def get_worsed_account_value(self):
        return self.worsed_account_value

class TB_Logs():
    def __init__(self, log_path):
        self.launch_TB(log_path)
        self.launch_notebook()

    def launch_TB(self, log_path):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_path])
        url = tb.launch()
        self.port = int(url.split(":")[-1][:-1])
        print(f"Tensorflow listening on {url}")

    def launch_notebook(self):
        notebook.display(self.port, height=800)
    
class position_analysis: 
    def __init__(self, model_name, folder_name, iter): 
        self.turbulence_list = [] 
        self.turbulence_threshold_list = []
        self.actions_df = pd.DataFrame()
        self.dates_list = []

        self.data_path = f"{model_name}_{folder_name}"
        self.iter = iter
    
    def load_action_data(self): 
        for i in range(63 + 63, 1198, 63):
            data = pd.read_csv(f"working_files/action_history_{self.data_path}{self.iter}_{i}.csv")
            self.turbulence_list.append(data['31'].tolist())
            self.turbulence_threshold_list.append(data['30'].tolist())
            self.dates_list.append(data['32'].tolist())
            self.actions_df = pd.concat([self.actions_df, data.loc[:, ((data.columns != '31') & (data.columns != '30') & (data.columns != '32'))]])

        self.turbulence_list = self._flatten_comprehension(self.turbulence_list)
        self.turbulence_threshold_list = self._flatten_comprehension(self.turbulence_threshold_list)
        self.dates_list = self._flatten_comprehension(self.dates_list)

        for i in range(len(self.turbulence_list)):
            if self.turbulence_list[i] == 0 and i == 0: 
                self.turbulence_list[i] = self.turbulence_list[i+1]
            elif self.turbulence_list[i] == 0: 
                self.turbulence_list[i] = (self.turbulence_list[i-1] + self.turbulence_list[i+1])/2

        self.actions_df = self.actions_df.reset_index(drop=True)
        self.actions_df = self.actions_df.round(3)
        for col in self.actions_df.columns:
            self.actions_df[col] = self.actions_df[col].cumsum()

    # Getters
    def get_volume_moved(self): 
        return self.actions_df.abs().sum()
    
    # Graphing
    def volume_graph(self, symbol):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.grid()
        ax1 = ax.twinx()

        p1 = ax.plot(self.actions_df[symbol], label='Assets Held', zorder=10, color="blue")
        ax.set(ylabel = "Volume")
        ax.tick_params(axis='y', colors="blue")
        ax.yaxis.label.set_color("blue")

        p2 = ax1.plot(self.turbulence_list, label='Market Turbulence', zorder=10, color="green")
        p3 = ax1.plot(self.turbulence_threshold_list, label='Turbulence Threshold', zorder=10, color="red")
        ax1.set(ylabel = "Turbulence")

        lns = p1+p2+p3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
    
    def volume_graph_fill_turb(self, symbols): 
        border_list_trade = []
        border_list_no_trade = []
        start = 0 
        side_flag = True # True means turbulence is less then thresh

        for i in range(len(self.turbulence_list)): 
            if (self.turbulence_list[i] < self.turbulence_threshold_list[i]) and (side_flag == True):
                pass
            elif (self.turbulence_list[i] < self.turbulence_threshold_list[i]) and (side_flag == False):
                border_list_no_trade.append((start, i))
                start = i
                side_flag = True
            elif (self.turbulence_list[i] > self.turbulence_threshold_list[i]) and (side_flag == True):
                border_list_trade.append((start, i))
                start = i
                side_flag = False
            elif (self.turbulence_list[i] > self.turbulence_threshold_list[i]) and (side_flag == False):
                pass
        if side_flag: border_list_trade.append((start, len(self.turbulence_list)))
        else: border_list_no_trade.append((start, len(self.turbulence_list)))

        preprocessed_path = "../Data/done_data.csv"
        if os.path.exists(preprocessed_path):
            data = pd.read_csv(preprocessed_path, index_col=0)


        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        ax = self._flatten_comprehension(ax)
        fig.suptitle('Position Movements for Top 4 Traded Symbols By Volume', fontsize=16)

        for s in range(len(symbols)): 
            tic = data.tic[:30].tolist()[int(symbols[s])]
            prices = data[(data.datadate.isin(self.dates_list))&(data.tic == tic)].adjcp.tolist()

            ax[s].grid()
            ax1 = ax[s].twinx()

            ax[s].plot(self.actions_df[symbols[s]], label='Assets Held', zorder=10, color="blue")
            ax[s].set(ylabel = "Volume", xlabel = "Days", title = f"Volume held for {tic}")

            for i in border_list_trade:
                ax[s].fill_between([i[0], i[1]], 0, max(self.actions_df[symbols[s]]), color='green', alpha=0.2, label="Turb Under Thresh")

            for i in border_list_no_trade:
                ax[s].fill_between([i[0], i[1]], 0, max(self.actions_df[symbols[s]]), color='red', alpha=0.2, label="Turb Over Thresh")

            ax1.plot(prices, color="purple", label="Asset Price")


            handles, labels = ax[s].get_legend_handles_labels()   
            handles2, labels2 = ax1.get_legend_handles_labels()   
            by_label = dict(zip(labels2+labels, handles2+handles)) 
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=4)

    # Helper Functions
    def _flatten_comprehension(self, matrix):
        return [item for row in matrix for item in row]

def probabilistic_sharpe_ratio(sharpe_ratio, bench_sharpe_ratio,daily_returns):
    def standard_deviation_sharpe_ratio(sharpe_ratio, num_obs, skewness, kurt):
        return np.sqrt(
            (1 - skewness*sharpe_ratio + 
            (kurt-1)/4*sharpe_ratio**2
            ) / (num_obs-1)
        )
    
    num_obs = len(daily_returns)
    skewness = skew(daily_returns, axis=0, bias=True)
    kurt = kurtosis(daily_returns, axis=0, bias=True)

    sr_diff = sharpe_ratio - bench_sharpe_ratio
    sr_vol = standard_deviation_sharpe_ratio(sharpe_ratio, num_obs, skewness, kurt)
    return norm.cdf(sr_diff / sr_vol)

def return_graph(account_value, model_name):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.set_title('Account Value')
    
    for i in range(len(account_value)):
        # print(account_value[i].columns)
        ax.plot(account_value[i]['account_value'], label = f'{model_name[i]} Account Value')
    
    dji = pd.read_csv("../data/^DJI.csv")
    test_dji=dji[(dji['Date']>='2016-01-01') & (dji['Date']<='2020-06-30')]
    test_dji = test_dji.reset_index(drop=True)
    ax.plot(test_dji.Open*(1000000/test_dji.Open.iloc[0]), label = 'DOW', color = 'black')
    ax.legend()
    fig.show()

