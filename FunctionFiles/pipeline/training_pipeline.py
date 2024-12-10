import sys 
from FunctionFiles.pipeline.preprocessors import *
# from Algorithms.VPDDPG.VP_DDPG_V2 import VP_DDPG

class training:
    def __init__(self, model_name, train_env, val_env, trade_env, env_wrapper):
        self.model_name = model_name
        self.train_env = train_env
        self.val_env = val_env
        self.trade_env = trade_env
        self.env_wrapper = env_wrapper

    def retraining_approach(self, df, rebalance_window, validation_window, unique_trade_dates, training_instance, turbulence_threshold_level, training_function):
        last_state = []

        insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
        insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
        insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, turbulence_threshold_level)

        for i in range(rebalance_window + validation_window, len(unique_trade_dates), rebalance_window):
            print("============================================")
            ## initial state is empty
            if i - rebalance_window - validation_window == 0:
                # inital state
                initial = True
            else:
                # previous state
                initial = False
            
            if turbulence_threshold_level < 1:
                # Tuning trubulence index based on historical data
                # Turbulence lookback window is one quarter
                end_date_index = df.index[df["datadate"] == unique_trade_dates[i - rebalance_window - validation_window]].to_list()[-1]
                start_date_index = end_date_index - validation_window*30 + 1

                historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
                historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
                historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

                if historical_turbulence_mean > insample_turbulence_threshold:
                    turbulence_threshold = insample_turbulence_threshold
                else:
                    turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
                print("turbulence_threshold: ", turbulence_threshold)

            elif turbulence_threshold_level >= 1: 
                turbulence_threshold = 10000000000

            ############## Environment Setup starts ##############
            train = data_split(df, start=20090000, end=unique_trade_dates[i - rebalance_window - validation_window]) ## training data
            env_train = self.env_wrapper([lambda: self.train_env(train, state_components=df.columns.tolist()[8:])])
            ############## Environment Setup ends ##############

            print(f"====== {self.model_name} Training========")
            model = training_function(env_train, model_name="{}_{}".format(self.model_name, i), training_instance=training_instance)

            ############## Trading starts ##############
            print("======Trading from: ", unique_trade_dates[i - rebalance_window], "to ", unique_trade_dates[i])
            last_state = self.DRL_prediction(df=df, model=model, name=f"{self.model_name}_{training_instance}",
                                                    last_state=last_state, iter_num=i,
                                                    unique_trade_date=unique_trade_dates,
                                                    rebalance_window=rebalance_window,
                                                    turbulence_threshold=turbulence_threshold,
                                                    initial=initial,
                                                    alg = False)
            ############## Trading ends ##############
        
    def predict_from_pretrained(self, df, rebalance_window, validation_window, unique_trade_dates, training_instance, turbulence_threshold_level, algorithm, model_path):
        last_state = []

        insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
        insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
        insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, turbulence_threshold_level)

        for i in range(rebalance_window + validation_window, len(unique_trade_dates), rebalance_window):
            print("============================================")
            ## initial state is empty
            if i - rebalance_window - validation_window == 0:
                # inital state
                initial = True
            else:
                # previous state
                initial = False
            
            if turbulence_threshold_level < 1:
                # Tuning trubulence index based on historical data
                # Turbulence lookback window is one quarter
                end_date_index = df.index[df["datadate"] == unique_trade_dates[i - rebalance_window - validation_window]].to_list()[-1]
                start_date_index = end_date_index - validation_window*30 + 1

                historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
                historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
                historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

                if historical_turbulence_mean > insample_turbulence_threshold:
                    turbulence_threshold = insample_turbulence_threshold
                else:
                    turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
                print("turbulence_threshold: ", turbulence_threshold)

            elif turbulence_threshold_level >= 1: 
                turbulence_threshold = 10000000000

            
            ############## Trading starts ##############
            print("======Trading from: ", unique_trade_dates[i - rebalance_window], "to ", unique_trade_dates[i])
            print(f'{model_path}/{training_instance}/{self.model_name}_{i}')
            last_state = self.DRL_prediction(df=df, model=f'{model_path}/{training_instance}/{self.model_name}_{i}', name=f"{self.model_name}_{training_instance}",
                                                    last_state=last_state, iter_num=i,
                                                    unique_trade_date=unique_trade_dates,
                                                    rebalance_window=rebalance_window,
                                                    turbulence_threshold=turbulence_threshold,
                                                    initial=initial,
                                                    alg = algorithm)
            ############## Trading ends ##############
    
# ================
# Helper Functions 
# ================
    
    def DRL_prediction(self, df, model, name, last_state, iter_num, unique_trade_date, rebalance_window, turbulence_threshold, initial, alg):
        ### make a prediction based on trained model###
        ## trading env
        print(iter_num)
        print(unique_trade_date[iter_num - rebalance_window], unique_trade_date[iter_num])
        trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
        env_trade = self.env_wrapper([lambda: self.trade_env(trade_data,
                                                    turbulence_threshold=turbulence_threshold,
                                                    initial=initial,
                                                    previous_state=last_state,
                                                    model_name=name,
                                                    iteration=iter_num,
                                                    state_components=df.columns.tolist()[8:])])
        obs_trade = env_trade.reset()

        if alg != False:
            model = alg.load(model, env=env_trade)

        for i in range(len(trade_data.index.unique())):
            action, _states = model.predict(obs_trade)
            obs_trade, rewards, dones, info = env_trade.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                last_state = env_trade.env_method(method_name = 'render')[0]
                action_memory = env_trade.env_method(method_name = 'get_action_history')[0]

        df_last_state = pd.DataFrame({'last_state': last_state})
        df_last_state.to_csv('working_files/last_state_{}_{}.csv'.format(name, i), index=False)

        # print(action_memory)
        df_action_memory = pd.DataFrame(action_memory, columns=np.arange(33))
        # print('working_files/action_history_{}_{}.csv'.format(name, iter_num))
        df_action_memory.to_csv('working_files/action_history_{}_{}.csv'.format(name, iter_num), index=False)
        return last_state

    def DRL_validation(self, model, test_data, test_env, test_obs) -> None:
        ###validation process###
        for i in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)

    def get_validation_sharpe(self, iteration):
        ###Calculate Sharpe ratio based on validation results###
        df_total_value = pd.read_csv('working_files/account_value_validation_{}.csv'.format(iteration), index_col=0)
        df_total_value.columns = ['account_value_train']
        df_total_value['daily_return'] = df_total_value.pct_change(1)
        sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
                df_total_value['daily_return'].std()
        return sharpe
    