from metaflow import FlowSpec, step, resources, IncludeFile
import pandas as pd
import numpy as np
from comet_ml import Experiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle

class OptionsTradingFlow(FlowSpec):

    @step
    def start(self):
        print("Let's start!")
        self.next(self.data_processing)

    @step
    def data_processing(self):
        """
        Load and preprocess data
        """
        print("Loading Data!")
        data = pd.read_csv('spy_2020_2022.csv',low_memory=False)
        print("Processing Data!")
        data = data[data[' [DTE]']>=30] # ignore options near the maturity
        # convert column types
        def convert(col):
            try:
                return float(col)
            except:
                return np.nan
        for i in [' [C_LAST]',' [C_IV]',' [C_VOLUME]',' [P_LAST]',' [P_IV]',' [P_VOLUME]']:
            data[i] = data[i].apply(convert)
        for j in [' [QUOTE_DATE]',' [EXPIRE_DATE]']:
            data[j] = pd.to_datetime(data[j])
        #data[' [DTE]'] = pd.to_timedelta(data[' [DTE]'],unit='d')
        data['[OVERALL_VOLUME]'] = np.nansum(data[[' [C_VOLUME]', ' [P_VOLUME]']], axis=1)

        data_train = data.copy()
        # map the underlying price
        underlying = data.groupby([' [QUOTE_DATE]'])[' [UNDERLYING_LAST]'].mean()
        data_train['UnderlyingExp'] = data_train[' [EXPIRE_DATE]'].map(underlying)
        spy_ETF = pd.read_csv('SPY_ETF.csv', index_col='Date', parse_dates=['Date'])
        spy_ETF = spy_ETF['Close']
        p_nan = np.isnan(data_train['UnderlyingExp'])
        data_train['UnderlyingExp'][p_nan] = data_train[' [EXPIRE_DATE]'][p_nan].map(spy_ETF)
        # calculate real p&l
        data_train['y_longcall'] = ((data_train['UnderlyingExp'] - data_train[' [STRIKE]']).clip(0) - data_train[' [C_LAST]'] > 0).astype(int)
        data_train['y_shortcall'] = ((data_train['UnderlyingExp'] - data_train[' [STRIKE]']).clip(0)- data_train[' [C_LAST]'] < 0).astype(int)
        data_train['y_longput'] = ((data_train[' [STRIKE]'] - data_train['UnderlyingExp']).clip(0)- data_train[' [P_LAST]'] > 0).astype(int)
        data_train['y_shortput'] = ((data_train[' [STRIKE]'] - data_train['UnderlyingExp']).clip(0)- data_train[' [P_LAST]'] < 0).astype(int)
        
        # binary/dummy variable of IV
        def dummy_IV(series):
            IV_median = np.nanmean(series)
            return (series>IV_median).astype(int)
        data_train['C_IV_binary'] = dummy_IV(data_train[' [C_IV]'])
        data_train['P_IV_binary'] = dummy_IV(data_train[' [P_IV]'])

        # separate into ITM/ATM/OTM types
        def xtm_call(x):
            if x[' [STRIKE]'] < 0.94 * x[' [UNDERLYING_LAST]']:
                return 'ITM'
            elif x[' [STRIKE]'] > 1.06 * x[' [UNDERLYING_LAST]']:
                return 'OTM'
            else:
                return 'ATM'
        def xtm_put(x):
            if x[' [STRIKE]'] < 0.94 * x[' [UNDERLYING_LAST]']:
                return 'OTM'
            elif x[' [STRIKE]'] > 1.06 * x[' [UNDERLYING_LAST]']:
                return 'ITM'
            else:
                return 'ATM'
        data_train['type_call'] = data_train.apply(xtm_call,axis=1)
        data_train['type_put'] = data_train.apply(xtm_put,axis=1)

        # add macroeconomic data
        macro = pd.read_excel('Macro.xlsx',sheet_name='Sheet1', index_col='Date', parse_dates=['Date'])
        for i in macro.columns:
            data_train[i] = data_train[' [QUOTE_DATE]'].map(macro[i])

        data_call = data_train[[' [DTE]',' [QUOTE_DATE]',' [STRIKE]',' [UNDERLYING_LAST]',
             ' [C_LAST]',' [C_IV]','C_IV_binary',' [C_VOLUME]','[OVERALL_VOLUME]',
             'type_call','UnderlyingExp', 'y_longcall', 'y_shortcall']
             +list(macro.columns)].copy()
        data_put = data_train[[' [DTE]',' [QUOTE_DATE]',' [STRIKE]',' [UNDERLYING_LAST]',
             ' [P_LAST]',' [P_IV]','P_IV_binary',' [P_VOLUME]','[OVERALL_VOLUME]',
             'type_put','UnderlyingExp','y_longput', 'y_shortput']
             +list(macro.columns)].copy()

        data_call = data_call[(data_call[' [C_VOLUME]'].notna()) &(data_call[' [C_VOLUME]'] != 0)]
        data_put = data_put[(data_put[' [P_VOLUME]'].notna()) &(data_put[' [P_VOLUME]'] != 0)]
        self.data_call = data_call
        self.data_put = data_put

        self.next(self.call, self.put)
    
    @step
    def call(self):
        "Processings for call"
        data_call = self.data_call
        data_call_pre = data_call.groupby([' [QUOTE_DATE]', ' [DTE]','type_call']).agg({'y_longcall': 'mean', 'y_shortcall': 'mean', ' [C_VOLUME]': 'mean'}).reset_index()
        data_call_pre[['C_ITM_VOLUME','C_ATM_VOLUME','C_OTM_VOLUME']] = 0
        data_call_pre.loc[data_call_pre['type_call'] == 'ITM', 'C_ITM_VOLUME'] = data_call_pre[' [C_VOLUME]']
        data_call_pre.loc[data_call_pre['type_call'] == 'ATM', 'C_ATM_VOLUME'] = data_call_pre[' [C_VOLUME]']
        data_call_pre.loc[data_call_pre['type_call'] == 'OTM', 'C_OTM_VOLUME'] = data_call_pre[' [C_VOLUME]']
        call_volumes = data_call_pre.groupby([' [QUOTE_DATE]', ' [DTE]']).agg({'C_ITM_VOLUME': 'max','C_ATM_VOLUME': 'max','C_OTM_VOLUME': 'max'}).reset_index()
        
        #Determine which kind of option has the highest probabilty of making profit
        P_profit_max_long = data_call_pre.groupby([' [QUOTE_DATE]', ' [DTE]'])['y_longcall'].idxmax()
        longcall_y = data_call_pre.loc[P_profit_max_long, [' [QUOTE_DATE]', ' [DTE]', 'type_call']].copy()
        longcall_y=longcall_y.reset_index()
        longcall_y['type_call'] = longcall_y['type_call'].replace({'OTM': 1, 'ATM': 0, 'ITM': -1})

        # for short
        P_profit_max_short = data_call_pre.groupby([' [QUOTE_DATE]', ' [DTE]'])['y_shortcall'].idxmax()
        shortcall_y = data_call_pre.loc[P_profit_max_short, [' [QUOTE_DATE]', ' [DTE]', 'type_call']].copy()
        shortcall_y=shortcall_y.reset_index()
        shortcall_y['type_call'] = shortcall_y['type_call'].replace({'OTM': 1, 'ATM': 0, 'ITM': -1})

        #Merge atasets
        data_longcall_pre = data_call.groupby([' [QUOTE_DATE]', ' [DTE]']).agg({'y_longcall': 'mean','C_IV_binary': 'mean',' [C_VOLUME]': 'mean','[OVERALL_VOLUME]': 'mean',' [UNDERLYING_LAST]': 'first','Unemployment_rate': 'first', 'GDP': 'first', 'M1': 'first', 'M2': 'first', 'Fed_target_rate': 'first', 'CCPI': 'first', '10y_tb_yield': 'first', 'Umich_inflation_expectation': 'first'}).reset_index()
        merged_longcall = pd.merge(pd.merge(data_longcall_pre, longcall_y[['type_call']],left_index=True, right_index=True), call_volumes[['C_ITM_VOLUME','C_ATM_VOLUME','C_OTM_VOLUME']],left_index=True, right_index=True)
        data_shortcall_pre = data_call.groupby([' [QUOTE_DATE]', ' [DTE]']).agg({'y_shortcall': 'mean','C_IV_binary': 'mean',' [C_VOLUME]': 'mean','[OVERALL_VOLUME]': 'mean',' [UNDERLYING_LAST]': 'first','Unemployment_rate': 'first', 'GDP': 'first', 'M1': 'first', 'M2': 'first', 'Fed_target_rate': 'first', 'CCPI': 'first', '10y_tb_yield': 'first', 'Umich_inflation_expectation': 'first'}).reset_index()
        merged_shortcall = pd.merge(pd.merge(data_shortcall_pre, shortcall_y[['type_call']],left_index=True, right_index=True), call_volumes[['C_ITM_VOLUME','C_ATM_VOLUME','C_OTM_VOLUME']],left_index=True, right_index=True)
        
        #ffill NANs
        merged_longcall = merged_longcall.ffill()
        merged_shortcall = merged_shortcall.ffill()
        self.merged_longcall = merged_longcall
        self.merged_shortcall = merged_shortcall
        self.data_longcall_pre = data_longcall_pre
        self.data_shortcall_pre = data_shortcall_pre

        self.next(self.longcall, self.shortcall)

    @step
    def put(self):
        "Processings for put"
        
        data_put = self.data_put
        data_put_pre = data_put.groupby([' [QUOTE_DATE]', ' [DTE]','type_put']).agg({'y_longput': 'mean', 'y_shortput': 'mean', ' [P_VOLUME]': 'mean'}).reset_index()
        data_put_pre[['P_ITM_VOLUME','P_ATM_VOLUME','P_OTM_VOLUME']] = 0
        data_put_pre.loc[data_put_pre['type_put'] == 'ITM', 'P_ITM_VOLUME'] = data_put_pre[' [P_VOLUME]']
        data_put_pre.loc[data_put_pre['type_put'] == 'ATM', 'P_ATM_VOLUME'] = data_put_pre[' [P_VOLUME]']
        data_put_pre.loc[data_put_pre['type_put'] == 'OTM', 'P_OTM_VOLUME'] = data_put_pre[' [P_VOLUME]']
        put_volumes = data_put_pre.groupby([' [QUOTE_DATE]', ' [DTE]']).agg({'P_ITM_VOLUME': 'max','P_ATM_VOLUME': 'max','P_OTM_VOLUME': 'max'}).reset_index()
        
        #Determine which kind of option has the highest probabilty of making profit
        P_profit_max_long = data_put_pre.groupby([' [QUOTE_DATE]', ' [DTE]'])['y_longput'].idxmax()
        longput_y = data_put_pre.loc[P_profit_max_long, [' [QUOTE_DATE]', ' [DTE]', 'type_put']].copy()
        longput_y=longput_y.reset_index()
        longput_y['type_put'] = longput_y['type_put'].replace({'OTM': 1, 'ATM': 0, 'ITM': -1})

        # for short
        P_profit_max_short = data_put_pre.groupby([' [QUOTE_DATE]', ' [DTE]'])['y_shortput'].idxmax()
        shortput_y = data_put_pre.loc[P_profit_max_short, [' [QUOTE_DATE]', ' [DTE]', 'type_put']].copy()
        shortput_y=shortput_y.reset_index()
        shortput_y['type_put'] = shortput_y['type_put'].replace({'OTM': 1, 'ATM': 0, 'ITM': -1})

        #Merge atasets
        data_longput_pre = data_put.groupby([' [QUOTE_DATE]', ' [DTE]']).agg({'y_longput': 'mean','P_IV_binary': 'mean',' [P_VOLUME]': 'mean','[OVERALL_VOLUME]': 'mean',' [UNDERLYING_LAST]': 'first','Unemployment_rate': 'first', 'GDP': 'first', 'M1': 'first', 'M2': 'first', 'Fed_target_rate': 'first', 'CCPI': 'first', '10y_tb_yield': 'first', 'Umich_inflation_expectation': 'first'}).reset_index()
        merged_longput = pd.merge(pd.merge(data_longput_pre, longput_y[['type_put']],left_index=True, right_index=True), put_volumes[['P_ITM_VOLUME','P_ATM_VOLUME','P_OTM_VOLUME']],left_index=True, right_index=True)
        data_shortput_pre = data_put.groupby([' [QUOTE_DATE]', ' [DTE]']).agg({'y_shortput': 'mean','P_IV_binary': 'mean',' [P_VOLUME]': 'mean','[OVERALL_VOLUME]': 'mean',' [UNDERLYING_LAST]': 'first','Unemployment_rate': 'first', 'GDP': 'first', 'M1': 'first', 'M2': 'first', 'Fed_target_rate': 'first', 'CCPI': 'first', '10y_tb_yield': 'first', 'Umich_inflation_expectation': 'first'}).reset_index()
        merged_shortput = pd.merge(pd.merge(data_shortput_pre, shortput_y[['type_put']],left_index=True, right_index=True), put_volumes[['P_ITM_VOLUME','P_ATM_VOLUME','P_OTM_VOLUME']],left_index=True, right_index=True)
        
        #ffill NANs
        merged_longput = merged_longput.ffill()
        merged_shortput = merged_shortput.ffill()
        self.merged_longput = merged_longput
        self.merged_shortput = merged_shortput
        self.data_longput_pre = data_longput_pre
        self.data_shortput_pre = data_shortput_pre
        self.next(self.longput, self.shortput)

    @step
    def longcall(self):
        "Long call model"
        #Split train and test dataset

        merged_longcall = self.merged_longcall
        data_longcall_pre = self.data_longcall_pre
        data_call = self.data_call

        split_date = pd.to_datetime('2021-12-31')
        call_train = merged_longcall[merged_longcall[' [QUOTE_DATE]'] <= split_date].copy()
        call_test = merged_longcall[merged_longcall[' [QUOTE_DATE]'] > split_date].copy()

        #Convert average IV_binary to binary
        mean_IV_train = call_train['C_IV_binary'].mean()
        call_train['C_IV_binary'] = (call_train['C_IV_binary'] > mean_IV_train).astype(int)
        call_test['C_IV_binary'] = (call_test['C_IV_binary'] > mean_IV_train).astype(int)

        #Calculate the average success rate of testset
        data_longcall_pre_test = data_longcall_pre[data_longcall_pre[' [QUOTE_DATE]']>split_date]
        P_or_L_mean = data_longcall_pre_test['y_longcall'].mean()
        print('Probability of profit when options are chosen randomly on testset:',P_or_L_mean)

        #Divide X and y
        rf_model = RandomForestClassifier(random_state=666)
        X_call_train=call_train.drop(columns=['type_call',' [QUOTE_DATE]','y_longcall'])
        y_call_train = call_train['type_call']
        X_call_test=call_test.drop(columns=['type_call',' [QUOTE_DATE]','y_longcall'])
        y_call_test = call_test['type_call']

        #GridSearch
        param_grid = {
            'n_estimators': [10,30,50,100,200],
            'max_depth': [None,5,10,20,40]
        }
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_call_train, y_call_train)
        print("Best Hyperparameters: ", grid_search.best_params_)
        best_rf_model = grid_search.best_estimator_
        #Make prediction
        y_call_train_pred = best_rf_model.predict(X_call_train)
        y_call_test_pred = best_rf_model.predict(X_call_test)

        #Accuracy on train and test dataset 
        train_accuracy = accuracy_score(y_call_train,y_call_train_pred)
        test_accuracy = accuracy_score(y_call_test,y_call_test_pred)
        print("Train accuracy:", train_accuracy)
        print("Test accuracy:", test_accuracy)

        #Calculate the probability of profit when using the model on the testset
        df_choice=data_longcall_pre_test[[' [QUOTE_DATE]',' [DTE]']]
        df_choice = df_choice.assign(NewColumn=y_call_test_pred)
        data_call_groupP = data_call.groupby([' [QUOTE_DATE]', ' [DTE]','type_call']).agg({'y_longcall': 'mean'}).reset_index()
        data_call_groupP=data_call_groupP[data_call_groupP[' [QUOTE_DATE]']>split_date]
        merged_choice = pd.merge(data_call_groupP, df_choice, on=[' [QUOTE_DATE]', ' [DTE]'], how='left')
        filtered_df = merged_choice[((merged_choice['type_call'] == 'ATM') & (merged_choice['NewColumn'] == 0)) |
                        ((merged_choice['type_call'] == 'OTM') & (merged_choice['NewColumn'] == 1)) |
                        ((merged_choice['type_call'] == 'ITM') & (merged_choice['NewColumn'] == -1))]
        print('Probability of profit when using the model on the testset:',filtered_df['y_longcall'].mean())

        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.p_random = P_or_L_mean
        self.p_recommend = filtered_df['y_longcall'].mean()

        # save datasets
        call_train.to_csv("call_train.csv")
        call_test.to_csv("call_test.csv")
        pickle.dump(best_rf_model,open('model_longcall.pkl','wb'))
        
        self.next(self.joincall)

    @step
    def shortcall(self):
        merged_shortcall = self.merged_shortcall
        data_shortcall_pre = self.data_shortcall_pre
        data_call = self.data_call

        split_date = pd.to_datetime('2021-12-31')
        call_train = merged_shortcall[merged_shortcall[' [QUOTE_DATE]'] <= split_date].copy()
        call_test = merged_shortcall[merged_shortcall[' [QUOTE_DATE]'] > split_date].copy()

        #Convert average IV_binary to binary
        mean_IV_train = call_train['C_IV_binary'].mean()
        call_train['C_IV_binary'] = (call_train['C_IV_binary'] > mean_IV_train).astype(int)
        call_test['C_IV_binary'] = (call_test['C_IV_binary'] > mean_IV_train).astype(int)

        #Calculate the average success rate of testset
        data_shortcall_pre_test = data_shortcall_pre[data_shortcall_pre[' [QUOTE_DATE]']>split_date]
        P_or_L_mean = data_shortcall_pre_test['y_shortcall'].mean()
        print('Probability of profit when options are chosen randomly on testset:',P_or_L_mean)

        #Divide X and y
        rf_model = RandomForestClassifier(random_state=666)
        X_call_train=call_train.drop(columns=['type_call',' [QUOTE_DATE]','y_shortcall'])
        y_call_train = call_train['type_call']
        X_call_test=call_test.drop(columns=['type_call',' [QUOTE_DATE]','y_shortcall'])
        y_call_test = call_test['type_call']

        #GridSearch
        param_grid = {
            'n_estimators': [10,30,50,100,200],
            'max_depth': [None,5,10,20,40]
        }
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_call_train, y_call_train)
        print("Best Hyperparameters: ", grid_search.best_params_)
        best_rf_model = grid_search.best_estimator_
        #Make prediction
        y_call_train_pred = best_rf_model.predict(X_call_train)
        y_call_test_pred = best_rf_model.predict(X_call_test)

        #Accuracy on train and test dataset 
        train_accuracy = accuracy_score(y_call_train,y_call_train_pred)
        test_accuracy = accuracy_score(y_call_test,y_call_test_pred)
        print("Train accuracy:", train_accuracy)
        print("Test accuracy:", test_accuracy)

        #Calculate the probability of profit when using the model on the testset
        df_choice=data_shortcall_pre_test[[' [QUOTE_DATE]',' [DTE]']]
        df_choice = df_choice.assign(NewColumn=y_call_test_pred)
        data_call_groupP = data_call.groupby([' [QUOTE_DATE]', ' [DTE]','type_call']).agg({'y_shortcall': 'mean'}).reset_index()
        data_call_groupP=data_call_groupP[data_call_groupP[' [QUOTE_DATE]']>split_date]
        merged_choice = pd.merge(data_call_groupP, df_choice, on=[' [QUOTE_DATE]', ' [DTE]'], how='left')
        filtered_df = merged_choice[((merged_choice['type_call'] == 'ATM') & (merged_choice['NewColumn'] == 0)) |
                        ((merged_choice['type_call'] == 'OTM') & (merged_choice['NewColumn'] == 1)) |
                        ((merged_choice['type_call'] == 'ITM') & (merged_choice['NewColumn'] == -1))]
        print('Probability of profit when using the model on the testset:',filtered_df['y_shortcall'].mean())

        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.p_random = P_or_L_mean
        self.p_recommend = filtered_df['y_shortcall'].mean()
        pickle.dump(best_rf_model,open('model_shortcall.pkl','wb'))

        self.next(self.joincall)

    @step
    def longput(self):
        
        merged_longput = self.merged_longput
        data_longput_pre = self.data_longput_pre
        data_shortput_pre = self.data_shortput_pre
        data_put = self.data_put

        split_date = pd.to_datetime('2021-12-31')
        put_train = merged_longput[merged_longput[' [QUOTE_DATE]'] <= split_date].copy()
        put_test = merged_longput[merged_longput[' [QUOTE_DATE]'] > split_date].copy()

        #Convert average IV_binary to binary
        mean_IV_train = put_train['P_IV_binary'].mean()
        put_train['P_IV_binary'] = (put_train['P_IV_binary'] > mean_IV_train).astype(int)
        put_test['P_IV_binary'] = (put_test['P_IV_binary'] > mean_IV_train).astype(int)

        #Calculate the average success rate of testset
        data_longput_pre_test = data_longput_pre[data_longput_pre[' [QUOTE_DATE]']>split_date]
        P_or_L_mean = data_longput_pre_test['y_longput'].mean()
        print('Probability of profit when options are chosen randomly on testset:',P_or_L_mean)

        #Divide X and y
        rf_model = RandomForestClassifier(random_state=666)
        X_put_train=put_train.drop(columns=['type_put',' [QUOTE_DATE]','y_longput'])
        y_put_train = put_train['type_put']
        X_put_test=put_test.drop(columns=['type_put',' [QUOTE_DATE]','y_longput'])
        y_put_test = put_test['type_put']

        #GridSearch
        param_grid = {
            'n_estimators': [10,30,50,100,200],
            'max_depth': [None,5,10,20,40]
        }
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_put_train, y_put_train)
        print("Best Hyperparameters: ", grid_search.best_params_)
        best_rf_model = grid_search.best_estimator_
        #Make prediction
        y_put_train_pred = best_rf_model.predict(X_put_train)
        y_put_test_pred = best_rf_model.predict(X_put_test)

        #Accuracy on train and test dataset 
        train_accuracy = accuracy_score(y_put_train,y_put_train_pred)
        test_accuracy = accuracy_score(y_put_test,y_put_test_pred)
        print("Train accuracy:", train_accuracy)
        print("Test accuracy:", test_accuracy)

        #Calculate the probability of profit when using the model on the testset
        df_choice=data_longput_pre_test[[' [QUOTE_DATE]',' [DTE]']]
        df_choice = df_choice.assign(NewColumn=y_put_test_pred)
        data_put_groupP = data_put.groupby([' [QUOTE_DATE]', ' [DTE]','type_put']).agg({'y_longput': 'mean'}).reset_index()
        data_put_groupP=data_put_groupP[data_put_groupP[' [QUOTE_DATE]']>split_date]
        merged_choice = pd.merge(data_put_groupP, df_choice, on=[' [QUOTE_DATE]', ' [DTE]'], how='left')
        filtered_df = merged_choice[((merged_choice['type_put'] == 'ATM') & (merged_choice['NewColumn'] == 0)) |
                        ((merged_choice['type_put'] == 'OTM') & (merged_choice['NewColumn'] == 1)) |
                        ((merged_choice['type_put'] == 'ITM') & (merged_choice['NewColumn'] == -1))]
        print('Probability of profit when using the model on the testset:',filtered_df['y_longput'].mean())

        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.p_random = P_or_L_mean
        self.p_recommend = filtered_df['y_longput'].mean()

        # save datasets
        put_train.to_csv("put_train.csv")
        put_test.to_csv("put_test.csv")

        pickle.dump(best_rf_model,open('model_longput.pkl','wb'))

        self.next(self.joinput)

    @step
    def shortput(self):
        
        merged_shortput = self.merged_shortput
        data_shortput_pre = self.data_shortput_pre
        data_put = self.data_put

        split_date = pd.to_datetime('2021-12-31')
        put_train = merged_shortput[merged_shortput[' [QUOTE_DATE]'] <= split_date].copy()
        put_test = merged_shortput[merged_shortput[' [QUOTE_DATE]'] > split_date].copy()

        #Convert average IV_binary to binary
        mean_IV_train = put_train['P_IV_binary'].mean()
        put_train['P_IV_binary'] = (put_train['P_IV_binary'] > mean_IV_train).astype(int)
        put_test['P_IV_binary'] = (put_test['P_IV_binary'] > mean_IV_train).astype(int)

        #Calculate the average success rate of testset
        data_shortput_pre_test = data_shortput_pre[data_shortput_pre[' [QUOTE_DATE]']>split_date]
        P_or_L_mean = data_shortput_pre_test['y_shortput'].mean()
        print('Probability of profit when options are chosen randomly on testset:',P_or_L_mean)

        #Divide X and y
        rf_model = RandomForestClassifier(random_state=666)
        X_put_train=put_train.drop(columns=['type_put',' [QUOTE_DATE]','y_shortput'])
        y_put_train = put_train['type_put']
        X_put_test=put_test.drop(columns=['type_put',' [QUOTE_DATE]','y_shortput'])
        y_put_test = put_test['type_put']

        #GridSearch
        param_grid = {
            'n_estimators': [10,30,50,100,200],
            'max_depth': [None,5,10,20,40]
        }
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_put_train, y_put_train)
        print("Best Hyperparameters: ", grid_search.best_params_)
        best_rf_model = grid_search.best_estimator_
        #Make prediction
        y_put_train_pred = best_rf_model.predict(X_put_train)
        y_put_test_pred = best_rf_model.predict(X_put_test)

        #Accuracy on train and test dataset 
        train_accuracy = accuracy_score(y_put_train,y_put_train_pred)
        test_accuracy = accuracy_score(y_put_test,y_put_test_pred)
        print("Train accuracy:", train_accuracy)
        print("Test accuracy:", test_accuracy)

        #Calculate the probability of profit when using the model on the testset
        df_choice=data_shortput_pre_test[[' [QUOTE_DATE]',' [DTE]']]
        df_choice = df_choice.assign(NewColumn=y_put_test_pred)
        data_put_groupP = data_put.groupby([' [QUOTE_DATE]', ' [DTE]','type_put']).agg({'y_shortput': 'mean'}).reset_index()
        data_put_groupP=data_put_groupP[data_put_groupP[' [QUOTE_DATE]']>split_date]
        merged_choice = pd.merge(data_put_groupP, df_choice, on=[' [QUOTE_DATE]', ' [DTE]'], how='left')
        filtered_df = merged_choice[((merged_choice['type_put'] == 'ATM') & (merged_choice['NewColumn'] == 0)) |
                        ((merged_choice['type_put'] == 'OTM') & (merged_choice['NewColumn'] == 1)) |
                        ((merged_choice['type_put'] == 'ITM') & (merged_choice['NewColumn'] == -1))]
        print('Probability of profit when using the model on the testset:',filtered_df['y_shortput'].mean())

        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.p_random = P_or_L_mean
        self.p_recommend = filtered_df['y_shortput'].mean()
        pickle.dump(best_rf_model,open('model_short.pkl','wb'))

        self.next(self.joinput)

    @step
    def joincall(self, inputs):
        self.longcall_train_accuracy = inputs.longcall.train_accuracy
        self.longcall_test_accuracy = inputs.longcall.test_accuracy
        self.longcall_p_random = inputs.longcall.p_random
        self.longcall_p_recommend = inputs.longcall.p_recommend

        self.shortcall_train_accuracy = inputs.shortcall.train_accuracy
        self.shortcall_test_accuracy = inputs.shortcall.test_accuracy
        self.shortcall_p_random = inputs.shortcall.p_random
        self.shortcall_p_recommend = inputs.shortcall.p_recommend

        self.next(self.join)

    @step
    def joinput(self, inputs):
        self.longput_train_accuracy = inputs.longput.train_accuracy
        self.longput_test_accuracy = inputs.longput.test_accuracy
        self.longput_p_random = inputs.longput.p_random
        self.longput_p_recommend = inputs.longput.p_recommend

        self.shortput_train_accuracy = inputs.shortput.train_accuracy
        self.shortput_test_accuracy = inputs.shortput.test_accuracy
        self.shortput_p_random = inputs.shortput.p_random
        self.shortput_p_recommend = inputs.shortput.p_recommend

        self.next(self.join)

    @step
    def join(self, inputs):
        self.metrics_model = {
            "train_accuracy_longcall": inputs.joincall.longcall_train_accuracy,
            "test_accuracy_longcall": inputs.joincall.longcall_test_accuracy,
            "random_longcall": inputs.joincall.longcall_p_random,
            "probability_of_profit_longcall": inputs.joincall.longcall_p_recommend,

            "train_accuracy_shortcall": inputs.joincall.shortcall_train_accuracy,
            "test_accuracy_shortcall": inputs.joincall.shortcall_test_accuracy,
            "random_shortcall": inputs.joincall.shortcall_p_random,
            "probability_of_profit_shortcall": inputs.joincall.shortcall_p_recommend,

            "train_accuracy_longput": inputs.joinput.longput_train_accuracy,
            "test_accuracy_longput": inputs.joinput.longput_test_accuracy,
            "random_longput": inputs.joinput.longput_p_random,
            "probability_of_profit_longput": inputs.joinput.longput_p_recommend,

            "train_accuracy_shortput": inputs.joinput.shortput_train_accuracy,
            "test_accuracy_shortput": inputs.joinput.shortput_test_accuracy,
            "random_shortput": inputs.joinput.shortput_p_random,
            "probability_of_profit_shortput": inputs.joinput.shortput_p_recommend,
        }
        experiment = Experiment(api_key="IbrLUBGMAj01omKfPmPiN2XGV", project_name="7773-2023finalproject", workspace="nyu-fre-7773-2021")
        print("Logging metrics")
        experiment.log_metrics(self.metrics_model)
        self.next(self.end)

    @step
    def end(self):
        print("All done!")
        pass
if __name__ == '__main__':
    OptionsTradingFlow()