import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime

st.markdown("# Option Recommender")
st.write("Group Project for Machine Learning")
st.write("*Qi Wu, Zhizhou Ji, Lisha Tong*")

st.markdown("## Introduction")
intro = "This is an option recommender, which takes your expectation of market returns and volatility,\
        and gives the recommended option type(call or put), strategy(long, short or spread),\
		and strike price(ATM/ITM/OTM)."
st.write(intro)

# 1. get expectation
st.write("Enter a date")
date = st.date_input(
    "You want option recommendation on ...",
    min_value=datetime.date(2020,1,2), max_value=datetime.date(2022,12,30),
    value=datetime.date(2020,1,2)
    )
date = pd.to_datetime(date)
print(date)
print(type(date))

r = st.selectbox(
   "Your expectation of market return:",
   ("Large positive", "Slight positive", "Slight negative", "Large negative"),
   index=None,
   placeholder="Select expected return...",
)
iv = st.selectbox(
   "Your expectation of volatility:",
   ("High", "Low"),
   index=None,
   placeholder="Select expected valatility...",
)
if r and iv:
    st.write(f'You selected:{r.lower()} return, {iv.lower()} volatility')

t = st.number_input(
    "Expected investment horizon:(>30 days)",
    value=None,
    placeholder="Enter investment horizon in days"
)
# 2. load models and data
long_call = pickle.load(open('model_longcall.pkl','rb'))
short_call = pickle.load(open('model_shortcall.pkl','rb'))
long_put = pickle.load(open('model_longput.pkl','rb'))
short_put = pickle.load(open('model_shortput.pkl','rb'))
#call = pd.read_csv()
#put = pd.read_csv()
call_train = pd.read_csv('call_train.csv',index_col='Unnamed: 0')
call_test = pd.read_csv('call_test.csv', index_col='Unnamed: 0')
call = pd.merge(call_train, call_test,how='outer')

put_train = pd.read_csv('put_train.csv',index_col='Unnamed: 0')
put_test = pd.read_csv('put_test.csv', index_col='Unnamed: 0')
put = pd.merge(put_train, put_test,how='outer')

call[' [QUOTE_DATE]'] = pd.to_datetime(call[' [QUOTE_DATE]'])
put[' [QUOTE_DATE]'] = pd.to_datetime(put[' [QUOTE_DATE]'])

quote_date = call[' [QUOTE_DATE]'].unique()

if r and iv and t and date:
    # 3. give recommendation(s)
    date_chosen = quote_date[quote_date>=date][0]
    st.write("## Recommendation")
    iv = 1 if iv == 'High' else 0
    XTM = {-1: "ITM",
        0: "ATM",
        1: "OTM"}
    

    if r == 'Large positive':
        model = long_call
        data_the_day = call.loc[call[' [QUOTE_DATE]']==date_chosen,:].copy()
        x = abs(data_the_day[" [DTE]"]-t).idxmin()
        data_the_day = data_the_day.loc[x,:].copy()
        data_the_day['C_IV_binary'] = iv
        data_the_day.drop(['type_call',' [QUOTE_DATE]','y_longcall'], inplace=True)
        ret = model.predict(np.array(data_the_day).reshape(1,-1))[0]
        recommend = XTM[ret]
        st.write(f"Recommend buying {recommend} call options")
    elif r == 'Slight positive':
        model1 = long_call
        model2 = short_call
        data_the_day = call.loc[call[' [QUOTE_DATE]']==date_chosen,:].copy()
        x = abs(data_the_day[" [DTE]"]-t).idxmin()
        data_the_day = data_the_day.loc[x,:].copy()
        data_the_day['C_IV_binary'] = iv
        data_the_day.drop(['type_call',' [QUOTE_DATE]','y_longcall'], inplace=True)
        ret1 = model1.predict(np.array(data_the_day).reshape(1,-1))[0]
        ret2 = model2.predict(np.array(data_the_day).reshape(1,-1))[0]
        if ret1>=ret2:
            model = short_put
            data_the_day = put.loc[put[' [QUOTE_DATE]']==date_chosen,:].copy()
            x = abs(data_the_day[" [DTE]"]-t).idxmin()
            data_the_day = data_the_day.loc[x,:].copy()
            data_the_day['P_IV_binary'] = iv
            data_the_day.drop(['type_put',' [QUOTE_DATE]','y_longput'], inplace=True)
            ret = model.predict(np.array(data_the_day).reshape(1,-1))[0]
            recommend = XTM[ret]
            st.write(f"Recommend selling {recommend} put options")
        else:
            recommend1 = XTM[ret1]
            recommend2 = XTM[ret2]
            st.write(f"Recommend construting bull spread by buying {recommend1} and selling {recommend2} call options")
    elif r == 'Slight negative':
        model = short_call
        data_the_day = call.loc[call[' [QUOTE_DATE]']==date_chosen,:].copy()
        x = abs(data_the_day[" [DTE]"]-t).idxmin()
        data_the_day = data_the_day.loc[x,:].copy()
        data_the_day['C_IV_binary'] = iv
        data_the_day.drop(['type_call',' [QUOTE_DATE]','y_longcall'], inplace=True)
        ret = model.predict(np.array(data_the_day).reshape(1,-1))[0]
        recommend = XTM[ret]
        st.write(f"Recommend selling {recommend} call options")
    elif r == 'Large negative':
        model = long_put
        data_the_day = put.loc[put[' [QUOTE_DATE]']==date_chosen,:].copy()
        x = abs(data_the_day[" [DTE]"]-t).idxmin()
        data_the_day = data_the_day.loc[x,:].copy()
        data_the_day['P_IV_binary'] = iv
        data_the_day.drop(['type_put',' [QUOTE_DATE]','y_longput'], inplace=True)
        ret = model.predict(np.array(data_the_day).reshape(1,-1))[0]
        recommend = XTM[ret]
        st.write(f"Recommend buying {recommend} put options")


    # 4. overall probability of profit
    st.markdown("## Overall Performance")

    probability_of_profit_longcall = 0.16348785849584846
    probability_of_profit_longput = 0.3083616313089608
    probability_of_profit_shortcall = 0.6491263750560466
    probability_of_profit_shortput = 0.7123389069400223

    random_longcall = 0.1039699094228833
    random_longput = 0.18102906437958657
    random_shortcall = 0.6925712909840342
    random_shortput = 0.6162022059787197

    test_accuracy_longcall = 0.8223804679552391
    test_accuracy_longput = 0.7740228013029316
    test_accuracy_shortcall = 0.7808748728382503
    test_accuracy_shortput = 0.6195032573289903

    train_accuracy_longcall = 0.8006069260978222
    train_accuracy_longput = 0.8869153873616565
    train_accuracy_shortcall = 0.8933416636915388
    train_accuracy_shortput = 0.9242234916101393

    st.write('Long Call')
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("Profitability Random", 
               str(random_longcall*100)[:4]+"%")
    lc2.metric("Profitability Recommend", 
               str(probability_of_profit_longcall*100)[:4]+"%",
               str(probability_of_profit_longcall/random_longcall*100-100)[:4]+"%")
    lc3.metric("Long Call Train Accuracy", 
               str(train_accuracy_longcall*100)[:4]+"%")
    lc4.metric("Long Call Test Accuracy", 
               str(test_accuracy_longcall*100)[:4]+"%",
               str(test_accuracy_longcall/train_accuracy_longcall*100-100)[:4]+"%")
    
    st.write('Short Call')
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Profitability Random", 
               str(random_shortcall*100)[:4]+"%")
    sc2.metric("Profitability Recommend", 
               str(probability_of_profit_shortcall*100)[:4]+"%",
               str(probability_of_profit_shortcall/random_shortcall*100-100)[:4]+"%")
    sc3.metric("Short Call Train Accuracy", 
               str(train_accuracy_shortcall*100)[:4]+"%")
    sc4.metric("Short Call Test Accuracy", 
               str(test_accuracy_shortcall*100)[:4]+"%",
               str(test_accuracy_shortcall/train_accuracy_shortcall*100-100)[:5]+"%")

    st.write('Long Put')
    lp1, lp2, lp3, lp4 = st.columns(4)
    lp1.metric("Profitability Random", 
               str(random_longput*100)[:4]+"%")
    lp2.metric("Profitability Recommend", 
               str(probability_of_profit_longput*100)[:4]+"%",
               str(probability_of_profit_longput/random_longput*100-100)[:4]+"%")
    lp3.metric("Long Put Train Accuracy", 
               str(train_accuracy_longput*100)[:4]+"%")
    lp4.metric("Long Put Test Accuracy", 
               str(test_accuracy_longput*100)[:4]+"%",
               str(test_accuracy_longput/train_accuracy_longput*100-100)[:5]+"%")

    st.write('Short Put')
    sp1, sp2, sp3, sp4 = st.columns(4)
    sp1.metric("Profitability Random", 
               str(random_shortput*100)[:4]+"%")
    sp2.metric("Profitability Recommend", 
               str(probability_of_profit_shortput*100)[:4]+"%",
               str(probability_of_profit_shortput/random_shortput*100-100)[:4]+"%")
    sp3.metric("Short Put Train Accuracy", 
               str(train_accuracy_shortput*100)[:4]+"%")
    sp4.metric("Short Put Test Accuracy", 
               str(test_accuracy_shortput*100)[:4]+"%",
               str(test_accuracy_shortput/train_accuracy_shortput*100-100)[:5]+"%")



#auther: Qi Wu

#Timestamp: 2023-12-14 22:59:43