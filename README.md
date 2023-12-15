# 7773_Options_Recommender
This is a description documentation for the final project of FRE-GY 7773 completed by Stars Group(Zhizhou Ji, Qi Wu, Lisha Tong).

## 1. Introduction

Simply, this project recommends option investment strategies and specific option contracts (S&P 500 ETF Trust options) that are more likely to be profitable to users. Based on extensive experience that predicting stock prices directly will suck, we leave the work of predicting future S&P 500 return to users. What we do is recommend strategies and specific contracts that are more likely to be profitable based on users’ predictions.
To achieve this, we made an app (my_app.py) based on Streamlit for users to choose their expected return (large positive, slight positive, slight negative, large negative) for S&P 500. These four expected return choices correspond to four investment strategies (long call, bull spread, short call, long put). And based on these four investment strategies, we trained four classification models by using metaflow (my_flow.py) to predict which kind of strike price (in the time, at the time, out the time) options are more likely to be profitable. In addition, our app also allows users to choose expected volatility (increase, decrease). After the user selects the investment date, expected return and expected volatility, our app will automatically call the appropriate strategies and models and provide the user with a specific list of recommended options.
From the COMET graph, we can see that compared to randomly selecting options, our models effectively improve the user's profit probability.

## 2. How to Use
### 2.1 How to Set up Environment
*Since Metaflow is not supported on Windows systems, we recommend running the code on the Sandbox https://outerbounds.com/sandbox/.
*Please run "pip install openpyxl" and "pip install comet_ml" in the terminal, and use the default version of the sandbox for the rest of the dependencies(See requirements).
*Then, put all the files in the GitHub repository in the same folder of the sandbox.(**Including spy_2020_2022.csv**, which cannot be uploaded to Git-hub by us, because it is larger than 25M.)
*Now, you can run our project directly using common python commands.
### 2.2 How to run
*Run **our_flow.py** by typing "COMET_API_KEY=xxx MY_PROJECT_NAME=yyy python our_app.py run" in the sandbox's terminal. Then you will get data and models required by our_app.py(The obtained data and models should be similar to those in **app_data.zip**).
*Then, run **our_app.py** by typing "streamlit run our_app.py" in the sandbox's terminal. Then you can open the UI to use our app.

## 3. Data

### 3.1 SPY Option Chains (spy_2020_2022.csv)
The main dataset we used is a combination of three years of SPDR S&P 500 ETF Trust ($SPY) options end of day quotes ranging from 01-2020 to 12-2022 ([$SPY Option Chains - Q1 2020 - Q4 2022 (kaggle.com)](https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022/data)). **Please download the dataset and save it in this folder.** From this data set, we can get SPY call and put options data with different strike prices at different expiration dates at each quote date, including trading volume, option price, implied volatility and Greeks, etc., with more than one million rows.
### 3.2 Macroeconomics Data (Macro.xlsx)
We want to use some macroeconomic data as features when training models. Therefore, we used the FRED plug-in in Excel to obtain eight different US macroeconomic indicators (unemployment rate, GDP growth rate, M1, M2, Fed target rate, CCPI, 10-year treasury bond yield, umich inflation expectation). In addition, we also used Excel to simply process the data to ensure that the data was of daily frequency.
### 3.3 Underlying ETF Prices (SPY_ETF.csv)
In order to know whether options we recommend are profitable, we need to know the daily prices of the SPY ETF, the underlying asset of the options. We download the data directly from Yahoo Finance (SPDR S&P 500 ETF Trust (SPY) Stock Historical Prices & Data - Yahoo Finance).

## 4. Our Flow

### 4.1 Start
Let's start!
### 4.2 Data Preprocessing
* We intercepted the data from January 2020 to January 2021 (quate date) to ensure that the expiration price of all option data can be obtained.
* We converted the implied volatility into a binary classification (the feature IV_binary), with values greater than the median being 1, otherwise 0.
* We divided all options into three categories (ITM, ATM, OTM) according to strike price.
* We merged Macro.csv into spy_2020_2022.csv.
* We split the data set into call options and put options.
* We removed rows with 0 volume.
### 4.3 Training Models in Parallel
We trained our four models in parallel. Among them, long call and short call models use the data_call dataset, and long put and short put use the data_put dataset. 
#### 4.3.1 Feature Engineering
Since different datasets and different models require different feature engineering, we cannot complete all the feature engineering before training the models separately.
* For each dataset, we need to do some feature engineering. For data_call, we calculated the average trading volume of all three types (ITM, ATM, OTM) of options on each expiration date corresponding to each quotation day.
* For each dataset, we calculated the average IV_binary of options on each expiration date corresponding to each quote date, and converted the average to binary.
* For each model in each dataset, we need to calculate whether investing (long or short) in that option will make a profit or a loss (1 for profit and 0 for loss) based on the expiration price.
* We calculated the average probability of profit of all three types (ITM, ATM, OTM) of options on each expiration date corresponding to each quote date and chose the type with the highest probability of profit as the value of our target variable y (ITM==-1, ATM==0, OTM==1). 
#### 4.3.2 Split Train and Test Dataset
We divide the data before August 31, 2020 into the training set, and the data after that into the test set.
#### 4.3.3 GridSearch and Fit Models
* We chose RandomForest model as our model.
* We took features DTE, Underlying_last, IV_binary, volume, etc. and macro data as X.
* We performed GridSearch on the two hyperparameters n_estimators and max_depth. Then, we chose the best parameters to train our models on the whole dataset.
#### 4.3.4 Make Predictions
We used the trained models to make predictions on test datasets. For example, for each expiration date corresponding to a certain quotation date, if the predicted value of y is 1, the OTM option should be selected.
#### 4.3.5 Calculated Metrics
* We calculated the accuracy of each model on the training set and test set.
* More importantly, we calculated the average profit probability on the test set assuming that investors randomly select options, as well as the average profit probability after using our model.
### 4.4 Join
We joined the separate paths and logged metrics to COMET.
### 4.5 End
Mataflow is finished!

## 5. Our App

We used Streamlit to build a simple app for users to select their expectation and get suggestions.
* We set selectboxes for users to choose a date they want advice on, their expectations of market return and volatility.
* We load models and data obtained from our flow.
* We recommend options investment strategies and specific options contracts for users based on their choices.
* We draw a P/L graph at maturity and show how our models improved the probability of profit.
