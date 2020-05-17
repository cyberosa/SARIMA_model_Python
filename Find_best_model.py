import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from math import sqrt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from warnings import catch_warnings, filterwarnings
from multiprocessing import cpu_count
from joblib import Parallel, delayed

def sarima_forecast2(history, config):
    order, sorder, trend, exog_raw = config

    try:
       # define model
        model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
        # fit model
        model_fit = model.fit(disp=False)
        # make one step forecast, start and end are the same
        #yhat = model_fit.predict(start=len(history), end=len(history), exog=exog_raw[len(history),:])
        pre = model_fit.get_prediction(start=len(history), end=len(history))
        yhat = pre.predicted_mean
        yhat_ci = pre.conf_int(alpha=0.5)
    except Exception as e:
        print(e)
        print("No prediction")
        print("Length: "+str(len(history)))
        yhat = [0] # no prediction
        yhat_ci = [0]
        model = None
        model_fit = None
    return yhat[0], yhat_ci, model, model_fit

# split a univariate dataset into train/test sets
# n_test: number of time steps to use in the test set
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

def walk_forward_validation2(data, n_test, cfg):
    predictions_df = pd.DataFrame()
    predictions = list()
    predictions_lci = list()
    predictions_hci = list()
    # split dataset
    train, test = train_test_split(data, n_test)

    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat, yhat_ci, model, results = sarima_forecast2(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        predictions_lci.append(yhat_ci[:,0])
        predictions_hci.append(yhat_ci[:,1])
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    #error = measure_rmse(test, predictions)
    predictions_df["PRED"] = predictions
    predictions_df["PRED_LCI"] = predictions_lci
    predictions_df["PRED_HCI"] = predictions_hci
    return [test, predictions_df, model, results]

def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

def score_model(data, n_test, cfg):
    result = None
    # convert config to a key
    order, sorder, trend, exog_data = cfg

    key = str(order)+', '+str(sorder)+', '+str(trend)
    # show all warnings and fail on exception if debugging
    # one failure during model validation suggests an unstable config
    try:
        # never show warnings when grid searching, too noisy
        with catch_warnings():
            filterwarnings("ignore")
            test, predictions_df, model, results = walk_forward_validation2(data, n_test, cfg)
            result = measure_rmse(test, predictions_df['PRED'])
    except:
        result = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)

def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        try:
            #  Parallel object with the number of cores to use and set it to the number of scores detected in your hardware
            executor = Parallel(n_jobs=cpu_count()-1, backend='multiprocessing', verbose=10)
            tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
            scores = executor(tasks)
        except:
            scores = [None]
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

def sarima_configs(seasonal=[0], exog_data=None):
    models = list()
    # define config lists
    p_params = [0, 1, 2, 3]
    d_params = [1]
    q_params = [0, 1, 2, 3]
    t_params = ['n','c','t','ct']
    P_params = [0]
    D_params = [0]
    Q_params = [0]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t, exog_data]
                                    models.append(cfg)
    return models

with open("daily-min-temperatures.csv") as f:
    temp_df = pd.DataFrame(csv.DictReader(f))

temp_df['Date'] = pd.to_datetime(temp_df.Date)

temp_df.set_index('Date')

temp_df['Temp'] = pd.to_numeric(temp_df.Temp)

if __name__ == '__main__':
    # define dataset
    data = temp_df['Temp'].values
    # data split (we have more than 3000 days)
    n_test = 265
    # model configs
    # To add seasonality of one year, we add 365, because we have daily data
    #cfg_list = sarima_configs(seasonal=[0, 12]) <- if we have monthly data
    cfg_list = sarima_configs(seasonal=[0],exog_data=None)
    print("Testing " + str(len(cfg_list)) + "models")
    # grid search
    scores = grid_search(data, cfg_list, n_test)
    #scores = grid_search(data, cfg_list, n_test)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)