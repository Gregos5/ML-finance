# Code written by Idan Malka, Date: 14/06/2023
# This code compares accuracy of three different classifiers methods applied over several stocks
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


print("Comparing 3 ML classifiers on different tickers by accuracy")
print("| TICK | rfc |  bc  |  etc |")
# add as many yfinance tickers 
TICKERS = ["MSFT", "TSLA", "AAPL", "AMZN"]
predictions = [0]*len(TICKERS)
result = [0]*len(TICKERS)
i = 0
for TICKER in TICKERS:

    # download yfinance data into dataframe: 1h resolution
    # update start date to more recent if data is unavailable
    df = yf.download(TICKER, start="2022-01-01", interval="60m")


    # get previous day data
    temp=df["Open"].shift(1)-df["Close"].shift(1)
    # return 1 if pDay up, -1 if pDay down 
    y=[1 if i>0 else -1 for i in temp]


    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.20, shuffle=True)

    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    pred_rfc = rfc.predict(x_test)
    rfc_acc = accuracy_score(y_test, pred_rfc)

    bc = BaggingClassifier()
    bc.fit(x_train, y_train)
    pred_bc = bc.predict(x_test)
    bc_acc = accuracy_score(y_test, pred_bc)

    etc = ExtraTreesClassifier()
    etc.fit(x_train, y_train)
    pred_etc = etc.predict(x_test)
    etc_acc = accuracy_score(y_test, pred_etc)


    result[i] = [rfc_acc, bc_acc, etc_acc]
    i+=1
    print(" ", TICKER, "{:.4f}".format(rfc_acc), "{:.4f}".format(bc_acc), "{:.4f}".format(etc_acc))

x_test["Position"] = pred_rfc

x_test["Delta"] =  x_test["Close"] - x_test["Open"]
x_test["Profit"] = x_test.Position * y_test
x_test["return_strategy"]= x_test["Profit"].cumsum()
# print(x_test.tail())

result=np.asmatrix(result)
maxm = np.amax(result[:,0])

# plt.plot(y_test)
# plt.plot(pred_etc)
x_test.plot(y='return_strategy')
plt.show()