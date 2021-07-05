import pandas as pd
df = pd.read_excel("C:\\Users\\jasyan\\Desktop\\occ.xlsx")
df.dropna(inplace=True)
df.head()

from statsmodels.tsa.api import ExponentialSmoothing

data_sr = pd.Series(df["occ"])
fit1 = ExponentialSmoothing(data_sr, seasonal_periods=7, trend='add', seasonal='add').fit(use_boxcox=True)
fit2 = ExponentialSmoothing(data_sr, seasonal_periods=7, trend='add', seasonal='mul').fit(use_boxcox=True)
fit3 = ExponentialSmoothing(data_sr, seasonal_periods=7, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
fit4 = ExponentialSmoothing(data_sr, seasonal_periods=7, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)

l1, = plt.plot(list(fit1.fittedvalues) + list(fit1.forecast(5)), marker='^')
l2, = plt.plot(list(fit2.fittedvalues) + list(fit2.forecast(5)), marker='*')
l3, = plt.plot(list(fit3.fittedvalues) + list(fit3.forecast(5)), marker='.')
l4, = plt.plot(list(fit4.fittedvalues) + list(fit4.forecast(5)), marker='.')

df["aa"] = list(fit1.fittedvalues)
df["am"] = list(fit2.fittedvalues)
df["aa damped"] = list(fit3.fittedvalues)
df["am damped"] = list(fit4.fittedvalues)

l5, = plt.plot(df["occ"], marker='.')
plt.legend(handles=[l1, l2, l3, l4, l5], labels=["aa", "am", "aa damped", "am damped", "data"], loc='best',
           prop={'size': 7})

plt.show()

df.to_excel("C:\\Users\\jasyan\\Desktop\\df_forcast.xlsx")