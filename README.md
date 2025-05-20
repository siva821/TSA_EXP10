### Name :Rajeshwari.M
### Reg no: 212224040262
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 20/05/25

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


data = pd.read_csv('/content/powerconsumption.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)


plt.plot(data.index, data['PowerConsumption_Zone1'])
plt.xlabel('Date')
plt.ylabel('Power Consumption Zone 1')
plt.title('Power Consumption Zone 1 Time Series')
plt.show()


def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


check_stationarity(data['PowerConsumption_Zone1'])


plot_acf(data['PowerConsumption_Zone1'])
plt.show()
plot_pacf(data['PowerConsumption_Zone1'])
plt.show()


train_size = int(len(data) * 0.8)
train, test = data['PowerConsumption_Zone1'][:train_size], data['PowerConsumption_Zone1'][train_size:]


sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()


predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)


mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)


plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Power Consumption Zone 1')
plt.title('SARIMA Model Predictions for Power Consumption Zone 1')
plt.legend()
plt.show()
~~~

### OUTPUT:

![image](https://github.com/user-attachments/assets/5b92e3af-d870-45a1-8deb-d322efe4c8ca)

![image](https://github.com/user-attachments/assets/52f750d2-f197-41bf-ba89-5c8d73c5408c)

![image](https://github.com/user-attachments/assets/526ac545-9b50-43ff-9f7e-4c74b37223fe)

![image](https://github.com/user-attachments/assets/3d5147f1-0a53-4f37-a989-c4ba0077ab7a)

![image](https://github.com/user-attachments/assets/e4be4d33-f7e0-4d9b-aef0-4b57c0dcbf39)

![image](https://github.com/user-attachments/assets/707de61b-5c2f-4ae9-9f48-ed857a621551)

### RESULT:
Thus the program is successfully implemented based on the SARIMA model.
