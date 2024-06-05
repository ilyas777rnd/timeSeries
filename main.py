# импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# принудительно отключим предупреждения системы
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# импортируем файл с данными о пассажирах
passengers = pd.read_csv("passengers.csv")
passengers.set_index('Month', inplace=True)
passengers.index = pd.to_datetime(passengers.index)

# обучающая выборка будет включать данные до декабря 1959 года включительно
train = passengers[:'1959-12']

# тестовая выборка начнется с января 1960 года (по сути, один год)
test = passengers['1960-01':]

plt.plot(train, color="black")
plt.plot(test, color="red")

# заголовок и подписи к осям
plt.title('Разделение данных о перевозках на обучающую и тестовую выборки')
plt.ylabel('Количество пассажиров')
plt.xlabel('Месяцы')

# добавим сетку
# plt.grid()
# plt.show()

# создадим объект этой модели
model = SARIMAX(train,
                order=(3, 0, 0),
                seasonal_order=(0, 1, 0, 12))

# применим метод fit
result = model.fit()

# тестовый прогнозный период начнется с конца обучающего периода
start = len(train)

# и закончится в конце тестового
end = len(train) + len(test) - 1

# применим метод predict
predictions = result.predict(start, end)
print(predictions)

# выведем три кривые (обучающая, тестовая выборка и тестовый прогноз)
plt.plot(train, color="black")
plt.plot(test, color="red")
plt.plot(predictions, color="green")

# заголовок и подписи к осям
plt.title("Обучающая выборка, тестовая выборка и тестовый прогноз")
plt.ylabel('Количество пассажиров')
plt.xlabel('Месяцы')

# добавим сетку
plt.grid()

plt.show()

# настроим поиск параметров на обучающей выборке
parameter_search = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True,
                              d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)  #

# выведем результат
print(parameter_search.summary())
