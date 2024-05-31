# импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# импортируем файл с данными о пассажирах
passengers = pd.read_csv("passengers.csv")
passengers.set_index('Month', inplace = True)
passengers.index = pd.to_datetime(passengers.index)

# зададим размер графика
plt.figure(figsize=(15, 8))

# поочередно зададим кривые (перевозки и скользящее среднее) с подписями и цветом
plt.plot(passengers, label='Перевозки пассажиров по месяцам', color='steelblue')
plt.plot(passengers.rolling(window=12).mean(), label='Скользящее среднее за 12 месяцев', color='orange')

# добавим легенду, ее положение на графике и размер шрифта
plt.legend(title='', loc='upper left', fontsize=14)

# добавим подписи к осям и заголовки
plt.xlabel('Месяцы', fontsize=14)
plt.ylabel('Количество пассажиров', fontsize=14)
plt.title('Перевозки пассажиров с 1949 по 1960 год', fontsize=16)

# выведем обе кривые на одном графике
plt.show()
