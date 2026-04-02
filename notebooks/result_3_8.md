# Задача

Ислледовать эвристические модификации метода Ньютона

# Функции

Функция Билла

Exponential Loss

# Данные

gisette для Exponential Loss, точка $x_0 = 0_n$

Метод Ньютона работал слишком долго на gisette, поэтому кол-во параметров было случайным образом уменьшено до 1000

# Железо

CPU: Intel i5-12700H

# Метод

Метод Ньютона

# Результаты

Иногда shift метод находит оптимум, а spectral метод не находит.


Траектории из разных начальных точек. Функция Билла

![Alt text](../figs/task8/traj_1.png)
![Alt text](../figs/task8/traj_2.png)
![Alt text](../figs/task8/traj_3.png)
![Alt text](../figs/task8/traj_4.png)
![Alt text](../figs/task8/traj_5.png)
![Alt text](../figs/task8/traj_6.png)
![Alt text](../figs/task8/traj_7.png)
![Alt text](../figs/task8/traj_8.png)
![Alt text](../figs/task8/traj_9.png)
![Alt text](../figs/task8/traj_10.png)

График изменения $\gamma$

![График изменения $\gamma$](../figs/task8/gamma_evo.png)

Можно увидеть, что оно очень быстро уменьшается

Вот геометрическое обоснование, что shift метод при большом $\gamma$ ведет себя как градиентный спуск, а спектральное усечение сохраняет "нью-
тоновское"поведение вдоль осей положительной кривизны.

![Alt text](../figs/task8/geom_p1.png)
![Alt text](../figs/task8/geom_p2.png)
![Alt text](../figs/task8/geom_p3.png)
![Alt text](../figs/task8/geom_p4.png)


Вот сравнение скорости методов. ExponentialLoss, gisette

![Вот сравнение скорости методов. ExponentialLoss, gisette](../figs/task8/time_comp.png)


Методы работают примерно с такой же скоростью. Разве что shift метод немног быстрее, видимо, из-за более обусловленной матрицы 



