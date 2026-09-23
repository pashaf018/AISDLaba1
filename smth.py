from random import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def predict_manually(sepal_length,sepal_width,petal_length,petal_width):
    if sepal_width > 3.8 or petal_length <= 1.9 or petal_width <= 0.6:
        return 'setosa'
    if sepal_length < 7 and (sepal_width < 2.2 or petal_length < 4.5 or petal_width < 1.4):
        return 'versicolor'
    return 'virginica'

df = pd.read_csv("iris.csv")

#Задание 2.a
print("\n\033[38;5;166mЗадание 2.a: Определить признаки объекта.\033[0m")
print("\tПризнаки объекта:")
keys = df.keys()
for i in range(len(keys)-1):
    print("\t\t" + str(i + 1) + ".\t" + keys[i])

#Задание 2.b
print("\n\033[38;5;166mЗадание 2.b: Определить баланс датасета(соотношение объектов между классами)\033[0m")
print("\tБаланс датасета:")
print("\t\tВид:\t\tКол-во:\t\tСоотношение:")
distribution = df['species'].value_counts()
props = df['species'].value_counts(normalize=True) * 100
balance = pd.concat(objs=[distribution,props],axis=1)
for species,count,prop in balance.itertuples():
    print(f"\t\t{species:15}{count:<10}{prop:<5.2f}%")

#Задание 2.c
print("\n\033[38;5;166mЗадание 2.c: Определить метки как разделяются объекты\033[0m")
with pd.option_context('display.max_columns', None):
    stats = df.groupby('species').agg(['mean','std','min','max'])
    print(stats)
#sns.pairplot(df,hue='species',markers='o')
#plt.suptitle('Разделение объектов по меткам', y=1.02)
#plt.show()

#Задание 2.d
print("\n\033[38;5;166mЗадание 2.d: Вручную сформировать правила\033[0m")
print("\tПравила:")
print("\t\tЕСЛИ sepal_width > 3.8 OR petal_length <= 1.9 OR petal_width <= 0.6 ТО setosa")
print("\t\tЕСЛИ sepal_width < 7 AND sepal_width < 2.2 OR petal_length < 4.5 OR petal_width < 1.4 ТО versicolor")
print("\t\tЕСЛИ НЕ setosa И НЕ versicolor ТО virginica")

#Задание 2.e
print("\n\033[38;5;166mЗадание 2.e: Сформировать прогноз - построить соответствие между объектом и классом с помощью"
      "\nклассов, автоматизированно через сформированные правила. Сохранить в csv файл.\033[0m")

test_data_manually = df
test_data_manually['prediction'] = [
    predict_manually(row[keys[0]],row[keys[1]],row[keys[2]],row[keys[3]])
    for _, row in df.iterrows()
]
keys = keys.append(pd.Index(['prediction']))

print("\tСоответствия:")
total_by_class = test_data_manually[keys[4]].value_counts()

test_data_manually['is_correct'] = test_data_manually[keys[4]] == test_data_manually[keys[5]]
results_by_class = test_data_manually.groupby(keys[4])['is_correct'].sum()
results_manually = pd.concat([results_by_class,total_by_class],axis=1)

print("\t\tВид:\tПравильно\tОбщее кол-во")
for species,corrects,total in results_manually.itertuples():
    print(f"\t\t{species:<15}{corrects:<15}{total:<15}")
print("\t\tОбщая точность: " + str(results_by_class.values.sum()) + '/' + str(total_by_class.values.sum()))

filename_manually = "test_data_manually.csv"
test_data_manually.to_csv(filename_manually)
print(test_data_manually)
print("\033[32mСохранение в csv файл: " + filename_manually + "\033[0m")

#Задание 3
print("\n\033[38;5;166mЗадание 3: Обучить модель дерево решений(Попробовать разные значения параметра глубины дерева 3-5-7-??)."
      "\nСохранить в csv файл")

depths = range(3,15,2)
train_acc,test_acc =  [],[]
train_prec,test_prec = [],[]

X = df.drop(['species'])
Y = df['species']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, shuffle=True,random_state=42)

for d in depths:
    model = DecisionTreeClassifier(max_depth=d,random_state=42)
    model.fit(X_train,Y_train)

    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

