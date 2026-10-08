from random import random

import pandas as pd
import numpy as np
from scipy.constants import metric_ton
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
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
test_data_manually.to_csv(filename_manually,index=False)
print(test_data_manually)
print("\033[32mСохранение в csv файл: " + filename_manually + "\033[0m")

#Задание 3
print("\n\033[38;5;166mЗадание 3: Обучить модель дерево решений(Попробовать разные значения параметра глубины дерева 3-5-7-??)."
      "\nСохранить в csv файл\033[0m")

depths = range(3,15,2)
train_acc,test_acc =  [],[]
train_prec,test_prec = [],[]

X = df.drop(columns=['species','prediction','is_correct'])
Y = df['species']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, shuffle=True,random_state=10)

for d in depths:
    model = DecisionTreeClassifier(max_depth=d,random_state=42)
    model.fit(X_train,Y_train)

    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    train_acc.append(accuracy_score(Y_train,Y_train_pred))
    train_prec.append(precision_score(Y_train,Y_train_pred,average='macro'))
    test_acc.append(accuracy_score(Y_test, Y_test_pred))
    test_prec.append(precision_score(Y_test, Y_test_pred, average='macro'))

filename_DecisionTree = 'test_data_DecisionTree.csv'
test_data_DecisionTree = pd.concat(objs=[X_test,pd.DataFrame({
    'species': Y_test,
    'prediction': Y_test_pred,
    'is_correct': Y_test == Y_test_pred
})],axis=1)
print("\033[32mСохранение в csv файл: " + filename_DecisionTree + "\033[0m")
test_data_DecisionTree.to_csv(filename_DecisionTree,index=False)

_, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))

ax1.plot(depths,train_acc,'o-',label='Train Accuracy')
ax1.plot(depths,train_prec,'o--',label='Train Precision')
ax1.set_xlabel("Глубина дерева")
ax1.set_ylabel("Точность")
ax1.set_title('Train')
ax1.legend()
ax1.grid(True)

ax2.plot(depths,test_acc,'s-',label='Test Accuracy')
ax2.plot(depths,test_prec,'s--',label='Test Precision')
ax2.set_xlabel("Глубина дерева")
ax2.set_ylabel("Точность")
ax2.set_title('Test')
ax2.legend()
ax2.grid(True)

plt.suptitle('Decision Tree')
plt.tight_layout()
plt.show()

#Задание 4
print("\n\033[38;5;166mЗадание 4: Обучить модель линейной регрессии(Попробовать разные значения итераций обучения 10,...100)."
      "\nСохранить в csv файл\033[0m")

losses_train, losses_test = [],[]
train_acc_reg,test_acc_reg = [],[]
train_prec_reg,test_prec_reg = [],[]
l2_vec_norms = []
l2_update_norms = []
iterations = []

prev_weights = None
model_lr = LogisticRegression(C=1, solver='lbfgs', max_iter=10,warm_start=True,random_state=42)

for i in range(30):
    iterations.append((i + 1) * 10)
    model_lr.fit(X_train,Y_train)

    weights = model_lr.coef_[0]

    l2_vec_norms.append(np.linalg.norm(weights))

    if prev_weights is not None:
        l2_update_norms.append(np.linalg.norm(weights - prev_weights))
    else:
        l2_update_norms.append(0)
    prev_weights = weights.copy()

    Y_pred_train = model_lr.predict(X_train)
    Y_pred_test = model_lr.predict(X_test)
    Y_proba_train = model_lr.predict_proba(X_train)
    Y_proba_test = model_lr.predict_proba(X_test)

    losses_train.append(log_loss(Y_train,Y_proba_train))
    losses_test.append(log_loss(Y_test,Y_proba_test))
    train_acc_reg.append(accuracy_score(Y_train,Y_pred_train))
    test_acc_reg.append(accuracy_score(Y_test, Y_pred_test))
    train_prec_reg.append(precision_score(Y_train,Y_pred_train,zero_division=0,average='macro'))
    test_prec_reg.append(precision_score(Y_test, Y_pred_test, zero_division=0,average='macro'))

    print(f"Итерация: {(i + 1) * 10}, train_acc = {train_acc_reg[-1]}, test_acc = {test_acc_reg[-1]}, l2 = {l2_vec_norms[-1]}, n_iter = {model_lr.n_iter_[0]}")

_, axes = plt.subplots(2,3, figsize=(18,10))

axes[0, 0].plot(iterations, losses_train, 'o-', label='Тренировочные потери')
axes[0, 0].plot(iterations, losses_test, 's-', label='Тестовые потери')
axes[0, 0].set_xlabel('Итерация')
axes[0, 0].set_ylabel('Потери')
axes[0, 0].set_title('График потерь')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(iterations, train_acc_reg, 'o-', label='Train Accuracy')
axes[0, 1].plot(iterations, test_acc_reg, 's-', label='Test Accuracy')
axes[0, 1].set_xlabel('Итерация')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Точность')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[0, 2].plot(iterations, train_prec_reg, 'o-', label='Train Precision')
axes[0, 2].plot(iterations, test_prec_reg, 's-', label='Test Precision')
axes[0, 2].set_xlabel('Итерация')
axes[0, 2].set_ylabel('Precision')
axes[0, 2].set_title('Precision')
axes[0, 2].legend()
axes[0, 2].grid(True)

axes[1, 0].plot(iterations, l2_vec_norms, 'o-')
axes[1, 0].set_xlabel('Итерация')
axes[1, 0].set_ylabel('L2-норма')
axes[1, 0].set_title('L2-норма вектора весов')
axes[1, 0].grid(True)

axes[1, 1].plot(iterations, l2_update_norms, 'o-')
axes[1, 1].set_xlabel('Итерация')
axes[1, 1].set_ylabel('Норма обновления')
axes[1, 1].set_title('L2-норма изменения весов')
axes[1, 1].grid(True)

axes[1, 2].axis('off')

plt.tight_layout()
plt.show()