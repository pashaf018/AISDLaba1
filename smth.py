import pandas as pd
from sklearn.model_selection import train_test_split

#COUNT

#setosa = 50
#versicolor = 50
#virginica = 50


#               sepal_length    sepal_width     petal_length    petal_width
#setosa
#min_values         4.3             2.3             1.0             0.1
#max_values         5.8             4.4             1.9             0.6

#versicolor
#min_values         4.9             2.0             3.0             1.0
#max_values         7.0             3.4             5.1             1.8

#virginica
#min_values         4.9             2.2             4.5             1.4
#max_values         7.9             3.8             6.9             2.5

#Правила
#Если petal_width < 0.6 AND petal_length < 1.9 ТО setosa
#Если sepal_width < 3.4 AND petal_length < 5.1 AND petal_width < 1.8 ТО versicolor
#Если НЕ setosa AND НЕ versicolor ТО virginica

df = pd.read_csv("test_data.csv")

dict = []
for index,row in df.iterrows():
    if int(row['petal_width']) < 0.6 and int(row['petal_length']) < 1.9:
        dict.insert(index,'setosa')
        continue
    if int(row['sepal_width']) < 3.4 and int(row['petal_length']) < 5.1 and int(row['petal_width']) < 1.8:
        dict.insert(index,'versicolor')
        continue
    dict.insert(index,'virginica')

df['species'] = dict

print(df)

results = pd.read_csv('test_data_answers.csv')
print(results)

df.to_csv('results_hand.csv',index=False)

# new_df = pd.DataFrame(data_to_save,index=None)
# new_df.to_csv('results_hands.csv',index=False,encoding='utf-8-sig')


# with open("iris.csv","r") as dataSet:
#     dataSetReader = csv.DictReader(dataSet)
#
#     values = dataSetReader.fieldnames
#     next(dataSetReader)
#     minValue = [999,999,999,999]
#     maxValue = [-1,-1,-1,-1]
#     for line in dataSetReader:
#         if line["species"] != "setosa":
#             continue
#         for i in range(4):
#             minValue[i] = min(minValue[i],float(line[values[i]]))
#             maxValue[i] = max(maxValue[i],float(line[values[i]]))
#     print(values)
#     print(minValue)
#     print(maxValue)
