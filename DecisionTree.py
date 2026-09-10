import pandas as pd
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.max_rows",None)

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")
print(train_data)

model = DecisionTreeClassifier(max_depth=5)
model.fit(train_data[["sepal_length","sepal_width","petal_length","petal_width"]], train_data["species"])
test_data["species"] = model.predict(test_data[["sepal_length","sepal_width","petal_length","petal_width"]])
print(test_data)

df = pd.read_csv('test_data_answers.csv')

count = 0
for index,row in df.iterrows():
    if row['species'] == test_data['species'][index]:
        count += 1
print(str(count) + '/30')

test_data.to_csv('results_decision_tree.csv',index=False)
