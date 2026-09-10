import pandas
from sklearn.tree import DecisionTreeClassifier

pandas.set_option("display.max_rows",None)

train_data = pandas.read_csv("train_data.csv")
test_data = pandas.read_csv("test_data.csv")
print(train_data)

model1 = DecisionTreeClassifier()
model1.fit(train_data[["sepal_length","sepal_width","petal_length","petal_width"]], train_data["species"])
test_data["species"] = model1.predict(test_data[["sepal_length","sepal_width","petal_length","petal_width"]])
print(test_data)

model = DecisionTreeClassifier(max_depth=7)
model.fit(train_data[["sepal_length","sepal_width","petal_length","petal_width"]], train_data["species"])
test_data["species"] = model.predict(test_data[["sepal_length","sepal_width","petal_length","petal_width"]])
print(test_data)