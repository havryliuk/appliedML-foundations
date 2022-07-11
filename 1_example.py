from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter('ignore', category=ConvergenceWarning)

iris = load_iris()
iris_dataframe = DataFrame(load_iris().data)
print(iris_dataframe.head())
print(iris_dataframe.describe())

model = LogisticRegression().fit(iris['data'], iris['target'])
print(model.predict(iris['data']))
