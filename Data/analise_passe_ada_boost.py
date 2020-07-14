from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from matplotlib import pyplot
import pickle
import joblib

nparray = numpy.array
pyplot.style.use('dark_background')


file_names = glob("/home/robofei/Documents/DataAnalyse/ALL/*Passe.csv")

# print(file_names)

array_passe: nparray = []


for f in file_names:
    array_passe.append(
        numpy.genfromtxt(
            f,
            dtype=int,
            delimiter=";",
            skip_header=1
        )
    )

array_passe = numpy.concatenate(array_passe)

y: nparray = array_passe[:, 0]
X: nparray = array_passe[:, [1, 2, 3,  4, 4, 6, 7, 8]]

x_axis: nparray = range(1, 50)
score_train: nparray = []
score_test: nparray = []

for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=38
        )

    ada_boost: AdaBoostRegressor = AdaBoostRegressor(
        base_estimator=None,
        n_estimators=7,
        learning_rate=1,
        loss='square',
        random_state=i
    ).fit(X_train, y_train)

    score_test.append(ada_boost.score(X_test, y_test))
    score_train.append(ada_boost.score(X_train, y_train))


pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.xlabel('random_state')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()

pyplot.show()
