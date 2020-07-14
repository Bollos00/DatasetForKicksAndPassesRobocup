
from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from matplotlib import pyplot
import pickle
import joblib

nparray = numpy.array
pyplot.style.use('dark_background')


file_names = glob("/home/robofei/Documents/DataAnalyse/ALL/*Chute.csv")

array_chute: nparray = []


for f in file_names:
    array_chute.append(
        numpy.genfromtxt(
            f,
            dtype=numpy.uint8,
            delimiter=";",
            skip_header=1
        )
    )

array_chute = numpy.concatenate(array_chute)

y: nparray = array_chute[:, 0]
X: nparray = array_chute[:, [1, 2, 3]]

x_axis: nparray = range(1, 200)
score_train: nparray = []
score_test: nparray = []

for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=25
        )

    ada_boost: AdaBoostRegressor = AdaBoostRegressor(
        base_estimator=None,
        n_estimators=1,
        learning_rate=1,
        loss='linear',
        random_state=i
    ).fit(X_train, y_train)

    score_test.append(ada_boost.score(X_test, y_test))
    score_train.append(ada_boost.score(X_train, y_train))

pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.xlabel('random_state')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()

pyplot.show()
