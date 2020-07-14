
from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
import pickle
import joblib
import time

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

x_axis: nparray = range(1, 200, 1)
score_train: nparray = []
score_test: nparray = []

start: float = time.time()
for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=i
        )

    forest: RandomForestRegressor = RandomForestRegressor(
        n_estimators=8,
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=6,
        min_weight_fraction_leaf=0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=i*5,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None
    ).fit(X_train, y_train)

    score_test.append(forest.score(X_test, y_test))
    score_train.append(forest.score(X_train, y_train))

end: float = time.time()

print("Score test: ", numpy.mean(score_test))
print("Score train: ", numpy.mean(score_train))
print("Time of operation: {} ms".format((end-start)*1e3/numpy.size(x_axis)))

pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.xlabel('???')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()

pyplot.show()
