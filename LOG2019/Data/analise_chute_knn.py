from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot
import joblib
import time
nparray = numpy.array
pyplot.style.use('dark_background')


file_names = glob("../ALL/*Chute.csv")

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

knn_out: KNeighborsRegressor = KNeighborsRegressor(n_neighbors=12,
                                                   weights='uniform',
                                                   n_jobs=1).fit(X, y)

joblib.dump(knn_out, "models/avaliacao_chute_knn.sav")

x_axis: nparray = range(1, 200, 1)
score_train: nparray = []
score_test: nparray = []

start: float = time.time()
for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=.2, random_state=i)

    knn: KNeighborsRegressor = KNeighborsRegressor(
        n_neighbors=12,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=3,
        metric='minkowski',
        metric_params=None,
        n_jobs=1
        ).fit(X_train, y_train)
    # knn: RadiusNeighborsRegressor = RadiusNeighborsRegressor(radius=5, weights='uniform').fit(X_train, y_train)

    score_test.append(knn.score(X_test, y_test))
    score_train.append(knn.score(X_train, y_train))

end: float = time.time()

print("Score test: ", numpy.mean(score_test))
print("Score train: ", numpy.mean(score_train))
print("Time of operation: {} ms".format(
    (end-start)*1e3/(numpy.size(x_axis)*numpy.size(y)))
      )

pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.xlabel('random_state')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()

pyplot.show()
