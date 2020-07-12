from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from matplotlib import pyplot
import pickle
import joblib

nparray = numpy.array
pyplot.style.use('dark_background')

def customized_weights(distances: nparray)->nparray:

    weights: nparray = nparray(numpy.full(distances.shape, 0), dtype='float')
# create a new array weights with the same dimension distances and fill
# the array with 0 element.
    for i in range(distances.shape[0]): # for each prediction:
        if distances[i, 0] >= 100: # if the smaller distance is greather than 100,
                                   # consider the nearest neighbor's weight as 1
                                   # and the neighbor weights will stay zero
            weights[i, 0] = 1
                                   # than continue to the next prediction
            continue

        for j in range(distances.shape[1]): # aply the weight function for each distance

            if (distances[i, j] >= 100):
                continue

            weights[i, j] = 1 - distances[i, j]/100

    return weights

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

knn_out: KNeighborsRegressor = KNeighborsRegressor(n_neighbors=10,weights='uniform',n_jobs=1).fit(X, y)

# pickle.dump(knn_out, open("avaliacao_chute_knn.sav", 'wb'))
joblib.dump(knn_out, "avaliacao_chute_knn.sav")

x_axis: nparray = range(1, 50)
score_train: nparray = []
score_test: nparray = []

for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=.2, random_state=i)

    knn: KNeighborsRegressor = KNeighborsRegressor(n_neighbors=10, weights=customized_weights ).fit(X_train, y_train)
    # knn: RadiusNeighborsRegressor = RadiusNeighborsRegressor(radius=5, weights='uniform').fit(X_train, y_train)

    score_test.append(knn.score(X_test, y_test))
    score_train.append(knn.score(X_train, y_train))

pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.xlabel('random_state')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()

pyplot.show()
