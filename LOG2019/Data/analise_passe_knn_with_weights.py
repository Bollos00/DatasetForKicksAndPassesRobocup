from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot
import joblib

nparray = numpy.array
pyplot.style.use('dark_background')


def customized_weights(distances: nparray) -> nparray:
    weights: nparray = nparray(numpy.full(distances.shape, 0), dtype='float')
    # create a new array weights with the same dimension distances and fill
    # the array with 0 element.
    for i in range(distances.shape[0]):  # for each prediction:
        if distances[i, 0] >= 200:  # if the smaller distance is greather than 100,
            # consider the nearest neighbor's weight as 1
            # and the neighbor weights will stay zero
            weights[i, 0] = 1
            # than continue to the next prediction
            continue

        for j in range(distances.shape[1]):  # aply the weight function for each distance
            # print(distances[i, j])

            if (distances[i, j] >= 200):
                break

            weights[i, j] = 1 - distances[i, j]/200

    return weights


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

knn_out: KNeighborsRegressor = KNeighborsRegressor(n_neighbors=20,
                                                   weights=customized_weights,
                                                   n_jobs=1).fit(X, y)

joblib.dump(knn_out, "models/avaliacao_passe_knn_with_weights.sav")

x_axis: nparray = range(1, 50)
score_train: nparray = []
score_test: nparray = []

for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(X,
                                                          y,
                                                          test_size=.2,
                                                          random_state=i)

    knn: KNeighborsRegressor = KNeighborsRegressor(
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=None
        ).fit(X_train, y_train)
    # knn: RadiusNeighborsRegressor = RadiusNeighborsRegressor(radius=500, weights=customized_weights).fit(X_train, y_train)

    score_test.append(knn.score(X_test, y_test))
    score_train.append(knn.score(X_train, y_train))


pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.xlabel('???')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()

pyplot.show()
