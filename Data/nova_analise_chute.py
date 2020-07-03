from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from matplotlib import pyplot

ndarray = numpy.array
pyplot.style.use('dark_background')

file_names = glob("/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/*Chute.csv")

array_chute: ndarray = []


for f in file_names:
    array_chute.append(
        numpy.genfromtxt(
            f,
            dtype=int,
            delimiter=";",
            skip_header=1))

array_chute = numpy.concatenate(array_chute)

y: ndarray = array_chute[:, 0]
X: ndarray = array_chute[:, [1, 2, 3]]

x_axis: ndarray = range(1, 50)
score_train: ndarray = []
score_test: ndarray = []

for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(X,
                                                          y,
                                                          test_size=.2,
                                                          random_state=1)

    # ridge: Ridge = Ridge(alpha=i*1e4).fit(X_train, y_train)
    # print("Training set score for {}-Ridge Regression model: {:.2f}".format(i,ridge.score(X_train, y_train)))
    # print("Test set score for {}-Ridge Regression model: {:.2f}".format(i,ridge.score(X_test, y_test)))
    # print('\n')
    # print("Parameters: ", ridge.coef_)
    # print('\n')

    # lasso: Lasso = Lasso(alpha=i).fit(X_train, y_train)
    # print("Training set score for {}-Lasso Regression model: {:.2f}".format(i, lasso.score(X_train, y_train)))
    # print("Test set score for {}-Lasso Regression model: {:.2f}".format(i, lasso.score(X_test, y_test)))
    # print('\n')
    # print("Parameters: ", lasso.coef_)
    # print('\n')
    #
    # score_test.append(lasso.score(X_test, y_test))
    # score_train.append(lasso.score(X_train, y_train))

    knn: KNeighborsRegressor = KNeighborsRegressor(n_neighbors=i).fit(X_train, y_train)

    # print("Training set score for {}-Nearest Neighbor model: {:.2f}".format(i,knn.score(X_train, y_train)))
    # print("Test set score for {}-Nearest Neighbor model: {:.2f}".format(i,knn.score(X_test, y_test)))
    # print('\n')
    score_test.append(knn.score(X_test, y_test))
    score_train.append(knn.score(X_train, y_train))

pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.xlabel('nieghbors')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")

pyplot.show()