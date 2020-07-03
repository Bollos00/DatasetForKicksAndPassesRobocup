from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from matplotlib import pyplot

ndarray = numpy.array
pyplot.style.use('dark_background')

file_names = glob("/home/robofei/Documents/ArquivosAnalise/TIGERS_MANHEIM/ATA/*Passe.csv")

array_passe: ndarray = []


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

y: ndarray = array_passe[:, 0]
X: ndarray = array_passe[:, [1, 2, 3,  4, 4, 6, 7, 8]]

x_axis: ndarray = range(1, 100)
score_train: ndarray = []
score_test: ndarray = []

for n in range(0,8):
    z = numpy.polyfit(X[:, n], y, 2)
    p = numpy.poly1d(z)
    pyplot.plot(X[:, n], p(X[:, n]), 'go')

    pyplot.plot(X[:, n], y, 'ro')

    pyplot.get_current_fig_manager().full_screen_toggle()
    pyplot.show()


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

    # print("Training set score for {}-Lasso Regression model: {:.2f}".format(i, lasso.score(X_train, y_train)))
    # print("Test set score for {}-Lasso Regression model: {:.2f}".format(i, lasso.score(X_test, y_test)))
    # print('\n')
    # print("Parameters: ", lasso.coef_)
    # print('\n')

    knn: KNeighborsRegressor = KNeighborsRegressor(n_neighbors=i).fit(X_train, y_train)
    # kernel: KernelRidge = KernelRidge(alpha=i*1e3).fit(X_train, y_train)

    score_test.append(knn.score(X_test, y_test))
    score_train.append(knn.score(X_train, y_train))

    # z = numpy.polyfit(y, lasso.predict(X), 1)
    # p = numpy.poly1d(z)
    # pyplot.plot(X, p(X),'-' )
    #
    # pyplot.plot(y_test, lasso.predict(X_test), 'ro')
    # pyplot.show()


pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.xlabel('nieghbors')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()

pyplot.show()