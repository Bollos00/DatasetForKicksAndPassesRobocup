import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import time
import analise_auxiliar
from typing import List

array_passe: numpy.ndarray = analise_auxiliar.get_array_from_pattern("ALL/*Passe.csv")

y: numpy.ndarray = array_passe[:, 0]
X: numpy.ndarray = array_passe[:, [1, 2, 3,  4, 4, 6, 7, 8]]

lr_out: LinearRegression = LinearRegression().fit(X, y)

joblib.dump(lr_out, "models/avaliacao_passe_lr.sav")


# for n in range(0,8):
#     z = numpy.polyfit(X[:, n], y, 2)
#     p = numpy.poly1d(z)
#     pyplot.plot(X[:, n], p(X[:, n]), 'go')

#     pyplot.plot(X[:, n], y, 'ro')

#     pyplot.show()

x_axis: List[int] = list(range(1, 200))
score_train: List[float] = []
score_test: List[float] = []

start: float = time.time()

for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(X,
                                                          y,
                                                          test_size=.2,
                                                          random_state=i)

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

    lr: LinearRegression = LinearRegression().fit(X_train, y_train)
    # kernel: KernelRidge = KernelRidge(alpha=i*1e3).fit(X_train, y_train)

    score_test.append(lr.score(X_test, y_test))
    score_train.append(lr.score(X_train, y_train))

    # print(lr.coef_)

    # z = numpy.polyfit(y, lasso.predict(X), 1)
    # p = numpy.poly1d(z)
    # pyplot.plot(X, p(X),'-' )
    #
    # pyplot.plot(y_test, lasso.predict(X_test), 'ro')
    # pyplot.show()


end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

analise_auxiliar.plot_results(x_axis, score_test, score_train)
