from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import pickle

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

lr_out: LinearRegression = LinearRegression().fit(X, y)

pickle.dump(lr_out, open("avaliacao_passe_lr.sav", 'wb'))


# for n in range(0,8):
#     z = numpy.polyfit(X[:, n], y, 2)
#     p = numpy.poly1d(z)
#     pyplot.plot(X[:, n], p(X[:, n]), 'go')

#     pyplot.plot(X[:, n], y, 'ro')

#     pyplot.show()

x_axis: nparray = range(1, 100)
score_train: nparray = []
score_test: nparray = []


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


pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.xlabel('random_state')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()

pyplot.show()
