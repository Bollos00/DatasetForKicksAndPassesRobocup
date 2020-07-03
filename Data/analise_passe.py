import matplotlib.pyplot as plt
import numpy.ma
import numpy.polynomial
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from typing import Optional
from typing import List

ndarray = numpy.array

plt.style.use('dark_background')

dataA: ndarray = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/PASSE/2019-07-03_14-09_ATA_ER-Force-vs-DEF_TIGERs_Mannheim2020-06-22_Passe.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataB: ndarray = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/PASSE/2019-07-04_05-37_ATA_ER-Force-vs-DEF_RoboTeam_Twente2020-06-23_Passe.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataC: ndarray = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/PASSE/2019-07-05_01-51_ATA_ER-Force-vs-DEF_RoboDragons2020-06-23_Passe.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataD: ndarray = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/PASSE/2019-07-05_09-11_ATA_ER-Force-vs-DEF_ZJUNlict2020-06-23_Passe.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataE: ndarray = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/PASSE/2019-07-05_11-51_ATA_ER-Force-vs-DEF_OP-AmP_Passe.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataF: ndarray = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/PASSE/2019-07-06_04-54_ATA_ER-Force-vs-DEF_TIGERs_Mannheim2020-06-24_Passe.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataG: ndarray = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/PASSE/2019-07-06_11-32_ATA_ER-Force-vs-DEF_ZJUNlict2020-06-_Passe.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataH: ndarray = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/PASSE/2019-07-07_05-26_ATA_ER-Force-vs-DEF_MRL2020-06-24_Passe.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataI: ndarray = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/PASSE/2019-07-07_06-34_ATA_ER-Force-vs-DEF_ZJUNlict2020-06-24_Passe.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

array_passe: ndarray = numpy.concatenate( (dataA,
                                           dataB,
                                           dataC,
                                           dataD,
                                           dataE,
                                           dataF,
                                           dataG,
                                           dataH,
                                           dataI) )

# for i in array_passe:
#     print(i)


y: ndarray = array_passe[:, 0]
X: ndarray = array_passe[:, [1, 4]]

# plt.plot(X[:,7], y, 'o')
# plt.xlabel("Parameter")
# plt.ylabel("Evaluation")
# plt.show()


for i in range(0, 20):

    [X_train, X_test, y_train, y_test] = train_test_split(X,
                                                          y,
                                                          test_size=.2,
                                                          random_state=i)

    lr: LinearRegression = LinearRegression().fit(X_train, y_train)
    knn: KNeighborsRegressor = KNeighborsRegressor(n_neighbors=11).fit(X_train, y_train)
    ridge: Ridge = Ridge(alpha=1e4).fit(X_train, y_train)
    lasso: Lasso = Lasso(alpha=2e2).fit(X_train, y_train)

    print('\n',i)
    print("Training set score for Linear Regression model: {:.2f}".format(lr.score(X_train, y_train)))
    print("Test set score for Linear Regression model: {:.2f}".format(lr.score(X_test, y_test)))
    print('\n')
    print("Training set score for Nearest Neighbor model: {:.2f}".format(knn.score(X_train, y_train)))
    print("Test set score for Nearest Neighbor model: {:.2f}".format(knn.score(X_test, y_test)))
    print('\n')
    print("Training set score for Ridge Regression model: {:.2f}".format(ridge.score(X_train, y_train)))
    print("Test set score for Ridge Regression model: {:.2f}".format(ridge.score(X_test, y_test)))
    print('\n')
    print("Training set score for Lasso Regression model: {:.2f}".format(lasso.score(X_train, y_train)))
    print("Test set score for Lasso Regression model: {:.2f}".format(lasso.score(X_test, y_test)))

    z = numpy.polyfit(y, knn.predict(X), 1)
    p = numpy.poly1d(z)
    plt.plot(X, p(X),'-' )

    plt.plot(y, knn.predict(X), 'ro')
    # plt.xlabel("Real")
    # plt.ylabel("Prediction")
    plt.show()
