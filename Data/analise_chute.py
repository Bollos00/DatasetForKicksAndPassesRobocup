import numpy
import numpy.ma
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

plt.style.use('dark_background')

dataA = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/CHUTE/2019-07-03_14-09_ATA_ER-Force-vs-DEF_TIGERs_Mannheim2020-06-22_Chute.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataB = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/CHUTE/2019-07-04_05-37_ATA_ER-Force-vs-DEF_RoboTeam_Twente2020-06-23_Chute.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataC = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/CHUTE/2019-07-05_01-51_ATA_ER-Force-vs-DEF_RoboDragons2020-06-23_Chute.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataD = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/CHUTE/2019-07-05_09-11_ATA_ER-Force-vs-DEF_ZJUNlict2020-06-23_Chute.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataE = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/CHUTE/2019-07-05_11-51_ATA_ER-Force-vs-DEF_OP-AmP_Chute.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataF = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/CHUTE/2019-07-06_04-54_ATA_ER-Force-vs-DEF_TIGERs_Mannheim2020-06-24_Chute.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataG = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/CHUTE/2019-07-06_11-32_ATA_ER-Force-vs-DEF_ZJUNlict2020-06-24_Chute.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataH = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/CHUTE/2019-07-07_05-26_ATA_ER-Force-vs-DEF_MRL2020-06-24_Chute.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataI = numpy.genfromtxt(
    "/home/robofei/Documents/ArquivosAnalise/ER_FORCE/ATA/CHUTE/2019-07-07_06-34_ATA_ER-Force-vs-DEF_ZJUNlict2020-06-24_Chute.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

array_passe = numpy.concatenate((dataA,
                                 dataB,
                                 dataC,
                                 dataD,
                                 dataE,
                                 dataF,
                                 dataG,
                                 dataH,
                                 dataI))

for i in array_passe:
    print(i)


y = array_passe[:, 0]
X = array_passe[:, [1, 2, 3 ]]

# plt.plot(X[:,7], y, 'o')
# plt.xlabel("Parameter")
# plt.ylabel("Evaluation")
# plt.show()


for i in range(0, 20):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.2,
                                                        random_state=i)

    lr = LinearRegression().fit(X_train, y_train)
    knn = KNeighborsRegressor(n_neighbors=12).fit(X_train, y_train)
    ridge = Ridge(alpha=300000).fit(X_train, y_train)
    lasso = Lasso(alpha = 5e2).fit(X_train, y_train)

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
    print("Training set score for Lasso Regression model: {:.2f}".format(ridge.score(X_train, y_train)))
    print("Test set score for Lasso Regression model: {:.2f}".format(ridge.score(X_test, y_test)))

    # z = numpy.polyfit(y_test, lr.predict(X_test), 1)
    # p = numpy.poly1d(z)
    # plt.plot(X, p(X),'-' )
    #
    # plt.plot(y_test, lr.predict(X_test), 'ro')
    plt.plot(ridge.coef_, 'ro', label='Ridge')
    plt.plot(lasso.coef_, 'gd', label='Lasso')
    plt.grid()
    plt.show()
