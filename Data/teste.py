import matplotlib.pyplot as plt
import numpy.ma
import numpy.polynomial
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

plt.style.use('dark_background')

dataA = numpy.genfromtxt(
    "/home/robofei/Documents/testeDatasAntigas-ER_FORCE/2019-07-03_14-09_ATA_ER-Force-vs-DEF_TIGERs_Mannheim.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataB = numpy.genfromtxt(
    "/home/robofei/Documents/testeDatasAntigas-ER_FORCE/2019-07-04_05-37_ATA_ER-Force-vs-DEF_RoboTeam_Twente.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataC = numpy.genfromtxt(
    "/home/robofei/Documents/testeDatasAntigas-ER_FORCE/2019-07-05_01-51_ATA_ER-Force-vs-DEF_RoboDragons2020-06-20.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataD = numpy.genfromtxt(
    "/home/robofei/Documents/testeDatasAntigas-ER_FORCE/2019-07-05_09-11_ATA_ER-Force-vs-DEF_ZJUNlict2020-06-20.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataE = numpy.genfromtxt(
    "/home/robofei/Documents/testeDatasAntigas-ER_FORCE/2019-07-05_11-51_ATA_ER-Force-vs-DEF_OP-AmP.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataF = numpy.genfromtxt(
    "/home/robofei/Documents/testeDatasAntigas-ER_FORCE/2019-07-06_04-54_ATA_ER-Force-vs-DEF_TIGERs_Mannheim.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataG = numpy.genfromtxt(
    "/home/robofei/Documents/testeDatasAntigas-ER_FORCE/2019-07-06_11-32_ATA_ER-Force-vs-DEF_ZJUNlict2020-06-20.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataH = numpy.genfromtxt(
    "/home/robofei/Documents/testeDatasAntigas-ER_FORCE/2019-07-07_05-26_ATA_ER-Force-vs-DEF_MRL2020-06-20.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

dataI = numpy.genfromtxt(
    "/home/robofei/Documents/testeDatasAntigas-ER_FORCE/2019-07-07_06-34_ATA_ER-Force-vs-DEF_ZJUNlict2020-06-20.csv",
    dtype=int,
    delimiter=";",
    skip_header=1)

numpy_array = numpy.concatenate( (dataA,
                                  dataB,
                                  dataC,
                                  dataD,
                                  dataE,
                                  dataF,
                                  dataG,
                                  dataH,
                                  dataI) )

# for i in numpy_array:
#     print (i)

arrray_passe = numpy_array[:,[5,6,7,8,9,10,11,12,13]]


arrray_passe = numpy.ma.masked_array(arrray_passe, arrray_passe == -1)



y = arrray_passe[:,0]
X = arrray_passe[:,[1,2,3,4,5,6,7,8]]


for i in range(0, 20):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.2,
                                                        random_state=i)

    lr = LinearRegression().fit(X_train, y_train)
    knn = KNeighborsRegressor(n_neighbors=11).fit(X_train, y_train)
    ridge = Ridge(alpha=1e4).fit(X_train, y_train)
    lasso = Lasso(alpha=2e2).fit(X_train, y_train)

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

    # z = numpy.polyfit(y, knn.predict(X), 1)
    # p = numpy.poly1d(z)
    # plt.plot(X, p(X),'-' )
    #
    plt.plot(y, knn.predict(X), 'ro')
    plt.xlabel("Real")
    plt.ylabel("Prediction")
    plt.show()
