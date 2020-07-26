from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from matplotlib import pyplot
import pickle
import joblib
import time

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

x_axis: nparray = range(1, 200, 1)
score_train: nparray = []
score_test: nparray = []

start: float = time.time()
for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=i
        )

    kr: KernelRidge = KernelRidge(
        alpha=1000,
        kernel='polynomial',
        gamma=9,
        degree=2,
        coef0=1,
        kernel_params=None).fit(X_train, y_train)

    score_test.append(kr.score(X_test, y_test))
    score_train.append(kr.score(X_train, y_train))

end: float = time.time()

print("Score test: ", numpy.mean(score_test))
print("Score train: ", numpy.mean(score_train))
print("Time of operation: {} ms".format(
    (end-start)*1e3/(numpy.size(x_axis)*numpy.size(y)))
      )

pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.xlabel('???')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()
pyplot.axis([1, 200, 0, 1])
pyplot.show()
