import analise_auxiliar
import numpy
from matplotlib import pyplot


def plot_data_analise(array: numpy.ndarray,
                      param_name: str):

    z = numpy.polyfit(array[:, 1], array[:, 0], 5)
    p = numpy.poly1d(z)
    pyplot.plot(range(25, 226), p(range(25, 226)), "r--")

    pesos: numpy.ndarray = numpy.full((5, 5), 0, dtype=numpy.uint8)

    for i in range(5):
        for j in range(5):
            generator = (
                a for a in array if (
                    a[1] in range(i*50, (i+1)*50 + 1))
                and (a[0] in range(j*50, (j+1)*50 + 1)
                     )
                )
            # for k in generator:
            #     print(k)
            # print("({1}, {0})".format(i, j), '\n')
            #
            # generator = (
            #     a for a in array if (
            #         a[1] in range(i*50, (i+1)*50 + 1))
            #     and (a[0] in range(j*50, (j+1)*50 + 1)
            #          )
            #     )
            pesos[i, j] = sum(1 for k in generator)
    print(pesos)
    for i in range(5):
        for j in range(5):
            pyplot.scatter(25 + 50*i,
                           25 + 50*j,
                           s=pesos[i, j]*100,
                           c="#00ff00")

    pyplot.xlabel(param_name)
    pyplot.ylabel("Avaliação")
    pyplot.axis([0, 251, 0, 251])

    pyplot.show()
