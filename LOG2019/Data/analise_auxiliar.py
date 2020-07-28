from glob import glob
import numpy


def getArrayFromPattern(pattern):

    array = []

    for f in glob("../{}".format(pattern)):
        array.append(
            numpy.genfromtxt(
                f,
                dtype=numpy.uint8,
                delimiter=";",
                skip_header=1
            )
        )

    return numpy.concatenate(array)
