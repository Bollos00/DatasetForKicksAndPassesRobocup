import numpy

class Weight(object):
    @staticmethod
    def weight(dist):
        raise NotImplementedError()
    
class WeightLinear(Weight):
    MAXIMUM_DISTANCE = 100
    
    @staticmethod
    def set_maximum_distance(new_max_dist):
        WeightLinear.MAXIMUM_DISTANCE = new_max_dist
    
    @staticmethod
    def weight(dist):
        v = 1 - dist/WeightLinear.MAXIMUM_DISTANCE
        if numpy.isnan(v) or v < 1e-6:
            v = 1e-6
        return v

class WeightGaussian(Weight):
    DEVIATION = 100
    A = 1/(DEVIATION*2*numpy.pi)
    B = -1/(2*(DEVIATION**2))
    
    @staticmethod
    def set_deviation(new_var):
        WeightGaussian.DEVIATION = new_var
        WeightGaussian.A = 1/(WeightGaussian.DEVIATION*2*numpy.pi)
        WeightGaussian.B = -1/(2*(WeightGaussian.DEVIATION**2))
                
    @staticmethod
    def weight(dist):
        v = WeightGaussian.A*numpy.exp(WeightGaussian.B*(dist**2))
        if numpy.isnan(v) or v < 1e-6:
            v = 1e-6
        return v

class WeightParabolic(Weight):
    MAXIMUM_DISTANCE_SQUARE = 100**2

    @staticmethod
    def set_maximum_distance(new_max_dist):
        WeightParabolic.MAXIMUM_DISTANCE_SQUARE = new_max_dist**2

    @staticmethod
    def weight(dist):
        v = 0.75*(1 - (dist*dist)/WeightParabolic.MAXIMUM_DISTANCE_SQUARE)
        if numpy.isnan(v) or v < 1e-6:
            v = 1e-6
        return v

class Weights:
    WEIGHT_FUNC = WeightLinear.weight
    
    @staticmethod
    def set_weight_func(new_func):
        Weights.WEIGHT_FUNC = new_func
    
    @staticmethod
    def weights(distances_weights):
        for i, var in enumerate(distances_weights):  # for each prediction:
            if numpy.size(var) == 0:
                continue

            for j, dist in enumerate(var):  # apply the weight function for each distanc
                distances_weights[i][j] = Weights.WEIGHT_FUNC(dist)

        return distances_weights
