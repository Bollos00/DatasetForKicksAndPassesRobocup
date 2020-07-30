import numpy
import analise_auxiliar
import analise_data_auxiliar

array_passe: numpy.ndarray = analise_auxiliar.get_array_from_pattern("ALL/*Passe.csv")

angulo_livre_passe: numpy.ndarray = array_passe[:, [0, 1]]  # X is 1 and Y is 0
distancia_passe: numpy.ndarray = array_passe[:, [0, 2]]
liberdade_marcacao_receptor: numpy.ndarray = array_passe[:, [0, 3]]
angulo_redirect: numpy.ndarray = array_passe[:, [0, 4]]
angulo_livre_chute_receptor: numpy.ndarray = array_passe[:, [0, 5]]
distancia_receptor_gol: numpy.ndarray = array_passe[:, [0, 6]]
liberdade_marcacao_passador: numpy.ndarray = array_passe[:, [0, 7]]
delta_xis: numpy.ndarray = array_passe[:, [0, 8]]

analise_data_auxiliar.plot_data_analise(angulo_livre_passe, "Ângulo livre passe")
analise_data_auxiliar.plot_data_analise(distancia_passe, "Distância passe")
analise_data_auxiliar.plot_data_analise(liberdade_marcacao_receptor, "Liberdade marcação receptor")
analise_data_auxiliar.plot_data_analise(angulo_redirect, "Ângulo redirect")
analise_data_auxiliar.plot_data_analise(angulo_livre_chute_receptor, "ângulo livre chute receptor")
analise_data_auxiliar.plot_data_analise(distancia_receptor_gol, "Distância receptor gol")
analise_data_auxiliar.plot_data_analise(liberdade_marcacao_passador, "Liberdade marcação passador")
analise_data_auxiliar.plot_data_analise(delta_xis, "Delta xis")
