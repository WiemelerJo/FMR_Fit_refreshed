class Model_info():
    # Will read models.txt and gen model infos: Name, Parameters, Bounds, init_vals
    def __init__(self):
        self.__read_models()

    def __call__(self, name:str, *args, **kwargs):
        return self._model[name]

    def __read_models(self):
        # Read from .txt
        self._model = {'Lorentz': {'Parameter_names': ['dB', 'R', 'A'],
                                   'Bounds': [(0.0, 1.5), (0.0, 2.4), (0.0, 15000.0)],
                                   'init_vals': [0.005, 0.09, 15.0],
                                   'dB': {'value': 0.005, 'min': 0.0, 'max': 1.5, 'step_size': 0.001},
                                   'R': {'value': 0.09, 'min': 0.0, 'max': 2.4, 'step_size': 0.0001},
                                   'A': {'value': 15.0, 'min': 0.0, 'max': 15000.0, 'step_size': 0.1}},

                       'Dyson': {'Parameter_names': ['alpha', 'dB', 'R', 'A'],
                                 'Bounds': [(0.0, 1.0), (0.0, 1.5), (0.0, 2.4), (0.0, 15000.0)],
                                 'init_vals': [0.000001, 0.005, 0.09, 15.0],
                                 'alpha': {'value': 0.000001, 'min': 0.0, 'max': 1.0, 'step_size': 0.0001},
                                 'dB': {'value': 0.005, 'min': 0.0, 'max': 1.5, 'step_size': 0.0001},
                                 'R': {'value': 0.09, 'min': 0.0, 'max': 2.4, 'step_size': 0.0001},
                                 'A': {'value': 15.0, 'min': 0.0, 'max': 15000.0, 'step_size': 0.1}
                                 }}

    def function(self):
        return