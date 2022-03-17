import numpy as np

from lmfit import Parameters, Model
from dataclasses import dataclass, asdict, field
from tools.func_gen import *

Fit_Models = Fit_Models()


@dataclass(order=True)
class Spectra:
    # Todo: Test serialization to json, and implement dump/load functionality if necessary
    # Dataclass to store individual spectra data, for easy access
    # Compiling several spectra into on object is done by the Measurement class
    sort_index: int = field(init=False, repr=False)

    x_data: np.array
    y_data: np.array
    function_str: str
    model_names: List[str]
    angle: float
    spectra_index: int
    model: Model = field(init=False, repr=True)
    parameter: Parameters = field(init=False, repr=True)
    fitted: bool = False
    dropped: bool = False

    def __post_init__(self):
        model = Fit_Models.getModelFunc(self.model_names, self.spectra_index)
        exec(model[0], locals())
        self.function_str = model[0]
        self.model = Model(locals()[model[1]])
        self.parameter = self.model.make_params()

        self.sort_index = self.spectra_index

    def __len__(self):
        if self.x_data.shape == self.y_data.shape:
            return self.x_data.shape[0]
        else:
            raise ValueError("x_data and y_data are not in same shape")

    def asdict(self):
        return asdict(self)


if __name__ == '__main__':
    measurement = []
    for i in range(0, 360):
        measurement.append(Spectra(np.arange(0, 4096, 1), np.arange(0, 4096, 1),
                                   'function_str', ['Lorentz 1', 'Dyson 2', 'Linear 3'], float(i), int(i)))
    # print(min(measurement).angle)
