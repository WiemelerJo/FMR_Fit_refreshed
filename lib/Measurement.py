import numpy as np

from Spectra import Spectra
from typing import List, Union


class Measurement(list):
    # Advanced List to store, load, save and maybe modify Spectra objects
    # Callable, by Measurement(int, int, ..) of unspecific amount of int
    # or Measurement([int, int, int, ..]) of list int
    # __call__ will sort self before spitting out desired spectra

    def __init__(self):
        super(Measurement, self).__init__()

    def loadFitFile(self):
        # Load previously fitted and saved file
        self.clear()
        print("Programier mich!")

    def loadMeasurement(self):
        # Load new measurement from file
        self.clear()
        print("Programier mich!")

    def add(self, spectra: Spectra):
        # Add spectra to existing Measurement object
        index = spectra.spectra_index
        if not self.isIndexed(spectra):
            self.insert(index, spectra)
        else:
            raise IndexError("Spectra with same Index already in use!\nSpectra:" + str(index))

    def append(self, spectra: Spectra) -> None:
        self.add(spectra)

    def delete(self, item: Union[Spectra, int]):
        # item:int deletes list index not spectra index!

        if item == type(int):
            self.pop(item)
        else:
            for spec in self:
                if spec == item:
                    self.remove(spec)

    def isIndexed(self, spectra: Spectra):
        # Look up if spectra with index spectra.spectra_index is already indexed in self
        # index = spectra.spectra_index
        try:
            indexed_spec = self[spectra.spectra_index]
            if indexed_spec == spectra:
                return True
            return False
        except IndexError:
            return False

    def __add__(self, other: Spectra):
        if isinstance(other, Spectra):
            self.add(other)
        else:
            raise NotImplementedError('Operation __add__ not implemented for object type: {}'.format(type(other)))

    def __str__(self) -> str:
        # return str([i.spectra_index for i in self])
        length = len(self)
        if length == 0:
            return "No Spectra indexed!"
        return str(length) + " Spectra indexed!"

    def __call__(self, *args, **kwargs) -> List[Spectra]:
        if args:
            self.sort()
            rtn_stmnt = []
            for arg in args:
                if isinstance(arg, int):
                    rtn_stmnt.append(self[arg])
                elif isinstance(arg, list):
                    for ar in arg:
                        rtn_stmnt.append(self[ar])
            return rtn_stmnt
        elif kwargs:
            raise NotImplementedError("Kwargs not Implemented")
        print("Supply argument of type Measurement(int, int, int, ...) or Measurement([int, int, int])!")


if __name__ == '__main__':
    meas1 = Measurement()
    test_spec = Spectra(np.arange(0, 4096, 1), np.arange(0, 4096, 1),
                        'function_str', ['Lorentz 1', 'Dyson 2', 'Linear 3'], float(22), int(2000))

    for i in range(0, 360):
        meas1.append(Spectra(np.arange(0, 4096, 1), np.arange(0, 4096, 1),
                             'function_str', ['Lorentz 1', 'Dyson 2', 'Linear 3'], float(i), int(i)))

    # random.shuffle(meas1)
    # print(meas1(1 ,2, 3))
    # print(max(meas1).spectra_index)
    # print(meas1.isIndexed(test_spec))
