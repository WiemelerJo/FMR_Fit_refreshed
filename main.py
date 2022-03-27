import sys
import pandas as pd
import numpy as np
import math as m
import pyqtgraph as pg
import json

from PyQt5.QtWidgets import QMainWindow, QApplication,QFileDialog, QCheckBox, QLabel, QMessageBox, QShortcut,\
    QTreeWidgetItem, QDoubleSpinBox
from PyQt5.QtCore import QThread, pyqtSignal, QSignalBlocker
from PyQt5.Qt import Qt, QUrl, QDesktopServices
from PyQt5.QtGui import QKeySequence
from PyQt5.uic import loadUi

from lib.CustomWidgets import *
from lib.Fitting import *
from lib.Measurement import Measurement
#from lib.Spectra import Spectra

from tools.func_gen import Fit_Models

from lmfit import Model, Parameters, Parameter

from typing import List, Set, Dict, Tuple, Optional

#Todo: Implement to be able to have different configuration of models per spectra in a dependence

class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("lib/Fit-programm.ui", self)

        # File Menu
        self.actionOpen.triggered.connect(self.openFileDialog)
        self.actionSave.triggered.connect(self.saveFileDialog)
        #self.actionExit.triggered.connect(self.Exit)

        self.Button_add_function.clicked.connect(self.add_func)
        self.Button_remove_function.clicked.connect(self.remove_func)

        self.pushButton.clicked.connect(self.test)
        self.Button_Fit.clicked.connect(self.makeFit)
        self.Button_dropped_points.clicked.connect(self.dropCurrentSpectra)
        self.Dropped_points_edit.editingFinished.connect(self.setExceptions)

        self.func_number = 1
        self.func_removed = []
        self.exceptions = []

        self.pltView = self.Plot_Indi_View
        self.pltViewRange = self.Plot_Indi_View

        self.populate_combobox()


    def test(self, *args, **kwargs):
        #self.clearTree()
        self.saveParameter()
        #self.loadParameter()
        #print(self.paramsLUT.get(self.i).get('models'))
        #self.setValuesForTree()
        #self.getValuesFromTree()
        #self.plotFitData()
        #print(self.Models.MODELS.get('Lorentz'))
        #print(self.paramsLUT.get(self.i).get('params'))
        #for name in self.paramsLUT.get(self.i).get('params'):
        #    print(name)
        #print(self.paramsLUT.get(self.i).get('params').get('A1'))

    def dropCurrentSpectra(self):
        excep = self.getExceptions() # Get current exceptions
        #excep = self.exceptions
        excep.append(self.i) # Append current spectra index
        text = ','.join(map(str, excep))
        self.Dropped_points_edit.setText(str(text))
        self.setExceptions()

    def setExceptions(self):
        self.exceptions = self.getExceptions()

    def getExceptions(self) -> list:
        # takes the string input from the exception text field inside the GUI and seperates the entries into a list
        x = str(self.Dropped_points_edit.text())  # saves everything that is in the editLine as x
        exceptions = []
        if x == '':
             return exceptions# if y is empty excep should also be empty

        exceptions = list(map(int, x.split(',')))  # else set excep as everything that is standing in editLine
        return exceptions

    def changeSpectra(self, spectra: int):
        # Change measured spectra
        # For now this is equal to an angle change
        # In future this could also be Frequency
        self.i = spectra
        self.plot()
        self.setValuesForTree()

        #self.plotFitData()

    def populate_combobox(self):
        # Get all possible function from Models.txt, get their names and add them as items in combobox
        self.Models = Fit_Models()
        self.comboBox_fit_model.addItems(list(self.Models.MODELS.keys()))

    def resetCurrentParamLUTentry(self):
        try:
            self.Measurement[self.i].parameter = None
        except AttributeError:
            self.openFileDialog()

    def add_func(self):
        self.resetCurrentParamLUTentry()
        # Get func name from combobox and add it to self.funcs
        # Prefere removed index over new index
        name = self.comboBox_fit_model.currentText()
        #self.funcs.append(name)
        number = self.func_number
        self.func_number += 1
        if self.func_removed:
            self.func_number -= 1
            self.func_removed.sort()
            number = self.func_removed[0]
            self.func_removed.pop(0)

        self.populate_tree(name, number)

    def remove_func(self):
        self.resetCurrentParamLUTentry()
        # Get selected Item/Function to remove
        # identify its index and add it to func_remove list
        # this removed index will then be prefered while adding
        tree = self.Parameter_tree
        root = tree.invisibleRootItem()
        for item in tree.selectedItems():
            name = item.text(0)
            try:
                index = name.split(" ")[1]
            except IndexError:
                name = item.parent().text(0)
                index = name.split(" ")[1]
            self.func_removed.append(index)
            root.removeChild(item)
            root.removeChild(item.parent())
        self.getValuesFromTree()
        self.plotFitData()

    def populate_tree(self, func_name: str, number: int):
        # Create treewidgetitem with name: func_name {number}
        # Add every Parameter as Child, with 3 QDoubleSpinbox for Value, Min, Max
        # Make every Branch/(Parent/Child) checkable to check fit usage

        tree = self.Parameter_tree

        Model = self.Models.MODELS.get(func_name)

        parent = QTreeWidgetItem(tree)
        parent.setText(0, func_name + ' {}'.format(number))
        parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

        for index, param in enumerate(Model['args']):
            child = QTreeWidgetItem(parent)
            child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
            child.setText(0, param)
            child.setCheckState(0, Qt.Checked)

            Stepsize_list = Model.get('stepSize')
            step = Stepsize_list[index]
            steps = [step, 10 * step, 10 * step]

            Bounds = Model.get('bounds')
            min = Bounds[index][0]
            max = Bounds[index][1]

            boundsMin = [min, -2 * abs(max), -2 * abs(max)]
            boundsMax = [max, 2 * max, 2 * max]

            Val_list = Model.get('initVals')
            val = Val_list[index]
            value = [val, min, max]

            signal_list = [self.dbs_value, self.dbs_bound_min, self.dbs_bound_max]

            for col in range(3):
                dbs = QDoubleSpinBox()
                dbs.setDecimals(6)

                dbs.setSingleStep(steps[col])
                dbs.setRange(boundsMin[col], boundsMax[col])
                dbs.setValue(value[col])

                dbs.valueChanged.connect(signal_list[col])

                tree.setItemWidget(child, col + 1, dbs)

        #Refresh plot
        self.getValuesFromTree()
        self.plotFitData()

    def dbs_value(self, *args):
        self.getValuesFromTree()
        self.plotFitData()
        #self.plot()
        #print(args)

    def dbs_bound_min(self, *args, **kwargs):
        print(args, kwargs)

    def dbs_bound_max(self, *args):
        print(args)

    def block_spinbox_signal(self, block:bool, widget:QWidget):
        # Temporarily block every signal from the Spinboxes. There is the Problem,
        # that everytime the value from these spinboxes is changed a signal is emited.
        # The bad thing now is, that there is no differentiation between user_event and machine_event,
        # therefore if the script changes the values to display the next angle,
        # the .connect() function will be called len(params) times, which has a min of 5 times with up to 42 times.

        for n in range(1, 3 + 1):
            dbs = self.Parameter_tree.itemWidget(widget, n)
            dbs.blockSignals(block)

    def setValuesForTree(self):
        root = self.Parameter_tree.invisibleRootItem()
        child_count = root.childCount()
        parameter = self.Measurement[self.i].parameter
        if parameter == None or parameter == False:
            return

        for i in range(child_count):
            item = root.child(i)
            func = item.text(0).split(" ")
            func_name = func[0]
            func_index = func[1]

            for n in range(item.childCount()):
                widget = item.child(n)
                param_name = widget.text(0).replace(" ", "") + func_index

                arg = self.Measurement[self.i].parameter[param_name]

                self.block_spinbox_signal(True, widget)
                dbs_val = self.Parameter_tree.itemWidget(widget, 1)
                dbs_val.setValue(arg.value)
                dbs_min = self.Parameter_tree.itemWidget(widget, 2)
                dbs_min.setValue(arg.min)
                dbs_max = self.Parameter_tree.itemWidget(widget, 3)
                dbs_max.setValue(arg.max)
                self.block_spinbox_signal(False, widget)
        self.plotFitData()

    def getValuesFromTree(self):
        # Reads the spin box values and saves them into spectra.parameter
        # If Spectra self.i hasnt been initiated yet, call self.setupParamaters()
        root = self.Parameter_tree.invisibleRootItem()
        child_count = root.childCount()
        funcNames = []

        #print(child_count)

        # get the names
        for i in range(child_count):
            item = root.child(i)
            name = item.text(0) #.split(" ")[0]
            funcNames.append(name)

        # Check if spectra has been init yet
        if self.Measurement[self.i].parameter == None:
            self.setupParameters(funcNames)

        # get spinbox values
        params = {}
        for i in range(child_count):
            item = root.child(i)
            func = item.text(0).split(" ")
            func_name = func[0]
            func_index = func[1]

            for n in range(item.childCount()):
                widget = item.child(n)
                param_name = widget.text(0).replace(" ", "") + func_index

                dbs_val = self.Parameter_tree.itemWidget(widget, 1)
                dbs_min = self.Parameter_tree.itemWidget(widget, 2)
                dbs_max = self.Parameter_tree.itemWidget(widget, 3)

                params[param_name] = {'value': dbs_val.value(), 'state': bool(widget.checkState(0)),
                                      'min': dbs_min.value(),
                                      'max': dbs_max.value()}

        LUTparams = self.Measurement[self.i].parameter
        for name in LUTparams:
            arg = params.get(name)
            LUTparams[name].value = arg.get("value")
            LUTparams[name].min = arg.get("min")
            LUTparams[name].max = arg.get("max")
            LUTparams[name].vary = arg.get("state")

    def setupParameters(self, names: list):
        # Called for NoneType entry in spectra.parameter
        # Create an lmfit Model using function names present in tree, then make lmfit Params out of this

        func = self.Models.getModelFunc(names, self.i)
        # func[0] is the function string, func[1] is its reference, func[2] is func_args
        exec(func[0], globals()) # Create function

        # with func[1] create lfmit Model()
        fit_model = Model(globals().get(func[1]))
        params = fit_model.make_params()

        self.Measurement[self.i].model = fit_model
        self.Measurement[self.i].model_names = names
        self.Measurement[self.i].parameter = params

    def loadModels(self, names: list):
        func = self.Models.getModelFunc(names, 0)
        # func[0] is the function string, func[1] is its reference, func[2] is func_args
        exec(func[0], globals())  # Create function

        # with func[1] create lfmit Model()
        fit_model = Model(globals().get(func[1]))
        return fit_model

    def openFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home', 'Converted Files (*.txt *.dat *.asc) ;;'
                                                                        ' Raw Files (*.DTA *.par)')

        if fname[0]:
            if fname[0][-3:] == 'DTA' or fname[0][-3:] == 'par':
                fname[0] = self.convert_bruker_to_ascii(fname[0])

            self.Measurement = Measurement()
            self.Measurement.loadMeasurement(fname[0])

            self.setupLoadedData()

            self.plot()

    def saveFileDialog(self):
        self.saveParameter()

    def setupLoadedData(self):
        x_data = self.Measurement[0].x_data

        self.i = 0  # self.i is the variable defining which spectra is plotted
        self.i_min = 0
        self.i_max = len(self.Measurement)

        self.select_datanumber.setValue(self.i)
        self.select_datanumber.setMinimum(self.i_min)
        self.select_datanumber.setMaximum(self.i_max)
        self.select_datanumber.valueChanged.connect(self.changeSpectra)

        H_min = min(x_data)
        H_max = max(x_data)
        self.H_ratio = (len(x_data)) / (H_max - H_min)
        self.H_offset = - self.H_ratio * H_min

        self.Plot_Indi_View.lr.setBounds([H_min, H_max])  # Set Range Boundaries for linearregionitem
        self.select_datanumber.setMaximum(self.i_max)
        self.progressBar.setMaximum(self.i_max - 1)

        self.clearTree()

    def clearTree(self):
        self.Parameter_tree.clear()
        self.func_number = 1

    def plotFitData(self):
        # Plot the fitted data + individual functions
        # This needs to be called for every change
        #self.getValuesFromTree()
        if self.i in self.exceptions:
            return

        spectra = self.Measurement[self.i]
        H_data = spectra.x_data
        Ampl_data = spectra.y_data

        params = spectra.parameter

        # evaluate the fit model with given parameters params then plot
        fitted_data = spectra.model.eval(params=params, B=H_data)

        pen_result = pg.mkPen((255, 0, 0), width=3)
        self.pltFitData.setData(H_data, fitted_data, name='Result', pen=pen_result)
        self.pltFitDataRange.setData(H_data, fitted_data, name='Result', pen=pen_result)

    def plot(self):
        # Plot the Background/Experiment
        # This only needs to be called once, every spectra change
        # First Plot Background, then prepare plot for fit data
        spectra = self.Measurement[self.i]
        H_data = spectra.x_data
        Ampl_data = spectra.y_data

        #j_min, j = self.getFitRegion()
        view = self.pltView
        view_range = self.pltViewRange

        view.plt.clear()  # Delete previous data
        view_range.plt_range.clear() # Delete previous data

        view.plt.plot(H_data, Ampl_data, name='Experiment', pen=(255, 255, 255))  # Plot Experiment data
        view_range.plt_range.plot(H_data, Ampl_data, pen=(255, 255, 255))

        view.plt.addItem(view.lr)  # Plot Linear Range Select Item
        view.plt.addItem(view.label, ignoreBounds=True)

        # Init Objects for Fitted Data Plot
        self.pltFitData = self.pltView.plt.plot()
        self.pltFitDataRange = self.pltViewRange.plt_range.plot()

    def getFitRegion(self):
        # Get the width of this blue moveable region in the main plot
        region = self.pltViewRange.lr.getRegion()
        return int(float(region[0]) * self.H_ratio + self.H_offset),\
               int(float(region[1]) * self.H_ratio + self.H_offset)

    def makeFit(self):
        if self.i in self.exceptions:
            #  If true, self.i is in exeptions and should not be fitted
            return

        # Take current spectra, params file and model entry to make a fit
        self.getValuesFromTree()

        spectra = self.Measurement[self.i]
        params = spectra.parameter

        H_data = spectra.x_data
        Ampl_data = spectra.y_data
        model = spectra.model

        j_min, j = self.getFitRegion()
        result = model.fit(Ampl_data[j_min:j], params, B=H_data[j_min:j])

        spectra.parameter = result.params
        self.setValuesForTree()

    def loadParameter(self):
        raise NotImplementedError
        #data = self.paramsLUT
        name = 'Test123'
        filenameJSON = name + '.json'
        filenameDAT = name + '.dat'
        filepath = ''

        with open(filepath + filenameJSON, 'r') as file:
            self.paramsLoaded = json.load(file)

        self.paramsLUT = {}
        for key, val_loaded in self.paramsLoaded.items():
            val = {}
            Param = Parameters()
            val['Ampl'] = np.array(val_loaded.get('Ampl'))
            val['Field'] = np.array(val_loaded.get('Field'))
            param = val_loaded.get('params')
            if param == None:
                val['params'] = None
                val['models'] = None
                self.paramsLUT[int(key)] = val
                self.setupLoadedData()
                self.plot()
            else:
                val['params'] = Param.loads(param)
                names_raw = val_loaded.get('models') # Is str(Name int)
                model = self.loadModels(names_raw)
                val['models'] = [model, names_raw]
                self.paramsLUT[int(key)] = val
                self.setupLoadedData()
                self.plot()

                names = []
                index = []
                for entry in names_raw:
                    model = entry.split(" ")
                    self.populate_tree(model[0], model[1])


        #self.paramsLUT = {int(k): v for k, v in self.paramsLUT.items()} #Convert index i from str to int


    def saveParameter(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Please select the file to save to", "",
                                                  "Text Files (*.dat)", options=options)
        if fileName and hasattr(self, 'Measurement'):
            name = fileName[:-4]

            filenameJSON = name + '.json'
            filenameDAT  = name + '.dat'

            with open(filenameJSON, 'w') as fileJSON, open(filenameDAT, 'w') as fileDAT:
                dataToWrite = self.Measurement.dumps()
                json.dump(dataToWrite, fileJSON)

                func_str = "Functions:"
                header = str() # Assume non changeing model for each spectra, to set param_names constant as a header
                for spectra in self.Measurement:
                    if spectra.parameter is not None:
                        if header == "":  # If header is not init, create header string and write it to file
                            func_str += str(spectra.model_names)
                            fileDAT.write(func_str + "\n")
                            if self.Measurement.FMR_type == "angdep":
                                header = "Angle [deg]\t"
                            else:
                                header = "Frequency [GHz]\t"

                            for name in spectra.parameter:
                                header += str(name) + "\t"
                            fileDAT.write(header + "\n")

                        #  Create parameter strings and write them to the file
                        line_2_write = str(spectra.angle) + "\t"
                        for (key, val) in spectra.parameter.valuesdict().items():
                            line_2_write += str(val) + "\t"
                        fileDAT.write(line_2_write + "\n")

    def first_guess(self, fit_range: tuple, exceptions: list) -> list:
        # Iterate over all spectras and extract approx parameters, to use as initial starting points for lineshape 1
        init_guess = []
        j_min, j = fit_range
        for l in range(self.i_min, self.i_max):
            if l not in exceptions:
                params = self.evaluate_min_max(Bdata[l][j_min:j], Adata[l][j_min:j])
                init_guess.append(params)
        init_guess = np.array(init_guess)
        return init_guess

    def evaluate_min_max(self, Bdata: list, Adata: list) -> list:
        # Evaluate lineshape parameters from "visuals"

        min_ampl_index = np.argmin(Adata, axis=None)
        max_ampl_index = np.argmax(Adata, axis=None)

        min_field = Bdata[int(np.unravel_index(min_ampl_index, Adata.shape)[0])]
        max_field = Bdata[int(np.unravel_index(max_ampl_index, Adata.shape)[0])]

        dB_PP = min_field - max_field
        Bres = min_field - dB_PP / 2 #Assuming a symmetric line
        Ampl = (abs(Adata[max_ampl_index]) + abs(Adata[min_ampl_index]))/2
        self.ui.label_test.setText('Bres, dB_PP, Ampl: %.3f, %.3f, %.3f' % (Bres, dB_PP, Ampl))

        return [Bres, dB_PP, Ampl]

    def convert_bruker_to_ascii(self, path: str) -> str:
        raise NotImplementedError
        # Return converted filename
        return

    def checkHeader(self, path: str) -> int:
        # Search the header and count unusable columns
        with open(path, 'r') as f:
            line_index = 1
            no_header_cnt = 0
            skip_value = 0
            while True:
                if line_index > 20 or no_header_cnt > 3:
                    # If read more than 20 lines or found more than 3 line without letters, break loop
                    break

                temp_line = f.readline()

                for i in temp_line:
                    if i not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '\n', '.',' ']:
                        # Find character different from list and declare a possible found header for this line
                        #print('Found possible Header', i, line_index)
                        skip_value = line_index
                        no_header_cnt = 0
                        break
                    elif i != ' ':  # Check if found data just a space, if not, then we have data.
                        # If no_header_cnt is bigger than 3, we found the header.
                        # skip_value then defines how many rows to skip
                        #print(no_header_cnt, i)
                        no_header_cnt += 1
                        break
                line_index += 1
        return skip_value


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__=="__main__":
    #appctxt = ApplicationContext() #
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MyForm()
    w.show()
    #exit_code = appctxt.app.exec_()  #
    #sys.exit(exit_code) #
    sys.exit(app.exec())