import sys
import pandas as pd
import numpy as np
import math as m
import pyqtgraph as pg

from PyQt5.QtWidgets import QMainWindow, QApplication,QFileDialog, QCheckBox, QLabel, QMessageBox, QShortcut,\
    QTreeWidgetItem, QDoubleSpinBox
from PyQt5.QtCore import QThread, pyqtSignal, QSignalBlocker
from PyQt5.Qt import Qt, QUrl, QDesktopServices
from PyQt5.QtGui import QKeySequence
from PyQt5.uic import loadUi

from lib.CustomWidgets import *
from lib.Fitting import *
from tools.func_gen import Fit_Models

from lmfit import Model, Parameters

from typing import List, Set, Dict, Tuple, Optional

class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("lib/Fit-programm.ui", self)

        # File Menu
        self.actionOpen.triggered.connect(self.openFileDialog)
        #self.actionSave.triggered.connect(self.saveFileDialog)
        #self.actionExit.triggered.connect(self.Exit)

        self.Button_add_function.clicked.connect(self.add_func)
        self.Button_remove_function.clicked.connect(self.remove_func)

        self.pushButton.clicked.connect(self.test)

        self.func_number = 1
        self.func_removed = []

        self.populate_combobox()


    def test(self, *args, **kwargs):
        self.getValuesFromTree()
        #print(self.Models.MODELS.get('Lorentz'))
        #print(self.paramsLUT.get(self.i).get('params'))
        #for name in self.paramsLUT.get(self.i).get('params'):
        #    print(name)
        #print(self.paramsLUT.get(self.i).get('params').get('A1'))

    def changeSpectra(self, spectra: int):
        self.i = spectra
        self.plot()

    def populate_combobox(self):
        self.Models = Fit_Models()
        self.comboBox_fit_model.addItems(list(self.Models.MODELS.keys()))

    def add_func(self):
        self.resetCurrentParamLUTentry()
        # Get func name from combobox and add it to self.funcs
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

    def populate_tree(self, func_name: str, number: int):
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

    def resetCurrentParamLUTentry(self):
        try:
            self.paramsLUT[self.i]['models'] = None
            self.paramsLUT[self.i]['params'] = None
        except AttributeError:
            self.openFileDialog()

    def getValuesFromTree(self):
        # Reads the spin box values and saves them into self.paramsLUT
        # If Spectra self.i hasnt been initiated yet, call self.setupParamaters()
        root = self.Parameter_tree.invisibleRootItem()
        child_count = root.childCount()
        funcNames = []

        # get the names
        for i in range(child_count):
            item = root.child(i)
            name = item.text(0) #.split(" ")[0]
            funcNames.append(name)

        # Check if spectra has been init yet
        if self.paramsLUT[self.i]['params'] == None:
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

        LUTparams = self.paramsLUT.get(self.i).get('params')
        for name in LUTparams:
            arg = params.get(name)
            LUTparams[name].value = arg.get("value")
            LUTparams[name].min = arg.get("min")
            LUTparams[name].max = arg.get("max")
            LUTparams[name].vary = arg.get("state")

    def setupParameters(self, names: dict):
        # Called for "None" entry in paramsLUT
        # Create an lmfit Model using function names present in tree, then make lmfit Params out of this

        func = self.Models.getModelFunc(names, self.i)
        # func[0] is the function string, func[1] is its reference, func[2] is func_args
        exec(func[0], globals()) # Create function

        # with func[1] create lfmit Model()
        fit_model = Model(globals().get(func[1]))
        params = fit_model.make_params()

        self.paramsLUT[self.i]['models'] = fit_model
        self.paramsLUT[self.i]['params'] = params

        #print(eval(test[1] + "(1, 2, 3, 4)"))

    def openFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home', 'Converted Files (*.txt *.dat *.asc) ;;'
                                                                        ' Raw Files (*.DTA *.par)')

        if fname[0]:
            if fname[0][-3:] == 'DTA' or fname[0][-3:] == 'par':
                fname[0] = self.convert_bruker_to_ascii(fname[0])

            row_skip = self.checkHeader(fname[0])
            df = pd.read_csv(fname[0], names=['index', 'Field [G]', 'Sample Angle [deg]', 'Intensity []']
                             ,skiprows=row_skip, sep='\s+')

            counts = np.unique(df['Field [G]'].to_numpy()).shape[0]
            # Counts is equivalent to the  number of Magnetic field steps in the measurement e.g. 1024,2048,4096....
            chunksize = int(df.shape[0] / counts)  # Number of measurements with length counts

            FieldData = np.split(np.true_divide(df['Field [G]'].to_numpy(), 10000), chunksize)
            AmplData = np.split(df['Intensity []'].to_numpy(), chunksize)
            AngleData = np.split(df['Sample Angle [deg]'].to_numpy(), chunksize)

            self.i = 0      # self.i is the variable defining which spectra is plotted
            self.i_min = 0
            self.i_max = chunksize

            self.select_datanumber.setValue(self.i)
            self.select_datanumber.setMinimum(self.i_min)
            self.select_datanumber.setMaximum(self.i_max)
            self.select_datanumber.valueChanged.connect(self.changeSpectra)

            self.H_min = min(FieldData[0])
            self.H_max = max(FieldData[0])
            self.H_ratio = (counts)/(self.H_max - self.H_min)
            self.H_offset = - self.H_ratio * self.H_min

            self.Plot_Indi_View.lr.setBounds([self.H_min, self.H_max])  # Set Range Boundaries for linearregionitem
            self.select_datanumber.setMaximum(self.i_max)
            self.progressBar.setMaximum(self.i_max - 1)

            self.createLUT(AmplData, FieldData, AngleData)

            self.plot()

    def createLUT(self, AmplData: list, FieldData: list, AngleData: list):
        # Now create LUT to associate fit parameters to a given spectra
        # The idea is to use a dict() and profit of look up speed due to hashing

        # { i: {Ampl: [amplitude data], Field: [field data], Angle: [angle data],
        #   models: [Lorentz1, Dyson2, Lorentz3, ...],
        #   params: [fitted params/values of Qdoublespinbox as lmfit Parameter object]
        #   }   }

        self.paramsLUT = {}
        for i in range(self.i_min, self.i_max):
            self.paramsLUT[i] = {'Ampl': AmplData[i],'Field': FieldData[i],'Angle': AngleData[i],
                                 'models': None, 'params': None}
        # because data hasn't been fited yet keys: models, params are empty

    def plotFitData(self):
        # Plot the fitted data + individual functions
        # This needs to be called for every change

        data = self.paramsLUT.get(self.i)
        H_data = data.get('Field')
        Ampl_data = data.get('Ampl')

        # j_min, j = self.getFitRegion()
        params = data.get('params')

        # evaluate the fit model with given parameters params then plot
        fitted_data = data.get('models').eval(params=params, B=H_data)

        pen_result = pg.mkPen((255, 0, 0), width=3)
        self.pltFitData.setData(H_data, fitted_data, name='Result', pen=pen_result)
        self.pltFitDataRange.setData(H_data, fitted_data, name='Result', pen=pen_result)


    def plot(self):
        # Plot the Background/Experiment
        # This only needs to be called once, every spectra change
        # First Plot Background, then prepare plot for fit data
        data = self.paramsLUT.get(self.i)
        H_data = data.get('Field')
        Ampl_data = data.get('Ampl')

        #j_min, j = self.getFitRegion()
        view = self.Plot_Indi_View
        view_range = self.Plot_Indi_View

        view.plt.clear()  # Delete previous data
        view_range.plt_range.clear() # Delete previous data

        view.plt.plot(H_data, Ampl_data, name='Experiment', pen=(255, 255, 255))  # Plot Experiment data
        view_range.plt_range.plot(H_data, Ampl_data, pen=(255, 255, 255))

        view.plt.addItem(view.lr)  # Plot Linear Range Select Item
        view.plt.addItem(view.label, ignoreBounds=True)

        params = data.get('params')

        if params == None or params == False:
            # No fit to plot, so end call here
            self.getValuesFromTree()
            #return

        # Prepare plots for fit data
        self.pltFitData = view.plt.plot()
        self.pltFitDataRange = view_range.plt_range.plot()
        self.plotFitData()

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

if __name__=="__main__":
    #appctxt = ApplicationContext() #
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MyForm()
    w.show()
    #exit_code = appctxt.app.exec_()  #
    #sys.exit(exit_code) #
    sys.exit(app.exec())