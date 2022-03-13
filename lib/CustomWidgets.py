import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from PyQt5.QtWidgets import QWidget, QVBoxLayout

class Plot_pyqtgraph(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self)
        self.vbl = QVBoxLayout()
        self.win = pg.GraphicsLayoutWidget()
        self.lr = pg.LinearRegionItem([0.01,0.15])
        self.lr.setZValue(-1)
        self.plt = self.win.addPlot()
        self.plt_range = self.win.addPlot()
        self.plt.addItem(self.lr)

        self.label = pg.TextItem(text='Hover Event', anchor=(0,0))
        self.plt.addItem(self.label, ignoreBounds=True)

        self.plt.hoverEvent = self.imageHoverEvent
        self.plt.mouseClickEvent = self.imageClickEvent

        self.plt.setLabel('left',"Amplitude (Arb. Units)")  # Y-Axis
        self.plt.setLabel('bottom',"Magnetic Field", units='T') # X-Axis
        self.plt.showGrid(True,True) # Show Grid

        self.plt_range.setLabel('left', "Amplitude (Arb. Units)")  # Y-Axis
        self.plt_range.setLabel('bottom', "Magnetic Field", units='T')  # X-Axis
        self.plt_range.showGrid(True, True)  # Show Grid

        self.lr.sigRegionChanged.connect(self.updatePlot)
        self.plt_range.sigXRangeChanged.connect(self.updateRegion)
        self.updatePlot()

        self.plt.addLegend()
        self.vbl.addWidget(self.win)
        self.setLayout(self.vbl)


    def updatePlot(self):
        self.plt_range.setXRange(*self.lr.getRegion(), padding=0)

    def updateRegion(self):
        self.lr.setRegion(self.plt_range.getViewBox().viewRange()[0])

    def imageClickEvent(self, event):
        pos = event.pos()
        ppos = self.plt.mapToParent(pos)
        print(ppos.x())

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.label.setText("")
            return
        pos = event.pos()
        x, y = pos.x(), pos.y()
        self.label.setText("Pos (Field,Angle): %0.1f, %0.1f" % (x, y))

class ParameterPlot(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self)
        pass

class Single_Plot(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self)
        pass


