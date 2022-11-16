# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'img.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QSlider, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from test_flows_1 import RealTimeImgTest, calculate_and_test
import time as tm

sensor_time = []
sensor_val = []
valid_time = []
valid = []
varing_img = []
img_loss = []
img_time = []
real_data_simulator = None

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(740, 645)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(240, 20, 322, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(275, 60, 291, 32))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color:green")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(260, 110, 211, 141))
        self.label_3.setObjectName("label_3")
        self.label_3.setScaledContents(True)

        self.canvas = Canvas(self.centralwidget, width=8, height=4, label="Abnormality (Images)")
        self.canvas.move(0, 260)


        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "BUET_Andromeda"))
        self.label.setText(_translate("MainWindow", "Real time abnormality Detection"))
        self.label_2.setText(_translate("MainWindow", "Simulation on images"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))

    def start_img_simulate(self):
        global real_data_simulator
        real_data_simulator = RealTimeImgTest()
        if real_data_simulator.result:
            self.canvas.clean()
            self.th = Thread()
            self.th.setTerminationEnabled(True)
            self.th.newShot.connect(self.updateGraph)
            self.th.start()
        else:
            self.show_message("Error", "Have you run the matlab script ?")
    def run_static(self):
        global sensor_time, sensor_val
        output = calculate_and_test()
        if output:
            [sensor_val, sensor_time] = output
            self.canvas.plot()
    def updateGraph(self):
        global varing_img
        pi = QtGui.QPixmap(varing_img)
        self.label_3.setPixmap(pi)
        self.canvas.plot(dynamic=True)

class Thread(QThread):
    # changePixmap = pyqtSignal(QtGui.QImage)
    newShot = pyqtSignal()
    #detectionShot = pyqtSignal(int)
    #new_hit = pyqtSignal(int)

    def __init__(self, parent=None):
        QThread.__init__(self)
        global real_data_simulator, sensor_time, sensor_val, valid, valid_time
        sensor_time = []
        sensor_val = []
        valid = []
        valid_time =[]
        self.timestamps = real_data_simulator.image_times
        self.start_time = tm.time()
        self.length = self.timestamps.__len__()

    def run(self):
        self.run_realtime_data_thread()
        #self.quit()
        #self.wait()

    def stop(self):
        self.quit()
        self.wait()

    def run_realtime_data_thread(self):
        global real_data_simulator, sensor_time, sensor_val, valid, valid_time, varing_img
        for i in range(self.length):
            while tm.time() - self.start_time < self.timestamps[i]:
                tm.sleep(0.001)
            valid.append(0.5)
            valid_time.append(self.timestamps[i])
            self.newShot.emit()
            [abnormality, timeshot] = real_data_simulator.calculate_step(i)
            if abnormality:
                sensor_val.append(abnormality)
                sensor_time.append(self.timestamps[i])
            varing_img = real_data_simulator.image_paths[i]
            self.newShot.emit()
        real_data_simulator.clear_session()
        self.stop()

class Canvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=90, label="example", plot_type="image"):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_title(label)
        self.plot_type = plot_type

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # self.plot()
        self.axes.set_ylim([-0.1, 1.1])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Abnormality (0 to 1)')
        self.axes.grid()
        if plot_type.__eq__("sensor"):
            self.axes.set_title('Abnormality based on IMU sensor values')
        else:
            self.axes.set_title('Abnormality based on images')
        # self.axes.set_axisbelow(True)
        # self.axes.yaxis.grid(color='gray')

    def clean(self):
        self.axes.cla()
        self.fig.legends = []
        self.axes.set_ylim([-0.1, 1.1])
        if self.plot_type.__eq__("sensor"):
            self.axes.set_title('Abnormality based on IMU sensor values')
        else:
            self.axes.set_title('Abnormality based on images')
        self.axes.set_xlabel('Time (seconds)')
        self.axes.set_ylabel('Abnormality (0 to 1)')
        self.axes.grid()

    def plot(self, dynamic=False):
        global sensor_val, sensor_time, valid, valid_time, sensor_ori, sensor_acc, sesor_vel, cat_wise_abnormality \
            , img_loss, img_time
        if self.plot_type.__eq__("image"):
            if not dynamic:
                self.clean()
            else:
                aaa, = self.axes.plot(valid_time, valid, 'C1', color='lightpink', label='line 1',
                                      linewidth=2)  # 'go-'
            bbb, = self.axes.plot(sensor_time, sensor_val, label="Abnormality")

            if dynamic:
                if self.fig.legends.__len__() == 0:
                    self.fig.legend((aaa, bbb), ('Image Available', 'Abnormality'), 'upper left')

        # self.axes.grid() #color='r', linestyle='-', linewidth=2
        self.draw()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.start_img_simulate()
    #ui.run_static()
    sys.exit(app.exec_())
