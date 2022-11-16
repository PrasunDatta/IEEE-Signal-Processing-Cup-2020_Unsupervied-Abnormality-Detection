# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ex.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QSlider, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
#from functools import partial
import csv
from utils import get_file_name_without_ext
from train_sensor import train_sensor_values
from test_sensor import test_sensor_values, RealTimeTestObj
from utils import get_default_model_path, get_info_path, get_bag_image_sample
from train_flows import calculate_flow_and_train
from test_flows import calculate_and_test
import matplotlib.pyplot as plt
import time as tm
import json
import os

min_lstm_len = 1
max_lstm_len = 7
lstm_len = 5
min_scaling = 2
max_scaling = 10
scaling = 6
min_border = 0.04
max_border = 0.1
border = 0.06
min_shifting = 2
max_shifting = 6
shifting = 3
cat_wise_abnormality = False
real_time_exp = False

train_bags_file = 'train_bags.txt'
test_bag_file = 'test_bag.txt'
bag_json_file = 'info.json'

#cmnd = "\"D:\\Program Files\\MATLAB\\R2018b\\bin\\matlab.exe\" -nodisplay -nosplash" \
       #" -nodesktop -r \"run('Mat.m');exit;\""
train_bags_selected = False
test_bag_selected = False
test_bag_path = ''
train_model_path = ''
test_model_path = get_default_model_path()
sensor_time = []
sensor_val = []
sesor_vel = []
sensor_acc = []
sensor_ori = []
valid_time = []
valid = []
img_loss = []
img_time = []
real_data_simulator = None

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        global min_lstm_len, max_lstm_len, lstm_len, min_scaling, max_scaling, scaling, min_border, max_border,\
            border, cat_wise_abnormality, real_time_exp, min_shifting, max_shifting, shifting
        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(740, 614)
        MainWindow.setMaximumSize(QtCore.QSize(720, 16777215))
        
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setEnabled(True)
        self.scrollArea.setGeometry(QtCore.QRect(0, 0, 710, 1581))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setAutoFillBackground(False)
        self.scrollArea.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setWidgetResizable(False)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 708, 1579))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollAreaWidgetContents.sizePolicy().hasHeightForWidth())
        self.scrollAreaWidgetContents.setSizePolicy(sizePolicy)
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label.setGeometry(QtCore.QRect(10, 39, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semibold")
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")

        # Plus Sign
        self.toolButton = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton.setGeometry(QtCore.QRect(400, 130, 41, 41))
        self.toolButton.setAutoFillBackground(False)
        self.toolButton.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.toolButton.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons/plus.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton.setIcon(icon)
        self.toolButton.setIconSize(QtCore.QSize(30, 30))
        self.toolButton.setObjectName("toolButton")
        self.toolButton.clicked.connect(self.openFileNamesDialog)

        # Use Submitted model option
        self.checkBox = QtWidgets.QCheckBox(self.scrollAreaWidgetContents)
        self.checkBox.setGeometry(QtCore.QRect(10, 20, 291, 17))
        self.checkBox.setObjectName("checkBox")
        #self.checkBox.stateChanged.connect(partial(self.use_submitted_model))
        self.checkBox.stateChanged.connect(self.use_submitted_model)

        # train button
        self.pushButton = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton.setGeometry(QtCore.QRect(215, 180, 90, 50))
        self.pushButton.setText("")
        icon1 = QtGui.QIcon()
        #icon1.addPixmap(QtGui.QPixmap("C:/Users/USER/Downloads/disabled_train.png"), QtGui.QIcon.Normal,QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap("gui/icons/enabled_train.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.pushButton.setIcon(icon1)
        self.pushButton.setIconSize(QtCore.QSize(100, 50))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.on_train_clicked)

        # Test Model intro
        self.label_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_2.setGeometry(QtCore.QRect(10, 240, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semibold")
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        # Browse for Test File
        self.toolButton_2 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_2.setGeometry(QtCore.QRect(400, 270, 41, 41))
        self.toolButton_2.setAutoFillBackground(False)
        self.toolButton_2.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons/browse.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_2.setIcon(icon)
        self.toolButton_2.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_2.setObjectName("toolButton_2")
        self.toolButton_2.clicked.connect(self.openFileNameDialog)

        # Files List Train & Test
        self.label_3 = QtWidgets.QPlainTextEdit(self.scrollAreaWidgetContents)
        self.label_3.setGeometry(QtCore.QRect(10, 70, 391, 101))
        self.label_3.setStyleSheet("padding-left:4; padding-top:4; padding-bottom:4; padding-right:4;")
        self.label_3.setObjectName("label_3")
        self.label_3.setReadOnly(True)
        self.label_4 = QtWidgets.QPlainTextEdit(self.scrollAreaWidgetContents)
        self.label_4.setGeometry(QtCore.QRect(10, 270, 391, 30))
        self.label_4.setStyleSheet("padding-left:4; padding-top:4; padding-bottom:4; padding-right:4;")
        self.label_4.setObjectName("label_4")
        self.label_3.setReadOnly(True)


        # model description
        self.plainTextEdit_3 = QtWidgets.QPlainTextEdit(self.scrollAreaWidgetContents)
        self.plainTextEdit_3.setGeometry(QtCore.QRect(230, 320, 261, 61))
        self.plainTextEdit_3.setObjectName("plainTextEdit_3")

        # Browse for model
        self.toolButton_3 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_3.setGeometry(QtCore.QRect(491, 350, 41, 41))
        self.toolButton_3.setAutoFillBackground(False)
        self.toolButton_3.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons/browse.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_3.setIcon(icon)
        self.toolButton_3.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_3.setObjectName("toolButton_3")
        self.toolButton_3.clicked.connect(self.openFolderDialog)

        # LSTM sequnce length for train
        self.horizontalSlider_4 = QtWidgets.QSlider(self.scrollAreaWidgetContents)
        self.horizontalSlider_4.setGeometry(QtCore.QRect(30, 200, 160, 22))
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.horizontalSlider_4.setMinimum(min_lstm_len)
        self.horizontalSlider_4.setMaximum(max_lstm_len)
        self.horizontalSlider_4.setValue(lstm_len)
        self.horizontalSlider_4.setTickPosition(QSlider.TicksBelow)
        self.horizontalSlider_4.setTickInterval(1)
        self.horizontalSlider_4.valueChanged.connect(self.lstm_len_changed)
        self.label_8 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_8.setGeometry(QtCore.QRect(40, 180, 151, 16))
        self.label_8.setObjectName("label_8")
        #line
        self.line = QtWidgets.QFrame(self.scrollAreaWidgetContents)
        self.line.setGeometry(QtCore.QRect(0, 230, 720, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        # options for testing
        self.label_9 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_9.setGeometry(QtCore.QRect(10, 300, 100, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semibold")
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.checkBox_2 = QtWidgets.QCheckBox(self.scrollAreaWidgetContents)
        self.checkBox_2.setGeometry(QtCore.QRect(10, 330, 201, 17))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_2.setChecked(cat_wise_abnormality)
        self.checkBox_2.stateChanged.connect(self.on_cat_wise_pref_changed)
        self.checkBox_3 = QtWidgets.QCheckBox(self.scrollAreaWidgetContents)
        self.checkBox_3.setGeometry(QtCore.QRect(10, 360, 131, 17))
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_3.setChecked(real_time_exp)
        self.checkBox_3.stateChanged.connect(self.on_real_time_pref_changed)
        self.checkBox_4 = QtWidgets.QCheckBox(self.scrollAreaWidgetContents)
        self.checkBox_4.setGeometry(QtCore.QRect(10, 385, 131, 17))
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_4.stateChanged.connect(self.on_edit_parameter_option)
        self.horizontalSlider = QtWidgets.QSlider(self.scrollAreaWidgetContents)
        self.horizontalSlider.setGeometry(QtCore.QRect(10, 435, 160, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.setMinimum(min_scaling)
        self.horizontalSlider.setMaximum(max_scaling)
        self.horizontalSlider.setValue(scaling)
        self.horizontalSlider.setTickPosition(QSlider.TicksBelow)
        self.horizontalSlider.setTickInterval(1)
        self.horizontalSlider.valueChanged.connect(self.scaling_changed)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.scrollAreaWidgetContents)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(10, 485, 160, 22))
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.horizontalSlider_2.setMinimum(int(min_border*100))
        self.horizontalSlider_2.setMaximum(int(max_border*100))
        self.horizontalSlider_2.setValue(int(border*100))
        self.horizontalSlider_2.setTickPosition(QSlider.TicksBelow)
        self.horizontalSlider_2.setTickInterval(1)
        self.horizontalSlider_2.valueChanged.connect(self.border_changed)
        self.horizontalSlider_3 = QtWidgets.QSlider(self.scrollAreaWidgetContents)
        self.horizontalSlider_3.setGeometry(QtCore.QRect(10, 535, 160, 22))
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.horizontalSlider_3.setMinimum(min_shifting)
        self.horizontalSlider_3.setMaximum(max_shifting)
        self.horizontalSlider_3.setValue(shifting)
        self.horizontalSlider_3.setTickPosition(QSlider.TicksBelow)
        self.horizontalSlider_3.setTickInterval(1)
        self.horizontalSlider_3.valueChanged.connect(self.shifting_changed)
        self.label_5 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_5.setGeometry(QtCore.QRect(10, 410, 151, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_6.setGeometry(QtCore.QRect(10, 460, 180, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_7.setGeometry(QtCore.QRect(10, 510, 180, 13))
        self.label_7.setObjectName("label_7")

        #image
        self.label_10 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_10.setGeometry(QtCore.QRect(210, 410, 221, 131))
        self.label_10.setObjectName("label_4")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        #self.label_10.setMaximumSize(QtCore.QSize(221, 131))
        #self.label_10.setPixmap(QtGui.QPixmap("bf2.jpg"))
        self.label_10.setScaledContents(True)

        # test button
        self.testButton = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.testButton.setGeometry(QtCore.QRect(315, 552, 75, 50))
        self.testButton.setText("")
        icon1 = QtGui.QIcon()
        # icon1.addPixmap(QtGui.QPixmap("C:/Users/USER/Downloads/disabled_train.png"), QtGui.QIcon.Normal,QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap("gui/icons/enabled_test.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.testButton.setIcon(icon1)
        self.testButton.setIconSize(QtCore.QSize(100, 50))
        self.testButton.setObjectName("testButton")
        self.testButton.clicked.connect(self.on_test_clicked)


        
        self.canvas = Canvas(self.scrollAreaWidgetContents, width=8, height=4, label="Abnormality Based on IMU Data",
                             plot_type="sensor")
        self.canvas.move(0, 600)

        self.canvas_2 = Canvas(self.scrollAreaWidgetContents, width=8, height=4, label="Abnormality Based on Image",
                               plot_type="image")
        self.canvas_2.move(0, 1000)

        # zoom button for graph 1 and 2
        self.zoomButton = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.zoomButton.setGeometry(QtCore.QRect(630, 610, 25, 25))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("gui/icons/zoom.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.zoomButton.setIcon(icon1)
        self.zoomButton.setIconSize(QtCore.QSize(20, 20))
        self.zoomButton.setObjectName("zoomButton")
        self.zoomButton.clicked.connect(self.on_zoom1_clicked)
        self.zoomButton_2 = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.zoomButton_2.setGeometry(QtCore.QRect(630, 1010, 25, 25))
        self.zoomButton_2.setIcon(icon1)
        self.zoomButton_2.setIconSize(QtCore.QSize(20, 20))
        self.zoomButton_2.setObjectName("zoomButton_2")
        self.zoomButton_2.clicked.connect(self.on_zoom2_clicked)
        
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        
        
        MainWindow.setCentralWidget(self.scrollArea)
        
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 499, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        global lstm_len, border, scaling, shifting
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "BUET (Team Name)"))
        self.label.setText(_translate("MainWindow", "Choose Normal Dataset(s) For Training"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.checkBox.setText(_translate("MainWindow", "Use Submitted Model for SP CUP 2020 Primary Stage"))
        #self.pushButton.setText(_translate("MainWindow", "Train Now"))
        self.label_2.setText(_translate("MainWindow", "Test Model on new bag file"))
        #self.pushButton_2.setText(_translate("MainWindow", "Choose File"))
        self.label_3.setPlainText(_translate("MainWindow", "No Dataset Added"))
        self.label_4.setPlainText(_translate("MainWindow", "Not Selected"))
        self.checkBox_2.setText(_translate("MainWindow", "View Category-wise Abnormality"))
        self.checkBox_3.setText(_translate("MainWindow", "Real-Time Experience"))
        self.checkBox_4.setText(_translate("MainWindow", "Change Parameters"))
        self.label_5.setText(_translate("MainWindow", "Scaling Factor = "+str(scaling)))
        self.label_6.setText(_translate("MainWindow", "Minimum σ = "+str(border)))
        self.label_7.setText(_translate("MainWindow", "Shifting Factor = "+str(shifting)))
        self.label_8.setText(_translate("MainWindow","Sequence Length For LSTM = "+str(lstm_len)))
        self.label_9.setText(_translate("MainWindow", "Options"))

        self.checkBox.setChecked(True)
        self.checkBox_4.setChecked(True)
        self.checkBox_4.setChecked(False)
        self.setPixMap()

    def openFolderDialog(self):
        global test_model_path
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self.MainWindow, 'Select model directory', './models/')
        if folder:
            #print(fileName)
            test_model_path = folder
            self.showModelInfo()

    def showModelInfo(self):
        global test_model_path
        if os.path.exists(get_info_path(test_model_path)):
            with open(get_info_path(test_model_path)) as json_file:
                data = json.load(json_file)
                seq_len = data['Sequence Length (Sensor)']
                cration_time = data['Creation Time']
                train_files = data['Train Files']
                total_file = data['Total Train Files']
                total_sequences = data['Total Sequences (Sensor)']
                total_opt_flow = data['Total Optical Flow (Images)']
                self.plainTextEdit_3.setPlainText("Using Model: {}\nSequence Length(Sensor): {}"
                                                  "\nTrained on: {}\nTrain Dataset: {}\nTotal Dataset: {}"
                                                  "\nTotal Sequences(Sensor): {}\nTotal Optical Flow(Images):{}"
                                                  .format(test_model_path,
                    seq_len, cration_time, train_files, total_file, total_sequences, total_opt_flow))
        else:
            self.plainTextEdit_3.setPlainText("No model found within "+test_model_path)




            
    def openFileNameDialog(self):
        global test_bag_selected, test_bag_path
        options = QtWidgets.QFileDialog.Options()
        #options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.MainWindow,"QFileDialog.getOpenFileName()",
                                                            "","Rosbag Files (*.bag)", options=options)
        if fileName:
            #print(fileName)
            self.label_4.setPlainText(fileName)
            file1 = open("test_bag.txt", "w")
            L = [fileName.replace('/', '\\') + "\n"]
            file1.writelines(L)
            file1.close()
            test_bag_path = get_file_name_without_ext(fileName.replace('/', '\\'))
            test_bag_selected = True
    
    def openFileNamesDialog(self):
        global train_bags_selected
        options = QtWidgets.QFileDialog.Options()
        #options |= QtWidgets.QFileDialog.DontUseNativeDialog
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self.MainWindow,"QFileDialog.getOpenFileNames()",
                                                          "", "Rosbag Files (*.bag)", options=options)
        if files:
            #print(files)
            files_list = ""
            files_eof = []
            for i, file in enumerate(files):
                if i:
                    files_list += "\n"
                files_list += str(i+1)+". "+file
                files_eof.append(file.replace('/', '\\')+"\n")
            self.label_3.setPlainText(files_list)
            file1 = open("train_bags.txt", "w")
            file1.writelines(files_eof)
            file1.close()
            train_bags_selected = True

    def disable_training(self):
        self.pushButton.setEnabled(False) # train button
        self.toolButton.setEnabled(False) # add train dataset
        self.label_3.setEnabled(False) # list of train dataset
        self.horizontalSlider_4.setEnabled(False)
        self.label_8.setEnabled(False)
        self.plainTextEdit_3.setEnabled(False)
        self.toolButton_3.setEnabled(False)


    def enable_training(self):
        self.pushButton.setEnabled(True) # train button
        self.toolButton.setEnabled(True) # add train dataset
        self.label_3.setEnabled(True) # list of train dataset
        self.horizontalSlider_4.setEnabled(True)
        self.label_8.setEnabled(True)
        self.plainTextEdit_3.setEnabled(True)
        self.toolButton_3.setEnabled(True)

    def enable_testing(self):
        self.testButton.setEnabled(True)
    def disable_testing(self):
        self.testButton.setEnabled(False)

    def on_train_clicked(self):
        global train_bags_selected, test_model_path
        if not train_bags_selected:
            self.show_message("Select Bag File!", "You haven't selected bag file(s) for training")
        else:
            self.show_dialog_for_matlab()
            result = train_sensor_values()
            if not result:
                self.show_message("Error", "Have you run the matlab script ?")
            else:
                test_model_path, about_model = result
                response = self.show_message("Training on sensor values done","Click ok to train on images")
                calculate_flow_and_train(test_model_path, about_model)
                self.showModelInfo()
                self.show_message("Success", "Train is successful. Using model \""+test_model_path+"\"")
        #with open(train_bags_file) as f:
            #for line in f:
                #print(get_file_name_without_ext(line))
                #line_count += 1"""

    def on_test_clicked(self):
        global test_bag_selected, test_model_path, scaling, shifting, border, sensor_time,\
            sensor_val, real_time_exp, real_data_simulator, sesor_vel, sensor_acc, sensor_ori, img_loss, img_time
        if not test_bag_selected:
            self.show_message("Select Bag File!", "You haven't selected any bag file for testing")
        else:
            self.show_dialog_for_matlab()
            self.show_sample_bag_image()
            if real_time_exp:
                real_data_simulator = RealTimeTestObj(test_model_path, scaling, shifting, border)
                if real_data_simulator.result:
                    self.canvas.clean()
                    self.th = Thread()
                    self.th.setTerminationEnabled(True)
                    self.th.newShot.connect(self.updateGraph)
                    self.th.start()
                else:
                    self.show_message("Error", "Have you run the matlab script ?")
            else:
                result = test_sensor_values(test_model_path, scaling, shifting, border)
                if not result:
                    self.show_message("Error", "Have you run the matlab script ?")
                else:
                    [sesor_vel, sensor_acc, sensor_ori, sensor_val, sensor_time] = result
                    self.canvas.plot()
                    response = self.show_message("Abnormality on sensor values detected",
                                                 "Click ok to detect abnormality on image samples")

                    output = calculate_and_test(test_model_path)
                    if output:
                        [img_loss, img_time] = output
                        self.canvas_2.plot()


    def show_sample_bag_image(self):
        pi = QtGui.QPixmap(get_bag_image_sample(test_bag_path))
        print(get_bag_image_sample(test_bag_path))
        self.label_10.setPixmap(pi)


    def show_dialog_for_matlab(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("RUN MATLAB SCRIPT")
        msg.setInformativeText("You need to run Matlab script \"BagExtractor.m\" and then click OK")
        msg.setWindowTitle("Message")
        msg.setDetailedText("This script will extract necessary sensor values and image data as csv and jpg file. \n"
                            "This will be extracted on data/{bag_file_name} folder.\n"
                            "If you are sure that you have already extracted these for selected bag file(s) you can "
                            "proceed. The Matlab script has been tested on MATLAB R2018b 9.5.0.944444 using "
                            "Windows 10 64 bit platform.")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        # msg.buttonClicked.connect(msgbtn)
        returnValue = msg.exec()
        #if returnValue == QMessageBox.Ok:
            #print('OK clicked')
        #else:
            #print('Cancel clicked')
    def show_message(self, str1, str2):
        msg = QMessageBox()
        msg.setText(str1)
        msg.setInformativeText(str2)
        msg.setWindowTitle("Message")
        msg.setStandardButtons(QMessageBox.Ok)
        returnValue = msg.exec()
        return  returnValue

    def on_zoom1_clicked(self):
        global sensor_time, sensor_val, cat_wise_abnormality, sensor_acc, sensor_ori, sesor_vel, real_time_exp
        bbb, = plt.plot(sensor_time, sensor_val) #, color='red'
        if not real_time_exp:
            if cat_wise_abnormality:
                ccc, = plt.plot(sensor_time, sensor_ori, label="Abnormality(Orientation)")
                ddd, = plt.plot(sensor_time, sensor_acc, label="Abnormality(Accelerometer)")
                eee, = plt.plot(sensor_time, sesor_vel, label="Abnormality(Angular Velocity)")
                plt.legend((bbb, ccc, ddd, eee), ('Abnormality', 'Abnormality(Orientation)',
                                               'Abnormality(Accelerometer)','Abnormality(Angular Velocity)'))
        axes = plt.gca()
        axes.grid()
        axes.set_ylim([-0.1, 1.1])
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Abnormality (0 to 1)')
        plt.show()

    def on_zoom2_clicked(self):
        global img_time, img_loss
        bbb, = plt.plot(img_time, img_loss)  # , color='red'
        """if not real_time_exp:
            if cat_wise_abnormality:
                ccc, = plt.plot(sensor_time, sensor_ori, label="Abnormality(Orientation)")
                ddd, = plt.plot(sensor_time, sensor_acc, label="Abnormality(Accelerometer)")
                eee, = plt.plot(sensor_time, sesor_vel, label="Abnormality(Angular Velocity)")
                plt.legend((bbb, ccc, ddd, eee), ('Abnormality', 'Abnormality(Orientation)',
                                                  'Abnormality(Accelerometer)', 'Abnormality(Angular Velocity)'))"""
        axes = plt.gca()
        axes.grid()
        axes.set_ylim([-0.1, 1.1])
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Abnormality (0 to 1)')
        plt.show()

    def on_cat_wise_pref_changed(self, int):
        global cat_wise_abnormality
        cat_wise_abnormality = self.checkBox_2.isChecked()

    def on_real_time_pref_changed(self, int):
        global real_time_exp
        real_time_exp = self.checkBox_3.isChecked()

    def parameter_option_enable(self, bool):
        self.horizontalSlider.setEnabled(bool)
        self.horizontalSlider_2.setEnabled(bool)
        self.horizontalSlider_3.setEnabled(bool)
        self.label_5.setEnabled(bool)
        self.label_6.setEnabled(bool)
        self.label_7.setEnabled(bool)

    def on_edit_parameter_option(self, int):
        self.parameter_option_enable(int > 1)


    def lstm_len_changed(self):
        global lstm_len
        lstm_len = self.horizontalSlider_4.value()
        self.label_8.setText("Sequence Length For LSTM = "+str(lstm_len))

    def scaling_changed(self):
        global scaling
        scaling = self.horizontalSlider.value()
        self.label_5.setText("Scaling Factor = "+str(scaling))

    def shifting_changed(self):
        global shifting
        shifting = self.horizontalSlider_3.value()
        self.label_7.setText("Shifting Factor = "+str(shifting))

    def border_changed(self):
        global border
        border = self.horizontalSlider_2.value()/100
        self.label_6.setText("Minimum σ = "+str(border))


    def use_submitted_model(self, int):
        global test_model_path
        if int < 1:
            self.enable_training()
        else:
            test_model_path = get_default_model_path()
            self.showModelInfo()
            self.disable_training()

    def setPixMap(self):
        pi = QtGui.QPixmap('gui/icons/1.jpg')
        self.label_10.setPixmap(pi)
        #self.label_10.setPixmap(QtGui.QPixmap('gui/icons/1.jpg'))

    def updateGraph(self):
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
        self.timestamps = real_data_simulator.timestamps
        self.delay = real_data_simulator.delay
        self.length = len(self.timestamps)
        self.start_time = tm.time()

    def run(self):
        self.run_realtime_data_thread()
        #self.quit()
        #self.wait()

    def stop(self):
        self.quit()
        self.wait()

    def run_realtime_data_thread(self):
        global real_data_simulator, sensor_time, sensor_val, valid, valid_time
        for i in range(self.length):
            while tm.time() - self.start_time < self.timestamps[i]:
                tm.sleep(0.001)
            valid.append(0.5)
            valid_time.append(self.timestamps[i])
            self.newShot.emit()
            [abnormality, timeshot] = real_data_simulator.calculate_step(i)
            if abnormality:
                sensor_val.append(abnormality)
                sensor_time.append(self.timestamps[i-self.delay])
            self.newShot.emit()
        real_data_simulator.clear_session()
        self.stop()



class Canvas(FigureCanvas):
    def __init__(self, parent = None, width = 5, height = 5, dpi = 90, label="example", plot_type="sensor"):
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
 
        #self.plot()
        self.axes.set_ylim([-0.1, 1.1])
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Abnormality (0 to 1)')
        self.axes.grid()
        if plot_type.__eq__("sensor"):
            self.axes.set_title('Abnormality based on IMU sensor values')
        else:
            self.axes.set_title('Abnormality based on images')
        # self.axes.set_axisbelow(True)
        #self.axes.yaxis.grid(color='gray')

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
        global sensor_val, sensor_time, valid, valid_time, sensor_ori, sensor_acc, sesor_vel, cat_wise_abnormality\
                , img_loss, img_time
        if self.plot_type.__eq__("sensor"):
            if not dynamic:
                self.clean()
            else:
                aaa, = self.axes.plot(valid_time, valid, 'C1', color='lightpink', label='line 1', linewidth=2) #'go-'
            bbb, = self.axes.plot(sensor_time, sensor_val, label = "Abnormality")
            if not dynamic:
                if cat_wise_abnormality:
                    ccc, = self.axes.plot(sensor_time, sensor_ori, label="Abnormality(Orientation)")
                    ddd, = self.axes.plot(sensor_time, sensor_acc, label="Abnormality(Accelerometer)")
                    eee, = self.axes.plot(sensor_time, sesor_vel, label="Abnormality(Angular Velocity)")
                    self.fig.legend((bbb, ccc, ddd, eee), ('Abnormality', 'Abnormality(Orientation)',
                                                   'Abnormality(Accelerometer)','Abnormality(Angular Velocity)'), 'upper left')
            if dynamic:
                if self.fig.legends.__len__() == 0:
                    self.fig.legend((aaa, bbb), ('IMU Data Available', 'Abnormality'), 'upper left')
        elif self.plot_type.__eq__("image"):
            self.clean()
            bbb, = self.axes.plot(img_time, img_loss, label="Abnormality")

        #self.axes.grid() #color='r', linestyle='-', linewidth=2
        self.draw()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app_icon = QtGui.QIcon()
    app_icon.addFile('gui/icons/buet_64.png', QtCore.QSize(16, 16))
    app_icon.addFile('gui/icons/buet_96.png', QtCore.QSize(24, 24))
    app_icon.addFile('gui/icons/buet_96.png', QtCore.QSize(32, 32))
    app_icon.addFile('gui/icons/buet_128.png', QtCore.QSize(48, 48))
    app_icon.addFile('gui/icons/buet_256.png', QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    app.setStyleSheet('QMainWindow{background-color: darkgray;border: 1px solid black;}')

    MainWindow = QtWidgets.QMainWindow()
    #MainWindow.setWindowTitle("BUET (Team Name)")
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
