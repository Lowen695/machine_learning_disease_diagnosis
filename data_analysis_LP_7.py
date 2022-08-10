import sys
from os import path, listdir

from PyQt6.QtGui import QIcon, QPixmap, QMovie
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt, QAbstractTableModel
from PyQt6.QtWidgets import QWidget, QApplication, QStackedWidget, QListWidget, QFileDialog, QTextEdit, QHBoxLayout, QComboBox,QScrollArea, QFrame
from PyQt6.QtWidgets import QFormLayout, QPushButton, QLabel, QLineEdit, QMessageBox, QTableView, QVBoxLayout,QCheckBox
from qt_material import apply_stylesheet

import pandas as pd
import pickle
import sklearn.ensemble._forest
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from seaborn import heatmap as snsh
from sklearn.metrics import confusion_matrix as confusionMatrix


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor('#9bb4b0')
        self.axes.set_facecolor('#9bb4b0')
        super().__init__(self.fig)

class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])

class GifTest(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        layout = QHBoxLayout()

        self.resize(500, 500)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.gifLabel = QLabel()
        self.movie = QMovie('images/200W.gif')
        self.gifLabel.setPixmap(QPixmap('images/icoLK.png'))
        self.gifLabel.setMovie(self.movie)
        self.movie.start()

        self.setWindowOpacity(0.3)
        layout.addWidget(self.gifLabel)

        self.setLayout(layout)

class GifTest2(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.resize(500, 500)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.gifLabel = QLabel(self)
        self.gifLabel.move(30,160)

        pixMap = QPixmap('images/icoLK.png')
        # pixMap2 = pixMap.transformed(QTransform().rotate(i))
        pixMap2 = pixMap.scaled(180,180)
        self.gifLabel.setPixmap(pixMap2)

class Data_LP(QWidget):
    def __init__(self):
        super().__init__()
        self.fileName = []
        self.FolderName = ''
        self.normalizedDataTable = pd.DataFrame()
        self.cm = ''
        self.initUI()
        self.stack1Layout()
        self.stack2Layout()
        self.stack3Layout()
    
    def initUI(self):
        self.resize(1260, 800)
        self.setWindowTitle('Lipid Knowledge data analysis')
        layout = QHBoxLayout()

        self.stack = QStackedWidget()
        self.stack1 = QWidget()
        self.stack2 = QWidget()
        self.stack3 = QWidget()
        self.stack.addWidget(self.stack1)
        self.stack.addWidget(self.stack2)
        self.stack.addWidget(self.stack3)
        self.list = QListWidget()
        self.list.addItems(['Import and normalize', 'Prediction', 'Analysis'])
        self.list.setCurrentRow(0)
        self.list.currentRowChanged.connect(self.display)

        layout.addWidget(self.list, 2)
        layout.addWidget(self.stack, 7)

        self.setLayout(layout)

    def stack1Layout(self):
        layoutStack1 = QFormLayout()

        self.loadIndexBt = QPushButton('                Load single name file')
        self.loadIndexBt.clicked.connect(self.loadSampleNumber)
        self.loadIndexTx = QLineEdit()
        self.loadIndexTx.setPlaceholderText('upload excel file with sample ID')

        self.loadDataBt = QPushButton('             Load single result file')
        self.loadDataBt.clicked.connect(self.loadResultFile)
        self.loadDataTx = QLineEdit()
        self.loadDataTx.setPlaceholderText('upload csv file of results')

        self.loadFolderBt = QPushButton('Load multiple files in folder')
        self.loadFolderBt.clicked.connect(self.loadFileFolder)
        self.loadFolderTx = QLineEdit()
        self.loadFolderTx.setPlaceholderText('upload multiple files in one folder')
        
        self.loadedBt = QPushButton('                                 Loaded file(s)')
        self.loadedTx = QTextEdit()
        self.loadedTx.setMaximumHeight(200)
        self.loadedTx.setEnabled(False)

        layoutSubBt = QHBoxLayout()
        submitBt = QPushButton('Submit')
        submitBt.clicked.connect(self.submitBtClicked)
        clearBt = QPushButton('Clear')
        clearBt.clicked.connect(self.clearBtClicked)

        layoutDownloadBt = QHBoxLayout()
        DownLoadBt = QPushButton('Download normalized results')
        DownLoadBt.clicked.connect(self.downloadBtClicked)
        submitForMLBt = QPushButton('Submit normalized results for analysis')
        submitForMLBt.clicked.connect(self.forMLDataSubmitBtClicked)
        layoutDownloadBt.addWidget(DownLoadBt)
        layoutDownloadBt.addWidget(submitForMLBt)
  
        layoutSubBt.addWidget(submitBt)
        layoutSubBt.addWidget(clearBt)
        layoutSubBt.addStretch()
        dummyLabel = QLabel()

        layoutStack1.addRow(self.loadIndexBt,self.loadIndexTx)
        layoutStack1.addRow(self.loadDataBt,self.loadDataTx)
        layoutStack1.addRow(self.loadFolderBt,self.loadFolderTx)
        layoutStack1.addRow(self.loadedBt, self.loadedTx)
        layoutStack1.addRow(dummyLabel, layoutSubBt)

        self.table = QTableView()
        layoutStack1.addRow(dummyLabel, self.table)
        layoutStack1.addRow(dummyLabel, layoutDownloadBt)

        # Separador = QSpacerItem(150,50,QSizePolicy.Policy.Expanding)
        # layoutStack1.addItem(Separador)
        
        self.stack1.setLayout(layoutStack1)

    def display(self, index):
        self.stack.setCurrentIndex(index)

    def loadSampleNumber(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"data files (*.csv *.xlsx *.xlsm)")
        if fname[0] != '':
            fileName = fname[0].split('/')[-1]
            self.loadIndexTx.setText(f'{fileName} was loaded')
            self.loadedTx.insertPlainText('Sample ID file: ' + fileName + '\n')
            self.fileName.append(fname[0])
            self.loadedTx.setEnabled(True)

    def loadResultFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"data files (*.csv *.xlsx *.xlsm)")
        if fname[0] != '':
            fileName = fname[0].split('/')[-1]
            self.loadDataTx.setText(f'{fileName} was loaded')
            # choice = QMessageBox.information(self, 'Meassage', 'Files were loaded successfully！\nLoad more?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel) 
            # print(choice)
            self.loadedTx.insertPlainText('Results file: ' + fileName + '\n')
            self.fileName.append(fname[0])
            self.loadedTx.setEnabled(True)

    
    def clearBtClicked(self):
        self.loadedTx.clear()
        self.loadDataTx.clear()
        self.loadIndexTx.clear()
        self.fileName = []
        self.FolderName = ''
        self.table.setModel(TableModel(pd.DataFrame([[' ',' ',' ',' ',' ',' ']]*12)))
        self.table.update()
        self.table.reset()
        self.loadedTx.setEnabled(False)
        self.normalizedDataTable = pd.DataFrame()


    def submitBtClicked(self):
        if self.FolderName != '':
            files = listdir(self.FolderName)
            try:
                nameList = pd.DataFrame()
                for item in range(0,len(files),2):
                    fileName = path.join(self.FolderName, files[item])
                    dffolder = pd.read_excel(fileName)
                    dffolder = dffolder.dropna(how='all').dropna(how='all',axis=1)
                    dffolder.reset_index(drop=True, inplace=True)
                    dffolder1 = pd.concat([dffolder.iloc[2:,2:3], dffolder.iloc[2:,7:8]], ignore_index=True, axis=0)
                    dffolder1.iloc[8:,0] = dffolder1.iloc[8:,1]
                    dffolder2=dffolder1.iloc[:,0].to_frame()
                    dffolder2.columns=[dffolder.iloc[0,0]]
                    nameList = pd.concat([nameList, dffolder2], axis=1)
                    finalNameList = nameList.transpose().stack().to_frame()
                resultList = pd.DataFrame()
                for item in range(1,len(files),2):
                    fileName = path.join(self.FolderName, files[item])
                    dffolderR = pd.read_csv(fileName, index_col=0, nrows=8)
                    dffolderR1 = dffolderR.iloc[:,:5]
                    dffolderR2 = dffolderR.iloc[:,6:11]
                    dffolderR3 = pd.concat([dffolderR1,dffolderR2], ignore_index=True, axis=0)
                    dffolderR3.iloc[8:,:5] = dffolderR3.iloc[8:,5:11]
                    dffolderR4 = dffolderR3.iloc[:,:5]
                    resultList = pd.concat([resultList,dffolderR4], axis=0)
                resultList.index=finalNameList.index
                finalResultsTable = pd.concat([finalNameList, resultList], ignore_index=True, axis=1)
                finalResultsTable.columns = ['ID', 'AG1','AG2','AG3','AG4','AG5']

                for i in range(len(finalResultsTable.index.levels[0])):
                    plates = finalResultsTable.index.levels[0][i]
                    subResultTable = finalResultsTable.loc[plates]
                    mean1 = subResultTable[subResultTable['ID']=='CAL'].iloc[:,1:][:2].mean()
                    subResultTable.iloc[:8, 1:] = subResultTable.iloc[:8, 1:].div(mean1)
                    mean2 = subResultTable[subResultTable['ID']=='CAL'].iloc[:,1:][2:].mean()
                    subResultTable.iloc[8:, 1:] = subResultTable.iloc[8:, 1:].div(mean2)
                    pd.concat([finalResultsTable,subResultTable])

                self.normalizedDataTable = finalResultsTable.round(4)
                self.model = TableModel(finalResultsTable.round(4))
                self.table.setModel(self.model)
                stylesheet = "::section{Background-color:rgb(200,200,200);border-radius:30px;}"
                self.table.horizontalHeader().setStyleSheet(stylesheet)
                self.table.sortByColumn(2, Qt.SortOrder.AscendingOrder)

                self.FolderName = ''
                self.loadedTx.clear()
                self.fileName = []
            except:
                QMessageBox.information(self,'Info','file name not correct or name and result files are not paired', QMessageBox.StandardButton.Ok)
                self.clearBtClicked()

        else:
            if len(self.fileName)  == 2:
                nameFilePath = ''
                resultFilePath = ''
                for filenamesChoice in self.fileName:
                    filenameLast4 = path.split(filenamesChoice)[1].split('.')[0][-4:]
                    if filenameLast4 == 'name':
                        nameFilePath = filenamesChoice
                    elif filenameLast4 == 'sult':
                        resultFilePath = filenamesChoice
                    else:
                        QMessageBox.information(self,'Info','Please name your file correctly', QMessageBox.StandardButton.Ok)
                if nameFilePath != '' and resultFilePath != '':
                    df = pd.read_excel(nameFilePath)
                    df2 = df.dropna(how='all').dropna(how='all',axis=1)
                    df2.reset_index(drop=True, inplace=True)
                    df3 = pd.concat([df2.iloc[2:,2:3], df2.iloc[2:,7:8]], ignore_index=True, axis=0)
                    df3.iloc[8:,0] = df3.iloc[8:,1]
                    df4=df3.iloc[:,0].to_frame()
                    df4.columns=[df2.iloc[0,0]]

                    dfr = pd.read_csv(resultFilePath, index_col=0, nrows=8)
                    dfr1 = dfr.iloc[:,:5]
                    dfr2 = dfr.iloc[:,6:11]
                    dfr3 = pd.concat([dfr1,dfr2], ignore_index=True, axis=0)
                    dfr3.iloc[8:,:5] = dfr3.iloc[8:,5:11]
                    dfr4 = dfr3.iloc[:,:5]
                    dfr4.index = df4.iloc[:,0]
                  
                    dfr4.columns = ['Ag1','Ag2','Ag3','Ag4','Ag5']

                    mean1 = dfr4.loc['CAL'][:2].mean()
                    mean2 = dfr4.loc['CAL'][2:].mean()
                    normalizedResult1 = dfr4[:8].div(mean1)
                    normalizedResult2 = dfr4[8:].div(mean2)
                    result = pd.concat([normalizedResult1, normalizedResult2])
                    result['ID'] = dfr4.index
                    result = result[['ID','Ag1','Ag2','Ag3','Ag4','Ag5']]

                    self.normalizedDataTable = result.round(4)
                    self.model = TableModel(result.round(4))
                    self.table.setModel(self.model)
                    stylesheet = "::section{Background-color:rgb(200,200,200);border-radius:30px;}"
                    self.table.horizontalHeader().setStyleSheet(stylesheet)
                    self.table.sortByColumn(2, Qt.SortOrder.AscendingOrder)
                else:
                    QMessageBox.information(self,'Info','Please upload one name file and one corresponding result file', QMessageBox.StandardButton.Ok)
                    self.clearBtClicked()
            elif len(self.fileName) > 2:
                choice = QMessageBox.warning(self, 'Info','Please submit multiple files in folder\nSubmit folder?', QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Ignore)
                if choice == QMessageBox.StandardButton.Ok:
                    self.clearBtClicked()
                    self.loadFileFolder()
                else:
                    self.clearBtClicked()
            else:
                QMessageBox.warning(self, 'Info','Please select paired name file and result file', QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Ignore)
                self.clearBtClicked()

    def loadFileFolder(self):
        fname = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if fname:
            self.FolderName = fname
            self.loadedTx.setText('Folder: ' + self.FolderName + ' was loaded\nwith files:\n')
            for item in listdir(self.FolderName):
                self.loadedTx.setEnabled(True)
                self.loadedTx.append(f'{str(item)}')

    
    def downloadBtClicked(self):
        if self.normalizedDataTable.shape[0] > 1:
            saveFileName = QFileDialog.getSaveFileName(self, 'Save file','','Excel files (*.xlsx);; CSV (*.csv);; HTML (*.html)')
            if saveFileName and saveFileName[1] == 'Excel files (*.xlsx)':
                self.normalizedDataTable.to_excel(saveFileName[0])
            elif  saveFileName and saveFileName[1] == 'CSV (*.csv)':
                self.normalizedDataTable.to_csv(saveFileName[0])
            elif saveFileName and saveFileName[1] == 'HTML (*.html)':
                self.normalizedDataTable.to_html(saveFileName[0])
            else:
                pass    
        else:
            QMessageBox.warning(self, 'Info','No data file can be downloaded', QMessageBox.StandardButton.Ok)

    def forMLDataSubmitBtClicked(self):
        if self.normalizedDataTable.shape[0] > 1:
            self.dataForML = self.normalizedDataTable
            QMessageBox.information(self, 'Info','Data was submitted for analysis', QMessageBox.StandardButton.Ok)
            self.list.setCurrentRow(1)
            self.stack.setCurrentIndex(1)
        else:
            QMessageBox.warning(self, 'Info','No data file for analysis', QMessageBox.StandardButton.Ok)

    
    def stack2Layout(self):
        layoutStack2 = QFormLayout()

        self.loadModelBt = QPushButton('                           Load model file')
        self.loadModelBt.clicked.connect(self.loadModel)
        self.loadModelTx = QLineEdit()
        self.loadModelTx.setPlaceholderText('upload pkl model file')

        self.predictBt = QPushButton('                                            Predict')
        self.predictBt.clicked.connect(self.predictSubmittedData)
        self.predictTx = QLineEdit()
        self.predictTx.setPlaceholderText('Predict pretreated and submitted data')
        
        self.tablePredict = QTableView()
        self.tablePredict.setMaximumHeight(1200)
        self.predictResultBt = QLabel()

        layoutDownloadBt = QHBoxLayout()
        DownLoadBt = QPushButton('Download predicted results')
        DownLoadBt.clicked.connect(self.downloadPredictedBtClicked)
        layoutDownloadBt.addWidget(DownLoadBt)
        layoutDownloadBt.addStretch()
        dummyBt = QLabel()

        layoutStack2.addRow(self.loadModelBt, self.loadModelTx)
        layoutStack2.addRow(self.predictBt, self.predictTx)
        layoutStack2.addRow(self.predictResultBt, self.tablePredict)
        layoutStack2.addRow(dummyBt, layoutDownloadBt)
        # Separador = QSpacerItem(150,50,QSizePolicy.Policy.Expanding)
        # layoutStack2.addItem(Separador)

        self.stack2.setLayout(layoutStack2)


    def loadModel(self):
        self.modelLoadedFname = QFileDialog.getOpenFileName(self, 'Select model', 'c:\\',"data files (*.pkl *.joblib)")
        if self.modelLoadedFname[0] != '':
            self.loadModelTx.setText(f'Model {self.modelLoadedFname[0]} was loaded')
            with open(self.modelLoadedFname[0], 'rb') as f:
                self.modelLoaded = pickle.load(f)
        else:
            QMessageBox.information(self,'Info','Please load a trained model')

    def predictSubmittedData(self):
        if hasattr(self, 'dataForML') and self.dataForML is not None:
            Xtest = self.dataForML.iloc[:, -5:]
            if hasattr(self, 'modelLoaded'):
                self.Ypredict = self.modelLoaded.predict(Xtest)
                self.dataForML['Predicted_results'] = self.Ypredict
                self.modelPredict = TableModel(self.dataForML.round(4))
                self.tablePredict.setModel(self.modelPredict)
                stylesheet = "::section{Background-color:rgb(200,200,200);border-radius:30px;}"
                self.table.horizontalHeader().setStyleSheet(stylesheet)
                self.table.sortByColumn(2, Qt.SortOrder.AscendingOrder)

            else:
                QMessageBox.warning(self, 'Error', 'No model selected')
        else:
            QMessageBox.warning(self, 'Error', 'No data input, please sumbit normalized data in previous tab')

    def downloadPredictedBtClicked(self):
        if hasattr(self, 'dataForML') and self.dataForML.shape[0] > 1:
            saveFileName = QFileDialog.getSaveFileName(self, 'Save file','','Excel files (*.xlsx);; CSV (*.csv);; HTML (*.html)')
            if saveFileName and saveFileName[1] == 'Excel files (*.xlsx)':
                self.dataForML.to_excel(saveFileName[0])
            elif  saveFileName and saveFileName[1] == 'CSV (*.csv)':
                self.dataForML.to_csv(saveFileName[0])
            elif saveFileName and saveFileName[1] == 'HTML (*.html)':
                self.dataForML.to_html(saveFileName[0])
            else:
                pass    
        else:
            QMessageBox.warning(self, 'Info','No results file can be downloaded', QMessageBox.StandardButton.Ok)
    
    def stack3Layout(self):
        layoutStack3 = QHBoxLayout()
        layoutStack3Left = QVBoxLayout()
        self.layoutStack3Right = QVBoxLayout()
        manualAutoBt = QComboBox(self)
        manualAutoBt.setPlaceholderText('Load sample true label')
        manualAutoBt.addItems(['Manually','Auto match in file'])
        manualAutoBt.currentIndexChanged.connect(self.manualAutoItemChanged)
        calSensSpeBt = QPushButton('Calculate Sensitivity and Specificity')
        calSensSpeBt.clicked.connect(self.sesSpeciFig)
        self.sensTextLabel = QLabel("<strong style='font-size:16; color:rgb(29,233,182)'>Sensitivity and specificity</strong>")

        self.tableStack3 = QTableView()
        self.tableStack3.setFixedWidth(400)
        # self.modelStack3 = TableModel()
        # self.tableStack3.setModel(self.modelStack3)
        stylesheet = "::section{Background-color:rgb(200,200,200);border-radius:30px;}"
        self.tableStack3.horizontalHeader().setStyleSheet(stylesheet)
        # self.tableStack3.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        self.sc = MplCanvas(self.layoutStack3Right, dpi=100)
        toolbar = NavigationToolbar(self.sc, self)


        self.layoutStack3Right.addWidget(calSensSpeBt)
        self.layoutStack3Right.addWidget(toolbar)
        self.layoutStack3Right.addWidget(self.sc)
        self.layoutStack3Right.addStretch(1)
        self.layoutStack3Right.addWidget(self.sensTextLabel)
        self.layoutStack3Right.addStretch(5)

        layoutStack3Left.addWidget(manualAutoBt)
        layoutStack3Left.addWidget(self.tableStack3)
        layoutStack3.addItem(layoutStack3Left)
        layoutStack3.addSpacing(10)
        layoutStack3.addItem(self.layoutStack3Right)

        self.stack3.setLayout(layoutStack3)

    def manualSubmitLabel(self):
        self.ManualSubmitLabelWindow = QWidget()
        self.ManualSubmitLabelWindow.setWindowTitle('True label input')
        self.ManualSubmitLabelWindow.setWindowIcon(QIcon('images/icoLK.ico'))
        self.ManualSubmitLabelWindow.move(250,176)
        self.scrolA = QScrollArea(self.ManualSubmitLabelWindow)
        self.scrolA.setWidgetResizable(True)
        self.scrolA.resize(200,800)
        inner = QFrame(self.scrolA)
        self.layoutForm = QFormLayout(inner)
        self.ManualSubmitLabelWindow.resize(200,800)

        if hasattr(self, 'dataForML') and hasattr(self.dataForML,'Predicted_results'):
            self.dfForConfusion = self.dataForML
            # print(self.dfForConfusion['Predicted_rsults'])
            self.dfForConfusionFinal = self.dfForConfusion[['ID','Predicted_results']]
            # print(self.dfForConfusionFinal)
            label0 = QLabel()
            label0.setText('Sample ID')
            label00 = QLabel()
            label00.setText('Patient?')
            self.layoutForm.addRow(label0, label00)
            for item in self.dfForConfusion['ID']:
                label = QLabel()
                checkbox = QCheckBox()
                label.setText(str(item))
                self.layoutForm.addRow(label,checkbox)
            submitBt = QPushButton('Submit')
            submitBt.clicked.connect(self.submitManualBtClicked)
            clearBt = QPushButton('Clear')
            clearBt.clicked.connect(self.clearManualBtClicked)
            self.layoutForm.addRow(submitBt, clearBt)

            self.scrolA.setWidget(inner)
            self.scrolA.show()
            self.ManualSubmitLabelWindow.show()
        else:
            QMessageBox.warning(self, 'Info','Please predict uploaded data first', QMessageBox.StandardButton.Ok)
    
    def submitLabelFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"data files (*.xlsx *.xlsm)")
        if fname[0] != '':
            self.submitedLabelFile = pd.read_excel(fname[0])
            if hasattr(self, 'dataForML') and hasattr(self.dataForML,'Predicted_results') and hasattr(self.submitedLabelFile, 'True_label'):
                self.dfForConfusion = self.dataForML
                # print(self.dfForConfusion['Predicted_rsults'])
                self.dfForConfusionFinal = self.dfForConfusion[['ID','Predicted_results']]
                self.dfForConfusionFinal = pd.merge(self.dfForConfusionFinal, self.submitedLabelFile)
                self.modelStack3 = TableModel(self.dfForConfusionFinal)
                self.tableStack3.setModel(self.modelStack3)
            else:
                QMessageBox.information(self, 'Info', 'Please predict uploaded data first')

    
    def submitManualBtClicked(self):
        self.checkedAsPatient = []
        for i in range(len(self.dfForConfusion['ID'])):
            w = self.layoutForm.itemAt(i+1, QFormLayout.ItemRole(1))
            ww = w.widget()
            text = ww.checkState()
            if text == Qt.CheckState.Checked:
                m = self.layoutForm.itemAt(i+1, QFormLayout.ItemRole(0))
                mm = m.widget()
                text1 = mm.text()
                self.checkedAsPatient.append(text1)
            i+=1
        self.dfForConfusionFinal['True_label'] = 0
        self.dfForConfusionFinal['True_label'][self.dfForConfusionFinal['ID'].astype(str).isin(self.checkedAsPatient)] = 1
        self.dfForConfusionFinal = self.dfForConfusionFinal[self.dfForConfusionFinal['ID'] != 'CAL']
        self.modelStack3 = TableModel(self.dfForConfusionFinal)
        self.tableStack3.setModel(self.modelStack3)
        self.ManualSubmitLabelWindow.close()

    def manualAutoItemChanged(self, i):
        if i == 0:
            self.manualSubmitLabel()
        else:
            self.submitLabelFile()

    
    def clearManualBtClicked(self):
        for i in range(len(self.dfForConfusion['ID'])):
            w = self.layoutForm.itemAt(i+1, QFormLayout.ItemRole(1))
            ww = w.widget()
            ww.setChecked(False)
    
    def sesSpeciFig(self):
        if len(self.cm)>0 and hasattr(self, 'dfForConfusionFinal') and hasattr(self.dfForConfusionFinal, 'True_label'):
            self.sc.axes.collections[0].colorbar.remove()
            self.sc.axes.cla()
            ax = self.sc.axes
            self.cm = confusionMatrix(self.dfForConfusionFinal['True_label'], self.dfForConfusionFinal['Predicted_results'])
            snsh(self.cm, annot=True, fmt='.2g', ax=ax, cmap='BrBG',cbar = True, linewidths=.5)
            ax.set_xlabel('Predicted labels', color='k')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(['Healthy','Patient'])
            ax.yaxis.set_ticklabels(['Healthy','Patient']) 
            self.sens_speci_cal(self.dfForConfusionFinal['True_label'], self.dfForConfusionFinal['Predicted_results'])
            self.sensTextLabel.setText(f"<strong style='font-size:16; color:rgb(29,233,182)'> Sensitivity: {self.sensitivity:.2%}, Specificity: {self.specificity:.2%} </strong>")
            self.sc.draw()
        else:
            if hasattr(self, 'dataForML') and hasattr(self.dfForConfusionFinal, 'True_label') and hasattr(self, 'dfForConfusionFinal'):
                ax = self.sc.axes
                self.cm = confusionMatrix(self.dfForConfusionFinal['True_label'], self.dfForConfusionFinal['Predicted_results'])
                snsh(self.cm, annot=True, fmt='.2g', ax=ax, cmap='BrBG',linewidths=.5)
                ax.set_xlabel('Predicted labels', color='k')
                ax.set_ylabel('True labels')
                ax.set_title('Confusion Matrix')
                ax.xaxis.set_ticklabels(['Healthy','Patient'])
                ax.yaxis.set_ticklabels(['Healthy','Patient']) 
                self.sens_speci_cal(self.dfForConfusionFinal['True_label'], self.dfForConfusionFinal['Predicted_results'])
                self.sensTextLabel.setText(f"<strong style='font-size:16; color:rgb(29,233,182)'> Sensitivity: {self.sensitivity:.2%}, Specificity: {self.specificity:.2%} </strong>")
                self.sc.draw()
            else:
                QMessageBox.information(self, 'Info','Please upload file with correct format', QMessageBox.StandardButton.Ok)
    
    def sens_speci_cal(self, y_test, y_predict):
        tn, fp, fn, tp = confusionMatrix(y_test, y_predict).ravel()
        self.specificity = tn / (tn + fp)
        self.sensitivity = tp /(tp + fn)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window1 = GifTest()
    window1.show()
    window2 = GifTest2()
    window2.show()
    QTest.qWait(3000)
    window1.close()
    window2.close()
    apply_stylesheet(app, theme='dark_teal.xml')
    window = Data_LP()
    window.setWindowIcon(QIcon('images/icoLK.ico'))
    window.show()
    sys.exit(app.exec())