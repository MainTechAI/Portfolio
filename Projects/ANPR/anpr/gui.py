from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget, QFileDialog, \
    QMessageBox, QAction, QVBoxLayout, QToolBar, QWidget, QLabel
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from yolo import run_detection
import cv2
import cfg
import subprocess
import sys


class External(QThread):
    countChanged = pyqtSignal()

    def run(self):
        run_detection()
        self.countChanged.emit()


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Application'
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon('resources\\1.png'))
        self.setFixedSize(self.width, self.height)
        self.center()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.lay = QVBoxLayout(self.central_widget)
        self.label = QLabel(self)

        self.toolbar = QToolBar(self)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        self.toolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.toolbar.setMovable(False)

        self.openAct = QAction(QIcon('resources\\2.png'), '', self)
        self.openAct.triggered.connect(self.on_click_openFileName)
        self.openAct.setStatusTip('Choose image for recognition')
        self.toolbar.addAction(self.openAct)

        self.runAct = QAction(QIcon('resources\\3.png'), '', self)
        self.runAct.triggered.connect(self.onButtonClick)
        self.runAct.setStatusTip('Start recognition')
        self.runAct.setEnabled(False)
        self.toolbar.addAction(self.runAct)

        self.runSave = QAction(QIcon('resources\\4.png'), '', self)
        self.runSave.triggered.connect(self.on_click_save)
        self.runSave.setStatusTip('Save image')
        self.runSave.setEnabled(False)
        self.toolbar.addAction(self.runSave)

        self.runCB = QAction(QIcon(''), '', self)
        self.runCB.triggered.connect(self.saveToClipboard)
        self.runCB.setEnabled(False)
        self.runCB.setStatusTip('Save to clipboard')
        self.runCB.setIconText("_---__ --")
        self.toolbar.addAction(self.runCB)

        self.statusBar()
        self.show()

    def onButtonClick(self):
        if (cfg.running == False):
            cfg.running = True
            self.runAct.setEnabled(True)
            self.openAct.setEnabled(False)
            self.runAct.setIcon(QIcon('resources\\31.png'))
            self.calc = External()
            self.calc.countChanged.connect(self.onCountChanged)
            self.calc.start()

    @pyqtSlot()
    def onCountChanged(self):
        image = cv2.imread('resources\\TEMP\\detected.jpg', cv2.IMREAD_UNCHANGED)
        k = 1.0
        if image.shape[0] > self.height or image.shape[1] > self.width:
            while image.shape[0] * k > self.height - 50 or image.shape[1] * k > self.width - 50:
                k = k - 0.01
            k = k + 0.01
        elif image.shape[0] < self.height or image.shape[1] < self.width:
            while image.shape[0] * k < self.height - 50 or image.shape[1] * k < self.width - 50:
                k = k + 0.01
            k = k - 0.01

        dim = (int(image.shape[1] * k), int(image.shape[0] * k))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite('resources\\TEMP\\resized.jpg', resized)
        pixmap = QPixmap('resources\\TEMP\\resized.jpg')
        self.label.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())
        self.lay.addWidget(self.label)
        self.runSave.setEnabled(True)
        self.runCB.setEnabled(True)
        self.runCB.setIconText(cfg.clipboard)
        self.runAct.setIcon(QIcon('resources\\3.png'))
        self.runAct.setEnabled(False)
        self.openAct.setEnabled(True)
        cfg.running = False

    @pyqtSlot()
    def on_click_openFileName(self):
        cfg.str_file = ""
        cfg.str_file = self.openFileNameDialog()
        if cfg.str_file:
            self.runAct.setEnabled(True)
            self.runSave.setEnabled(False)
            self.runCB.setEnabled(False)
            self.runCB.setIconText("_---__ --")

            image = cv2.imread(cfg.str_file, cv2.IMREAD_UNCHANGED)

            k = 1.0
            if image.shape[0] > self.height or image.shape[1] > self.width:
                while image.shape[0] * k > self.height - 50 or image.shape[1] * k > self.width - 50:
                    k = k - 0.01
                k = k + 0.01
            elif image.shape[0] < self.height or image.shape[1] < self.width:
                while image.shape[0] * k < self.height - 50 or image.shape[1] * k < self.width - 50:
                    k = k + 0.01
                k = k - 0.01

            dim = (int(image.shape[1] * k), int(image.shape[0] * k))
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            cv2.imwrite('resources\\TEMP\\resized.jpg', resized)
            pixmap = QPixmap('resources\\TEMP\\resized.jpg')
            self.label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
            self.lay.addWidget(self.label)

    @pyqtSlot()
    def saveToClipboard(self):
        txt = cfg.clipboard
        subprocess.run(['clip.exe'], input=txt.strip().encode('utf-16'), check=True)

    @pyqtSlot()
    def on_click_save(self):
        str_save = self.saveFileDialog()
        cfg.save_path = ''
        cfg.save_path = str_save
        if str_save:
            for_save = cv2.imread('resources\\TEMP\\detected.jpg')
            print(str_save)
            if str_save.endswith('.jpg'):
                cv2.imwrite(str_save, for_save)
            else:
                cv2.imwrite(str_save + '.jpg', for_save)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Quit', "Are you sure you want to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open", "", "Image Files (*.jpg)", options=options)
        if fileName:
            return (fileName)

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save", "image.jpg", "Image Files (*.jpg)",
                                                  options=options)
        if fileName:
            return (fileName)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = App()
    sys.exit(app.exec_())
