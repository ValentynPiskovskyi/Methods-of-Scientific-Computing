from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, \
     QMessageBox, QLabel, QButtonGroup
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets
from ui.interface import Ui_MainWindow

from scipy.fft import dctn, idctn
from PIL import Image, ExifTags
import numpy

WIN_X = 100 
WIN_Y = 100 
WIN_W = 720 
WIN_H = 770

MIN_IMG_W = 200
MIN_IMG_H = 200
MAX_IMG_W = 450
MAX_IMG_H = 450

RGB_CHANNELS = 3

COLOR_INTERFACE = [68, 84, 106]
COLOR_GRID_MONOCHROME = 150



class Worker(QObject):
    finished = pyqtSignal(object)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        result = self.function(*self.args, **self.kwargs)
        self.finished.emit(result)




class MyWindow(QMainWindow, Ui_MainWindow):

    ### Program Initialization

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Attributes
        self.path = ""
        self.img = numpy.zeros((MIN_IMG_W, MIN_IMG_H, RGB_CHANNELS))
        self.img_cut = numpy.zeros((MIN_IMG_W, MIN_IMG_H, RGB_CHANNELS))
        self.img_cut_grid = numpy.zeros((MIN_IMG_W, MIN_IMG_H, RGB_CHANNELS))
        self.img_transformed = numpy.zeros((MIN_IMG_W, MIN_IMG_H, RGB_CHANNELS))
        self.f_changed = False
        self.blocks = []
        self.N = 0
        self.F = 0
        self.d = 0
        self.isRgb = False
        self.format = "bmp"

        # Set MainWindow's size and position
        self.setGeometry(WIN_X, WIN_Y, WIN_W, WIN_H)

        # Set initial size of the two image visualization windows
        self.labelImageOriginal.setFixedSize(350, 350)
        self.labelImageTransformed.setFixedSize(350, 350)

        self.radioGroup = QButtonGroup(self)

        # Add radio buttons to the group
        self.radioGroup.addButton(self.radioButtonOriginal, id=0)
        self.radioGroup.addButton(self.radioButtonCropped, id=1)
        self.radioGroup.addButton(self.radioButtonCroppedWithGrid, id=2)

        # Connect widgets to the methods handling the corresponding events.
        self.buttonLoadFile.clicked.connect(self.on_click_load_file)
        self.buttonPreview.clicked.connect(self.on_click_preview)
        self.buttonStart.clicked.connect(self.on_click_start)
        self.buttonSave.clicked.connect(self.on_click_save)
        self.lineEditF.textChanged.connect(self.on_text_changed_f)
        self.radioGroup.buttonClicked[int].connect(self.on_radio_group_button_clicked_original)


    ### Image Loading

    def open_image(self):
        self.labelStatus.setText("Loading the image")

        if self.format.lower() == "bmp":
            pil_img = Image.open(self.path)
        else:
            pil_img = self.load_image_correct_orientation(self.path) # Check for correct orientation

        dimensions = len(numpy.array(pil_img).shape)

        if dimensions == 2 : # Graysacale
            self.isRgb = False
            pil_img = pil_img.convert("L")
        elif dimensions == 3: # RGB
            self.isRgb = True
        else:
            self.show_error("Unsupported image format. The image size should be: W x H [x C].")
            return
        
        return pil_img
    

    def callback_image_opened(self, pil_img):
        # Load image and rise numeric precision, setting float64 format. Use correct orientation.
        self.labelStatus.clear()
        self.img = numpy.copy(numpy.array((pil_img), dtype = numpy.float64))
        self.show_image(self.img, self.labelImageOriginal)

        # Change settings of widgets
        self.buttonPreview.setEnabled(True)
        self.buttonSave.setEnabled(False)
        self.radioButtonCropped.setEnabled(False)
        self.radioButtonCroppedWithGrid.setEnabled(False)
        self.labelImageTransformed.clear()


    def on_click_load_file(self):
        self.path, _ = QFileDialog.getOpenFileName()

        if not self.path:
            self.show_error("No image selected")
            return
        
        self.format = self.path.split(".")[-1]
        self.start_thread(self.open_image, self.callback_image_opened)


    # Division of the input image into blocks

    def on_click_preview(self):
        self.F = int(self.lineEditF.text())
        
        # Load image and rise numeric precision, setting float64 format.
        img = numpy.copy(self.img)
        
        # Cut the image to the size of the largest multiple of F possible along either X or Y axis, removing residuals.
        x_nb = int(img.shape[0] / self.F) # Number of blocks along the X axis
        y_nb = int(img.shape[1] / self.F) # Number of blocks along the Y axis
        
        min_nb = min(x_nb, y_nb)

        if self.isRgb:
            self.img_cut = numpy.array(img[: self.F * min_nb, 
                                           : self.F * min_nb, 
                                           :])
        else:
            self.img_cut = numpy.array(img[: self.F * min_nb, 
                                           : self.F * min_nb])

        self.N = self.img_cut.shape[0]
        
        # Add the grid with F x F blocks on the original image and create the list of blocks. 
        self.img_cut_grid, self.blocks = self.add_grid_get_blocks(
            self.img_cut)

        # Display the image with the F x F blocks grid projected onto it.
        self.show_image(self.img_cut, self.labelImageOriginal)
        
        self.buttonPreview.setEnabled(False)
        self.buttonStart.setEnabled(True)
        self.radioButtonCropped.setEnabled(True)
        self.radioButtonCroppedWithGrid.setEnabled(True)
        self.radioButtonCropped.setChecked(True)
        self.labelImageOriginal.setScaledContents(True)


    def add_grid_get_blocks(self, img):
        img_mod = img.copy()
        num_blocks = int(self.N / self.F)

        if self.isRgb:
            grid_color = COLOR_INTERFACE
        else:
            grid_color = COLOR_GRID_MONOCHROME

        blocks = []

        for i in range(num_blocks):
            img_mod[:, self.F * i] = grid_color # Columns
            img_mod[self.F * i, :] = grid_color # Rows
            for j in range(num_blocks):
                blocks.append(img[self.F * i : self.F * (i + 1), 
                                  self.F * j : self.F * (j + 1)])

        return img_mod, blocks


    ### Image transformation

    def transform_image(self):
        self.labelStatus.setText("Transforming the image")
        self.d = int(self.lineEditD.text())

        # Check for consistency of F and d values
        if self.d > (2 * self.F - 2):
            self.show_error("Value d must be between \
                            0 and 2F - 2")
            return

        # Create the list of c coefficients resulting from 
        # the application of DCT 2D on each of the blocks.
        if self.isRgb:
            self.img_transformed = \
                self.transform_rgb(self.blocks)
        else:
            self.img_transformed = \
                self.transform_grayscale(self.blocks)
        
        print("Transformed image: ", self.img_transformed.shape)

        self.show_image(self.img_transformed, self.labelImageTransformed)
        self.buttonSave.setEnabled(True)


    def callback_transform_image(self):
        self.labelStatus.clear()


    def on_click_start(self):
        self.start_thread(self.transform_image, 
                          self.callback_transform_image)


    def transform_grayscale(self, blocks, color = "tab:blue"):
        c_list = []
        for b in blocks: 
            c = numpy.array(dctn(b, norm = "ortho"))
            c_list.append(c)
        
        # Create the list of c coefficients cut according to the specific rule. 
        # Each value in position (i, j) such that i + j >= d is set to 0.
        c_list_modified = []
        for c in c_list:
            c_list_modified.append(self.remove_freq(c))

        # Create the list of frequencies reconstructed with the IDCT 2D.
        ff_list = []
        for i, c in enumerate(c_list_modified):
            ff = numpy.array(idctn(c, norm = "ortho"))
            ff = numpy.vstack(
                [numpy.array([self.limit_number(x) 
                              for x in r]) for r in ff])
            ff_list.append(ff)

        # Assemble the image by putting together the ff matrices.         
        rows = []
        nb = int(self.N / self.F) # Number of blocks along a single axis.

        for i in range(nb):
            row = ff_list[nb * i + 0] # First block within a row.

            for j in range(1, nb):
                # Concatenate the following blocks to the row.
                row = numpy.concatenate((row, 
                                         ff_list[nb * i + j]), 
                                        axis = 1) 
            rows.append(row)

        result = numpy.vstack(rows) # Stack the resulting rows vertically.
        return result


    def transform_rgb(self, blocks):
        channels = []

        # Iterate over RGB channels
        for (ch, color) in zip(range(3), ["tab:red", "tab:green", "tab:blue"]): 
            img = self.transform_grayscale([b[:, :, ch] 
                                            for b in blocks],
                                            color = color)
            channels.append(img)
        
        result = numpy.stack(channels, axis = 2)
        return result


    def remove_freq(self, c):
        c_modified = c.copy()

        for i in range(self.F):
            for j in range(self.F):
                if i + j >= self.d:
                    c_modified[i, j] = 0
        
        return c_modified


    def limit_number(self, x):
        if x > 255:
            return 255
        if x < 0:
            return 0
        return x
    

    ### Image storage
    
    def on_click_save(self):
        filter = ";;".join([f"{fmt.upper()} (*.{fmt})" for fmt in ["bmp", "jpg", "png"]])
        save_path, selected_filter = QFileDialog.getSaveFileName(self, "Save Image", f"image.{self.format}", filter)

        if not save_path:
            return
        
        if '.' not in save_path:
            save_path += f".{self.format}"

        height = self.img_transformed.shape[0]
        width = self.img_transformed.shape[1]
        bytes_per_line = width
        
        qimage = QImage(numpy.uint8(self.img_transformed), width, height, bytes_per_line * (RGB_CHANNELS if self.isRgb else 1), 
                        QtGui.QImage.Format_RGB888 if self.isRgb else QtGui.QImage.Format_Grayscale8)
        qimage.save(save_path)
    

    def on_radio_group_button_clicked_original(self, id) :
        if id == 0:   
            self.show_image(self.img, self.labelImageOriginal)
        elif id == 1:
            self.show_image(self.img_cut, self.labelImageOriginal)
        elif id == 2:
            self.show_image(self.img_cut_grid, self.labelImageOriginal)
            
    
    def on_text_changed_f(self):
        self.buttonPreview.setEnabled(True)
        self.buttonStart.setEnabled(False)


    ### Additional Helper methods

    def show_image(self, img : numpy.ndarray, qlabel : QLabel) : 
        height = img.shape[0]
        width = img.shape[1]
        bytes_per_line = width
        
        qimage = QImage(numpy.uint8(img), width, height, bytes_per_line * (RGB_CHANNELS if self.isRgb else 1), 
                        QtGui.QImage.Format_RGB888 if self.isRgb else QtGui.QImage.Format_Grayscale8)

        qpixmap = QPixmap(qimage)

        qlabel.setPixmap(qpixmap)
        qlabel.setScaledContents(True)
        qlabel.setFixedSize(min(max(img.shape[0], MIN_IMG_W), MAX_IMG_W), 
                                             min(max(img.shape[1], MIN_IMG_H), MAX_IMG_W))
    

    def show_error(self, message) :
        error_message = QMessageBox()
        error_message.setText(message)
        error_message.setIcon(QMessageBox.Critical)
        error_message.setWindowTitle("Error")
        error_message.exec_()
    

    def load_image_correct_orientation(self, path):
        img = Image.open(path)

        # Fix EXIF orientation (if any)
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation, None)

                if orientation_value == 3:
                    img = img.rotate(180, expand=True)
                elif orientation_value == 6:
                    img = img.rotate(270, expand=True)
                elif orientation_value == 8:
                    img = img.rotate(90, expand=True)
        except Exception as e:
            #self.show_error(f"Image with unsupported orientation metadata: {e}")
            print(f"Image with unsupported orientation metadata: {e}")
        
        return img
        

    def start_thread(self, function, callback, *args, **kwargs):
        self.thread = QThread()
        self.worker = Worker(function, *args, **kwargs)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(callback)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()


app = QApplication([])
window = MyWindow()
window.show()
app.exec_()