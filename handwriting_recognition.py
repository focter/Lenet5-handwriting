import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QWidget,
    QGridLayout,
)
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import tensorflow as tf
from PIL import Image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "手写数字识别"
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setMinimumSize(500, 400)
        self.main_widget = QWidget()
        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        self.canvas = Canvas()
        self.canvas.setFixedSize(300, 300)

        self.label = QLabel()
        self.label.setFixedSize(100, 50)
        self.label.setText("识别结果")
        self.label.setStyleSheet("font-size:15px;color:red")

        self.clear_button = QPushButton("清除")
        self.clear_button.setFixedSize(100, 50)
        self.clear_button.clicked.connect(self.canvas.clear)

        self.recognize_button = QPushButton("识别")
        self.recognize_button.setFixedSize(100, 50)
        self.recognize_button.clicked.connect(self.recognize)

        self.main_layout.addWidget(self.canvas, 0, 0, 3, 1)
        self.main_layout.addWidget(self.label, 0, 1)
        self.main_layout.addWidget(self.clear_button, 1, 1)
        self.main_layout.addWidget(self.recognize_button, 2, 1)

        self.model = tf.keras.models.load_model("lenet5_mnist_model")  # 加载模型

    def recognize(self):
        result = self.canvas.recognize(self.model)
        self.label.setText("识别结果: " + str(result))


class Canvas(QLabel):
    x0 = -10
    y0 = -10
    x1 = -10
    y1 = -10

    def __init__(self):
        super(Canvas, self).__init__()
        self.pixmap = QPixmap(300, 300)
        self.pixmap.fill(Qt.white)
        self.Color = Qt.blue
        self.penwidth = 10

    def paintEvent(self, event):
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(self.Color, self.penwidth, Qt.SolidLine))
        painter.drawLine(self.x0, self.y0, self.x1, self.y1)

        label_painter = QPainter(self)
        label_painter.drawPixmap(2, 2, self.pixmap)

    def mousePressEvent(self, event):
        self.x1 = event.x()
        self.y1 = event.y()

    def mouseMoveEvent(self, event):
        self.x0 = self.x1
        self.y0 = self.y1
        self.x1 = event.x()
        self.y1 = event.y()
        self.update()

    def clear(self):
        self.x0 = -10
        self.y0 = -10
        self.x1 = -10
        self.y1 = -10
        self.pixmap.fill(Qt.white)
        self.update()

    def recognize(self, model):
        arr = self.pixmap_to_array(self.pixmap)
        arr = 255 - arr[:, :, 2]
        arr = self.clip_image(arr)
        arr = self.resize_image(arr)
        arr = arr.reshape(1, 28, 28, 1) / 255.0
        predictions = model.predict(arr)
        result = np.argmax(predictions)
        return result

    def pixmap_to_array(self, pixmap):
        size = pixmap.size()
        h = size.height()
        w = size.width()
        channel_count = 4  # assuming RGBA
        image = pixmap.toImage()
        s = image.bits().asstring(h * w * channel_count)
        arr = np.frombuffer(s, dtype=np.uint8).reshape((h, w, channel_count))
        return arr

    def clip_image(self, image):
        # clip non-white margins
        non_empty_columns = np.where(image.min(axis=0) < 255)[0]
        non_empty_rows = np.where(image.min(axis=1) < 255)[0]
        crop_box = (
            min(non_empty_rows),
            max(non_empty_rows),
            min(non_empty_columns),
            max(non_empty_columns),
        )

        image = image[crop_box[0] : crop_box[1] + 1, crop_box[2] : crop_box[3] + 1]
        return image

    def resize_image(self, image, size=(28, 28)):
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(size, Image.ANTIALIAS)
        return np.array(pil_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
