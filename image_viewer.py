import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGraphicsView,
    QGraphicsScene,
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QIntValidator
from PyQt5.QtCore import Qt, QRectF


class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor(211, 211, 211))  # Light Gray
        self.setFrameShape(QGraphicsView.NoFrame)
        self.pixmap_item = None

    def set_image(self, pixmap):
        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.1
        if event.angleDelta().y() < 0:
            factor = 0.9
        self.scale(factor, factor)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.image_viewer = ImageViewer()
        self.layout.addWidget(self.image_viewer)

        self.info_layout = QHBoxLayout()
        self.entity_name_label = QLabel("Entity Name: ")
        self.entity_value_label = QLabel("Entity Value: ")
        self.info_layout.addWidget(self.entity_name_label)
        self.info_layout.addWidget(self.entity_value_label)
        self.layout.addLayout(self.info_layout)

        self.navigation_layout = QHBoxLayout()

        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.load_previous_image)
        self.navigation_layout.addWidget(self.back_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.load_next_image)
        self.navigation_layout.addWidget(self.next_button)

        self.goto_layout = QHBoxLayout()
        self.goto_label = QLabel("Go to index:")
        self.goto_input = QLineEdit()
        self.goto_input.setValidator(
            QIntValidator(0, 9999999)
        )  # Adjust max value as needed
        self.goto_button = QPushButton("Go")
        self.goto_button.clicked.connect(self.goto_index)
        self.goto_layout.addWidget(self.goto_label)
        self.goto_layout.addWidget(self.goto_input)
        self.goto_layout.addWidget(self.goto_button)

        self.navigation_layout.addLayout(self.goto_layout)
        self.layout.addLayout(self.navigation_layout)

        self.df = pd.read_csv("./dataset/train.csv")
        self.current_index = 0
        self.load_image()

    def load_image(self):
        if 0 <= self.current_index < len(self.df):
            row = self.df.iloc[self.current_index]
            image_filename = os.path.basename(row["image_link"])
            image_path = os.path.join("./dataset/train", image_filename)

            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                self.image_viewer.set_image(pixmap)
                self.entity_name_label.setText(f"Entity Name: {row['entity_name']}")
                self.entity_value_label.setText(f"Entity Value: {row['entity_value']}")
                self.setWindowTitle(f"Image Viewer - Index: {self.current_index}")
            else:
                print(f"Image not found: {image_path}")
        else:
            print(f"Invalid index: {self.current_index}")

    def load_next_image(self):
        self.current_index = (self.current_index + 1) % len(self.df)
        self.load_image()

    def load_previous_image(self):
        self.current_index = (self.current_index - 1) % len(self.df)
        self.load_image()

    def goto_index(self):
        try:
            index = int(self.goto_input.text())
            if 0 <= index < len(self.df):
                self.current_index = index
                self.load_image()
            else:
                print(f"Index out of range: {index}")
        except ValueError:
            print("Please enter a valid integer index")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
