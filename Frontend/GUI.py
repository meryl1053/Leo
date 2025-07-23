from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QStackedWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QSizePolicy,
    QGraphicsOpacityEffect
)
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QRect
from PyQt5.QtGui import QIcon, QFont, QMovie, QColor, QTextCharFormat, QPixmap
from dotenv import dotenv_values
import sys
import os

env_vars = dotenv_values(".env")
Assistantname = env_vars.get("Assistantname") or "Assistant"
curr_dir = os.getcwd()
old_chat_messages = ""
TempDirPath = rf"{curr_dir}/Frontend/Files"
GraphicsDirPath = rf"{curr_dir}/Frontend/Graphics"

def AnswerModifier(Answer):
    return "\n".join([line for line in Answer.split("\n") if line.strip()])

def QueryModifier(Query):
    new_query = Query.lower().strip()
    question_words = ["what", "who", "where", "when", "why", "how", "which", "whose", "whom", "can you", "what's", "where's", "how's"]
    if any(word + " " in new_query for word in question_words):
        new_query = new_query.rstrip(".?!") + "?"
    else:
        new_query = new_query.rstrip(".?!") + "."
    return new_query.capitalize()

def setMicrophoneStatus(Command):
    with open(f"{TempDirPath}/Mic.data", "w", encoding='utf-8') as file:
        file.write(Command)

def GetMicrophoneStatus():
    with open(f"{TempDirPath}/Mic.data", "r", encoding='utf-8') as file:
        return file.read().strip()

def SetAssistantStatus(Status):
    with open(f"{TempDirPath}/Status.data", "w", encoding='utf-8') as file:
        file.write(Status)

def GetAssistantStatus():
    with open(f"{TempDirPath}/Status.data", "r", encoding='utf-8') as file:
        return file.read().strip()

def GetMicButtonInitialized():
    setMicrophoneStatus("False")

def MicButtonClosed():
    setMicrophoneStatus("True")

def GraphicsDirectoryPath(Filename):
    return f'{GraphicsDirPath}/{Filename}'

def TempDirectoryPath(Filename):
    return f'{TempDirPath}/{Filename}'

def ShowTextToScreen(Text):
    with open(f"{TempDirPath}/Responses.data", "w", encoding='utf-8') as file:
        file.write(Text)

modern_style = """
QWidget {
    background-color: #000000;
    color: #E0E0E0;
    font-family: 'Verdana', Tahoma, Geneva, Verdana, sans-serif;
}

QPushButton {
    background-color: transparent;
    color: #fafbfc;
    border: none;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 14px;
    /* Removed unsupported 'transition' property */
}

QPushButton:hover {
    color: #1F6FEB;
}

QPushButton:pressed {
    color: #155BBB;
}

QLabel {
    font-size: 14px;
}

QTextEdit {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #333;
    font-size: 14px;
    color: #EEE;
}

QScrollBar:vertical {
    border: none;
    background: #2C2C2C;
    width: 10px;
    margin: 0px 0px 0px 0px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #4DA1FF;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

QStackedWidget {
    border-radius: 12px;
}
"""

class ChatSection(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #000000;")

        layout = QVBoxLayout(self)
        # Reduced margins to give more space to chat box
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(10)

        self.chat_text_edit = QTextEdit()
        self.chat_text_edit.setReadOnly(True)
        self.chat_text_edit.setTextInteractionFlags(Qt.NoTextInteraction)
        self.chat_text_edit.setFrameStyle(QFrame.NoFrame)
        # Set minimum height for the chat box to make it bigger
        self.chat_text_edit.setMinimumHeight(500)
        # Give the chat box more priority in stretching
        layout.addWidget(self.chat_text_edit, stretch=3)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        format = QTextCharFormat()
        format.setForeground(QColor(Qt.cyan))
        self.chat_text_edit.setCurrentCharFormat(format)

        self.gif_label = QLabel()
        movie = QMovie(GraphicsDirectoryPath("Jarvis.gif"))
        # Reduced GIF size to give more space to chat box
        movie.setScaledSize(QSize(400, 225))
        self.gif_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.gif_label.setMovie(movie)
        movie.start()
        self.gif_label.setStyleSheet("background: transparent;")
        # Set maximum height for gif to prevent it from taking too much space
        self.gif_label.setMaximumHeight(250)
        layout.addWidget(self.gif_label, stretch=1)

        self.label = QLabel("")
        self.label.setStyleSheet("color: #4DA1FF; font-size: 14px; margin-right: 100px; margin-top: -20px;")
        self.label.setAlignment(Qt.AlignRight)
        # Set fixed height for the status label
        self.label.setMaximumHeight(30)
        layout.addWidget(self.label, stretch=0)

        font = QFont("Segoe UI", 12)
        self.chat_text_edit.setFont(font)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.loadMessages)
        self.timer.timeout.connect(self.SpeechRecogText)
        self.timer.start(100)

        self.chat_text_edit.viewport().installEventFilter(self)

    def loadMessages(self):
        global old_chat_messages
        try:
            with open(TempDirectoryPath("Responses.data"), "r", encoding='utf-8') as file:
                messages = file.read()
                if messages and messages != old_chat_messages:
                    self.addMessage(messages, color='#E0E0E0')
                    old_chat_messages = messages
        except FileNotFoundError:
            pass

    def SpeechRecogText(self):
        try:
            with open(TempDirectoryPath("Status.data"), "r", encoding='utf-8') as file:
                self.label.setText(file.read())
        except FileNotFoundError:
            self.label.setText("")

    def addMessage(self, message, color):
        cursor = self.chat_text_edit.textCursor()
        format = QTextCharFormat()
        format.setForeground(QColor(color))
        cursor.setCharFormat(format)
        cursor.insertText(message + "\n")
        self.chat_text_edit.setTextCursor(cursor)

class InitialScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #000000;")

        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 100)

        self.greeting_label = QLabel(f"Hello!, I'm {Assistantname}. I am here to assist you.")
        self.greeting_label.setStyleSheet("""
            color: #FFFFFF;
            font-size: 32px;
            font-weight: 300;
            margin-bottom: 5px;
            margin-top: 150px;
        """)
        self.greeting_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.greeting_label, alignment=Qt.AlignCenter)

        gif_label = QLabel()
        movie = QMovie(GraphicsDirectoryPath("Jarvis.gif"))
        movie.setScaledSize(QSize(840, 472))
        gif_label.setMovie(movie)
        gif_label.setAlignment(Qt.AlignCenter)
        movie.start()
        gif_label.setStyleSheet("background: transparent;")
        layout.addWidget(gif_label, alignment=Qt.AlignCenter)

        self.icon_label = QLabel()
        pixmap = QPixmap(GraphicsDirectoryPath("Mic_on.png")).scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.icon_label.setPixmap(pixmap)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFixedSize(120, 120)
        layout.addWidget(self.icon_label, alignment=Qt.AlignCenter)

        self.label = QLabel("")
        self.label.setStyleSheet("color: #4DA1FF; font-size: 14px; margin-bottom: 0;")
        layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.setFixedHeight(screen_height)
        self.setFixedWidth(screen_width)

        self.toggled = True
        self.toggle_icon()
        self.icon_label.mousePressEvent = self.toggle_icon

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.SpeechRecogText)
        self.timer.start(100)

    def SpeechRecogText(self):
        try:
            with open(TempDirectoryPath("Status.data"), "r", encoding='utf-8') as file:
                self.label.setText(file.read())
        except FileNotFoundError:
            self.label.setText("")

    def load_icon(self, path, width=50, height=50):
        pixmap = QPixmap(path).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.icon_label.setPixmap(pixmap)

    def toggle_icon(self, event=None):
        if self.toggled:
            self.load_icon(GraphicsDirectoryPath("Mic_on.png"))
            GetMicButtonInitialized()
        else:
            self.load_icon(GraphicsDirectoryPath("Mic_off.png"))
            MicButtonClosed()
        self.toggled = not self.toggled

class MessageScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        screen = QApplication.primaryScreen()
        geometry = screen.availableGeometry()
        layout = QVBoxLayout()
        # Reduced top spacing to give more room to chat
        layout.setContentsMargins(0, 10, 0, 0)
        layout.addWidget(ChatSection())
        self.setLayout(layout)
        self.setFixedHeight(geometry.height())
        self.setFixedWidth(geometry.width())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        screen = QApplication.primaryScreen()
        geometry = screen.availableGeometry()

        self.stacked_widget = QStackedWidget()
        self.initial_screen = InitialScreen()
        self.message_screen = MessageScreen()

        self.stacked_widget.addWidget(self.initial_screen)
        self.stacked_widget.addWidget(self.message_screen)

        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.setSpacing(10)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        home_btn = QPushButton("Home")
        home_btn.setIcon(QIcon(GraphicsDirectoryPath("Home.png")))
        home_btn.clicked.connect(lambda: self.animateTransition(0))

        chat_btn = QPushButton("Chat")
        chat_btn.setIcon(QIcon(GraphicsDirectoryPath("Chats.png")))
        chat_btn.clicked.connect(lambda: self.animateTransition(1))

        button_layout.addWidget(home_btn)
        button_layout.addWidget(chat_btn)
        button_layout.addStretch()

        container_layout.addLayout(button_layout)
        container_layout.addWidget(self.stacked_widget)

        container.setLayout(container_layout)

        self.setGeometry(geometry)
        self.setStyleSheet(modern_style)
        self.setCentralWidget(container)

        self.showFullScreen()

    def animateTransition(self, target_index):
        current_index = self.stacked_widget.currentIndex()
        if current_index == target_index:
            return

        current_widget = self.stacked_widget.currentWidget()
        next_widget = self.stacked_widget.widget(target_index)

        # Set up opacity effects
        if not current_widget.graphicsEffect():
            current_opacity = QGraphicsOpacityEffect(current_widget)
            current_widget.setGraphicsEffect(current_opacity)
        else:
            current_opacity = current_widget.graphicsEffect()

        if not next_widget.graphicsEffect():
            next_opacity = QGraphicsOpacityEffect(next_widget)
            next_widget.setGraphicsEffect(next_opacity)
        else:
            next_opacity = next_widget.graphicsEffect()

        current_opacity.setOpacity(1)
        next_opacity.setOpacity(0)

        next_widget.setVisible(True)

        # Fade animations
        fade_out = QPropertyAnimation(current_opacity, b"opacity")
        fade_out.setDuration(500)
        fade_out.setStartValue(1)
        fade_out.setEndValue(0)
        fade_out.setEasingCurve(QEasingCurve.InOutQuad)

        fade_in = QPropertyAnimation(next_opacity, b"opacity")
        fade_in.setDuration(500)
        fade_in.setStartValue(0)
        fade_in.setEndValue(1)
        fade_in.setEasingCurve(QEasingCurve.InOutQuad)

        def on_fade_finished():
            self.stacked_widget.setCurrentIndex(target_index)
            current_widget.hide()
            current_opacity.setOpacity(1)
            next_opacity.setOpacity(1)

        anim_group = QParallelAnimationGroup()
        anim_group.addAnimation(fade_out)
        anim_group.addAnimation(fade_in)
        anim_group.finished.connect(on_fade_finished)
        anim_group.start()

        self.anim_group = anim_group  # Prevent garbage collection

def GraphicalUserInterface():
    # Set attributes before creating QApplication instance
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    GraphicalUserInterface()
