from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFrame, QGraphicsOpacityEffect, QSizePolicy,
    QGraphicsDropShadowEffect, QStackedWidget, QTextEdit, QScrollArea,
    QGridLayout, QListWidget, QListWidgetItem, QSplitter, QFileDialog,
    QProgressBar, QSpacerItem, QLineEdit
)
from PyQt5.QtCore import (
    Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, 
    QParallelAnimationGroup, QRect, QThread, pyqtSignal, QPoint,
    QSequentialAnimationGroup
)
from PyQt5.QtGui import (
    QIcon, QFont, QMovie, QColor, QTextCharFormat, QPixmap, 
    QPainter, QPainterPath, QLinearGradient, QPen, QBrush, QPalette,
    QRadialGradient, QFontDatabase, QTransform
)
import sys
import os
import math
import tempfile

# Try to import optional dependencies
try:
    from dotenv import dotenv_values
    env_vars = dotenv_values(".env")
    DOTENV_AVAILABLE = True
except ImportError:
    print("Warning: python-dotenv not available. Using default settings.")
    env_vars = {}
    DOTENV_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Warning: speech_recognition not available. Voice features will be disabled.")
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

try:
    from Backend.TextToSpeech import TextToSpeech
    TTS_AVAILABLE = True
except ImportError:
    print("Warning: TextToSpeech backend not available. Using print fallback.")
    TTS_AVAILABLE = False
    def TextToSpeech(text):
        print(f"[TTS]: {text}")

# Environment setup
Assistantname = env_vars.get("Assistantname", "LEO")
curr_dir = os.getcwd()
FrontendDir = os.path.join(curr_dir, "Frontend")
TempDirPath = os.path.join(FrontendDir, "Files")
GraphicsDirPath = os.path.join(FrontendDir, "Graphics")
old_chat_messages = ""

class ChatInputBar(QWidget):
    """Elegant chat input bar that slides up from bottom"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.is_visible = False
        self.initUI()
        
        # Start hidden
        self.hide()
    
    def initUI(self):
        # Main container with rounded background
        self.setStyleSheet("""
            QWidget {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 25px;
                border: 2px solid rgba(79, 172, 254, 0.3);
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)
        
        # Chat input field
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message here... 💬")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background: rgba(245, 247, 250, 0.8);
                border: 2px solid rgba(79, 172, 254, 0.2);
                border-radius: 20px;
                padding: 12px 18px;
                font-size: 16px;
                color: #2d3748;
                font-weight: 400;
            }
            QLineEdit:focus {
                border: 2px solid rgba(79, 172, 254, 0.6);
                background: rgba(245, 247, 250, 1.0);
            }
        """)
        self.chat_input.setMinimumHeight(45)
        
        # Send button
        self.send_btn = QPushButton("🚀")
        self.send_btn.setFixedSize(50, 45)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 rgba(79, 172, 254, 0.8),
                                           stop:1 rgba(0, 242, 254, 0.8));
                border: none;
                border-radius: 22px;
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 rgba(79, 172, 254, 1.0),
                                           stop:1 rgba(0, 242, 254, 1.0));
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 rgba(60, 140, 220, 1.0),
                                           stop:1 rgba(0, 200, 220, 1.0));
            }
        """)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        
        # Close button
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(35, 35)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(239, 68, 68, 0.8);
                border: none;
                border-radius: 17px;
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(239, 68, 68, 1.0);
            }
        """)
        self.close_btn.setCursor(Qt.PointingHandCursor)
        
        # Connect signals
        self.send_btn.clicked.connect(self.send_message)
        self.close_btn.clicked.connect(self.hide_chat)
        self.chat_input.returnPressed.connect(self.send_message)
        
        # Add to layout
        layout.addWidget(self.chat_input)
        layout.addWidget(self.send_btn)
        layout.addWidget(self.close_btn)
        
        self.setLayout(layout)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 10)
        self.setGraphicsEffect(shadow)
    
    def show_chat(self):
        """Show chat bar with slide animation"""
        if self.is_visible:
            return
            
        self.is_visible = True
        
        # Position at bottom of parent
        if self.parent_widget:
            parent_rect = self.parent_widget.rect()
            self.setFixedSize(min(600, parent_rect.width() - 100), 75)
            
            # Start position (below screen)
            start_x = (parent_rect.width() - self.width()) // 2
            start_y = parent_rect.height()
            
            # End position (visible at bottom)
            end_x = start_x
            end_y = parent_rect.height() - self.height() - 30
            
            # Set initial position and show
            self.move(start_x, start_y)
            self.show()
            
            # Animate slide up
            self.slide_animation = QPropertyAnimation(self, b"pos")
            self.slide_animation.setDuration(300)
            self.slide_animation.setStartValue(QPoint(start_x, start_y))
            self.slide_animation.setEndValue(QPoint(end_x, end_y))
            self.slide_animation.setEasingCurve(QEasingCurve.OutCubic)
            self.slide_animation.start()
            
            # Focus on input
            self.chat_input.setFocus()
    
    def hide_chat(self):
        """Hide chat bar with slide animation"""
        if not self.is_visible:
            return
            
        self.is_visible = False
        
        # Animate slide down
        current_pos = self.pos()
        end_y = self.parent_widget.rect().height()
        
        self.slide_animation = QPropertyAnimation(self, b"pos")
        self.slide_animation.setDuration(250)
        self.slide_animation.setStartValue(current_pos)
        self.slide_animation.setEndValue(QPoint(current_pos.x(), end_y))
        self.slide_animation.setEasingCurve(QEasingCurve.InCubic)
        self.slide_animation.finished.connect(self.hide)
        self.slide_animation.start()
        
        # Clear input
        self.chat_input.clear()
    
    def send_message(self):
        """Handle sending message"""
        message = self.chat_input.text().strip()
        if message:
            # Process the message
            if self.parent_widget:
                self.parent_widget.process_chat_message(message)
            
            # Clear input
            self.chat_input.clear()
            
            # Hide chat bar after sending
            QTimer.singleShot(100, self.hide_chat)
    
    def resizeEvent(self, event):
        """Handle resize to maintain position"""
        super().resizeEvent(event)
        if self.is_visible and self.parent_widget:
            # Reposition when parent resizes
            parent_rect = self.parent_widget.rect()
            new_x = (parent_rect.width() - self.width()) // 2
            new_y = parent_rect.height() - self.height() - 30
            self.move(new_x, new_y)

class AnimatedRobot(QWidget):
    """Custom animated robot widget that exactly matches the HTML version"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.float_offset = 0
        self.wobble_angle = 0
        self.scale_factor = 1.0
        self.hover_scale = 1.0
        self.brightness = 1.0
        self.rotation_y = 0
        self.is_hovered = False
        self.click_animation = False
        
        # Animation timers
        self.float_timer = QTimer()
        self.float_timer.timeout.connect(self.update_float)
        self.float_timer.start(50)  # ~20 FPS
        
        self.wobble_timer = QTimer()
        self.wobble_timer.timeout.connect(self.update_wobble)
        self.wobble_timer.start(40)  # ~25 FPS
        
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.update_rotation)
        self.rotation_timer.start(100)  # ~10 FPS for gentle rotation
        
        self.setMinimumSize(260, 260)
        self.setMaximumSize(260, 260)
        self.setCursor(Qt.PointingHandCursor)
        
        # Load robot image
        self.robot_pixmap = None
        self.load_robot_image()
    
    def load_robot_image(self):
        """Load the robot image"""
        robot_path = "/Users/meryl10/Downloads/Robot2.png"
        if os.path.exists(robot_path):
            self.robot_pixmap = QPixmap(robot_path)
        else:
            # Create a fallback robot emoji pixmap
            self.robot_pixmap = self.create_emoji_pixmap("🤖", 220)
    
    def create_emoji_pixmap(self, emoji, size):
        """Create a pixmap from emoji"""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        font = QFont()
        font.setPointSize(size // 4)
        painter.setFont(font)
        painter.setPen(QColor(100, 150, 255))
        
        painter.drawText(pixmap.rect(), Qt.AlignCenter, emoji)
        painter.end()
        
        return pixmap
    
    def update_float(self):
        """Update floating animation (matches CSS gentleFloat)"""
        self.float_offset += 0.05  # Slower for gentle effect
        self.update()
    
    def update_wobble(self):
        """Update wobble animation (matches CSS robotWobble)"""
        self.wobble_angle += 0.08  # Matches 2.5s cycle
        self.update()
    
    def update_rotation(self):
        """Update gentle rotation (matches CSS gentleRotate)"""
        self.angle += 0.02  # Very slow for 12s cycle
        self.update()
    
    def enterEvent(self, event):
        """Mouse enter event - start hover animation"""
        self.is_hovered = True
        self.start_hover_animation()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Mouse leave event - end hover animation"""
        self.is_hovered = False
        self.start_unhover_animation()
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """Mouse press event - trigger click animation"""
        if event.button() == Qt.LeftButton:
            self.click_animation = True
            # Call parent's robot_clicked method
            if hasattr(self.parent(), 'robot_clicked'):
                self.parent().robot_clicked()
            
            # Reset click animation after short delay
            QTimer.singleShot(200, lambda: setattr(self, 'click_animation', False))
        super().mousePressEvent(event)
    
    def start_hover_animation(self):
        """Start hover scale animation"""
        pass  # Handle in paintEvent for simplicity
    
    def start_unhover_animation(self):
        """Start unhover scale animation"""
        pass  # Handle in paintEvent for simplicity
    
    def paintEvent(self, event):
        if not self.robot_pixmap:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Calculate animation values
        
        # Gentle float (matches gentleFloat keyframes)
        float_y = -12 * math.sin(self.float_offset * 2 * math.pi / 4)  # 4 second cycle
        
        # Wobble animation (matches robotWobble keyframes)
        wobble_progress = self.wobble_angle % (2 * math.pi)
        wobble_scale = 1.0 + 0.05 * math.sin(wobble_progress)  # Scale 1.0 to 1.05
        wobble_rotation = 2 * math.sin(wobble_progress * 2)  # -2 to +2 degrees
        wobble_y = -15 * (math.sin(wobble_progress) ** 2)  # 0 to -15px
        
        # Gentle rotation (matches gentleRotate keyframes)
        rotation_progress = (self.angle % (2 * math.pi)) / (2 * math.pi)  # 0 to 1
        if rotation_progress < 0.25:
            gentle_rotation = 5 * (rotation_progress / 0.25)  # 0 to 5 degrees
        elif rotation_progress < 0.5:
            gentle_rotation = 5 * (1 - (rotation_progress - 0.25) / 0.25)  # 5 to 0 degrees
        elif rotation_progress < 0.75:
            gentle_rotation = -5 * ((rotation_progress - 0.5) / 0.25)  # 0 to -5 degrees
        else:
            gentle_rotation = -5 * (1 - (rotation_progress - 0.75) / 0.25)  # -5 to 0 degrees
        
        # Hover effects
        hover_scale = 1.05 if self.is_hovered else 1.0
        hover_rotation_z = 5 if self.is_hovered else 0
        hover_y_offset = -10 if self.is_hovered else 0
        brightness_factor = 1.2 if self.is_hovered else 1.05
        
        # Click effect
        click_scale = 0.95 if self.click_animation else 1.0
        
        # Calculate final transforms
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        total_scale = wobble_scale * hover_scale * click_scale
        total_y_offset = float_y + wobble_y + hover_y_offset
        total_rotation = wobble_rotation + hover_rotation_z
        
        # Create transform
        transform = QTransform()
        transform.translate(center_x, center_y + total_y_offset)
        transform.scale(total_scale, total_scale)
        transform.rotate(total_rotation)
        transform.translate(-110, -110)  # Half of image size (220/2)
        
        painter.setTransform(transform)
        
        # Apply brightness effect (simulated with opacity)
        painter.setOpacity(min(1.0, brightness_factor / 1.2))
        
        # Draw the robot image
        painter.drawPixmap(0, 0, 220, 220, self.robot_pixmap)
        
        # Draw status indicator (green dot)
        painter.resetTransform()
        painter.setOpacity(1.0)
        
        # Status indicator position (top-right of robot)
        status_x = center_x + 80
        status_y = center_y - 80 + total_y_offset
        
        # Pulsing animation for status
        pulse = 0.8 + 0.2 * math.sin(self.float_offset * 3)
        status_size = 16 * pulse
        
        # White border
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(status_x - status_size/2 - 3), int(status_y - status_size/2 - 3), 
                          int(status_size + 6), int(status_size + 6))
        
        # Green center
        painter.setBrush(QBrush(QColor(72, 187, 120)))  # #48bb78
        painter.drawEllipse(int(status_x - status_size/2), int(status_y - status_size/2), 
                          int(status_size), int(status_size))

class ElegantButton(QPushButton):
    """Custom button that matches the HTML control icons"""
    
    def __init__(self, icon_text, parent=None, is_secondary=False):
        super().__init__(icon_text, parent)
        self.icon_text = icon_text
        self.is_secondary = is_secondary
        self.hover_offset = 0
        self.hover_scale = 1.0
        self.is_hovered = False
        self.press_scale = 1.0
        
        self.setFixedSize(60, 60)
        self.setCursor(Qt.PointingHandCursor)
        
        # Remove default button styling
        self.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                font-size: 32px;
            }
        """)
        
        # Setup shadow effect
        self.shadow_effect = QGraphicsDropShadowEffect()
        self.shadow_effect.setBlurRadius(25)
        self.shadow_effect.setColor(QColor(0, 0, 0, 20))  # rgba(0, 0, 0, 0.08)
        self.shadow_effect.setOffset(0, 8)
        self.setGraphicsEffect(self.shadow_effect)
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(16)  # ~60 FPS
        
        self.target_hover_offset = 0
        self.target_hover_scale = 1.0
        
    def enterEvent(self, event):
        """Mouse enter - start hover animation"""
        self.is_hovered = True
        self.target_hover_offset = -8
        self.target_hover_scale = 1.1
        
        # Update shadow for hover
        if self.is_secondary:
            self.shadow_effect.setColor(QColor(168, 237, 234, 76))  # rgba(168, 237, 234, 0.3)
        else:
            self.shadow_effect.setColor(QColor(102, 126, 234, 64))  # rgba(102, 126, 234, 0.25)
        self.shadow_effect.setBlurRadius(40)
        self.shadow_effect.setOffset(0, 15)
        
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Mouse leave - end hover animation"""
        self.is_hovered = False
        self.target_hover_offset = 0
        self.target_hover_scale = 1.0
        
        # Reset shadow
        self.shadow_effect.setColor(QColor(0, 0, 0, 20))
        self.shadow_effect.setBlurRadius(25)
        self.shadow_effect.setOffset(0, 8)
        
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """Mouse press - scale down"""
        if event.button() == Qt.LeftButton:
            self.press_scale = 0.95
            self.target_hover_offset = -4
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Mouse release - scale back up"""
        self.press_scale = 1.0
        if self.is_hovered:
            self.target_hover_offset = -8
        super().mouseReleaseEvent(event)
    
    def update_animation(self):
        """Update hover animation"""
        # Smooth interpolation
        self.hover_offset += (self.target_hover_offset - self.hover_offset) * 0.2
        self.hover_scale += (self.target_hover_scale - self.hover_scale) * 0.2
        
        # Move the button
        current_pos = self.pos()
        base_y = getattr(self, 'base_y', current_pos.y())
        if not hasattr(self, 'base_y'):
            self.base_y = current_pos.y()
        
        new_y = self.base_y + self.hover_offset
        self.move(current_pos.x(), int(new_y))
        
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate center
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = 30
        
        # Create gradient background
        if self.is_secondary:
            gradient = QLinearGradient(0, 0, self.width(), self.height())
            gradient.setColorAt(0, QColor(168, 237, 234, 76))  # rgba(168, 237, 234, 0.3)
            gradient.setColorAt(1, QColor(254, 214, 227, 76))  # rgba(254, 214, 227, 0.3)
        else:
            gradient = QLinearGradient(0, 0, self.width(), self.height())
            gradient.setColorAt(0, QColor(255, 255, 255, 204))  # rgba(255, 255, 255, 0.8)
            gradient.setColorAt(1, QColor(255, 255, 255, 102))  # rgba(255, 255, 255, 0.4)
        
        # Draw background circle
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 76), 2))  # rgba(255, 255, 255, 0.3)
        
        # Apply scale
        scale = self.hover_scale * self.press_scale
        scaled_radius = radius * scale
        
        painter.drawEllipse(int(center_x - scaled_radius), int(center_y - scaled_radius), 
                          int(scaled_radius * 2), int(scaled_radius * 2))
        
        # Draw icon text
        painter.setPen(QColor(77, 81, 86))  # Dark text color
        font = painter.font()
        font.setPointSize(int(20 * scale))
        painter.setFont(font)
        
        painter.drawText(self.rect(), Qt.AlignCenter, self.icon_text)

class ElegantMainWidget(QWidget):
    """Main widget that replicates the HTML design exactly"""
    
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.status_messages = [
            "Ready to assist you",
            "How can I help today?",
            "AI systems online",
            "Standing by...",
            "Waiting for your command"
        ]
        self.current_message_index = 0
        
        self.initUI()
        self.setup_animations()
        
        # Create chat input bar
        self.chat_bar = ChatInputBar(self)
    
    def initUI(self):
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 45, 40, 45)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignCenter)
        
        # Title
        self.title = QLabel("Hi I'm Leo")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("""
            QLabel {
                font-size: 36px;
                font-weight: 300;
                color: #2d3748;
                letter-spacing: -1px;
                margin-bottom: 6px;
            }
        """)
        layout.addWidget(self.title)
        
        # Subtitle
        self.subtitle = QLabel("Your intelligent companion")
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #718096;
                font-weight: 400;
                margin-bottom: 30px;
            }
        """)
        layout.addWidget(self.subtitle)
        
        # Add spacing
        layout.addSpacing(15)
        
        # Robot container
        robot_container = QWidget()
        robot_container.setFixedHeight(280)
        robot_layout = QHBoxLayout(robot_container)
        robot_layout.setAlignment(Qt.AlignCenter)
        
        self.robot = AnimatedRobot()
        robot_layout.addWidget(self.robot)
        
        layout.addWidget(robot_container)
        
        # Status text
        self.status_text = QLabel("Ready to assist you")
        self.status_text.setAlignment(Qt.AlignCenter)
        self.status_text.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #4a5568;
                font-weight: 400;
                margin: 25px 0;
            }
        """)
        layout.addWidget(self.status_text)
        
        # Controls
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setAlignment(Qt.AlignCenter)
        controls_layout.setSpacing(30)
        
        # Create control buttons
        self.voice_btn = ElegantButton("🎤")
        self.voice_btn.clicked.connect(self.activate_voice)
        
        self.chat_btn = ElegantButton("💬", is_secondary=True)
        self.chat_btn.clicked.connect(self.open_chat)
        
        self.file_btn = ElegantButton("📁", is_secondary=True)
        self.file_btn.clicked.connect(self.upload_file)
        
        self.help_btn = ElegantButton("❓")
        self.help_btn.clicked.connect(self.show_help)
        
        controls_layout.addWidget(self.voice_btn)
        controls_layout.addWidget(self.chat_btn)
        controls_layout.addWidget(self.file_btn)
        controls_layout.addWidget(self.help_btn)
        
        layout.addWidget(controls_container)
        
        # Version info
        self.version_label = QLabel("v2.1")
        self.version_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.version_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                color: #a0aec0;
                font-weight: 300;
                position: absolute;
                bottom: 16px;
                right: 20px;
            }
        """)
        
        # Add version to a container for positioning
        version_container = QWidget()
        version_layout = QHBoxLayout(version_container)
        version_layout.addStretch()
        version_layout.addWidget(self.version_label)
        version_layout.setContentsMargins(0, 20, 0, 0)
        
        layout.addWidget(version_container)
        
        self.setLayout(layout)
    
    def setup_animations(self):
        """Setup status text rotation and other animations"""
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.rotate_status_message)
        self.status_timer.start(4000)  # 4 seconds like HTML
    
    def rotate_status_message(self):
        """Rotate status messages like in HTML"""
        # Fade out
        self.fade_out_status()
        
        # Change text after fade
        QTimer.singleShot(200, self.change_status_text)
    
    def fade_out_status(self):
        """Fade out status text"""
        self.status_text.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #4a5568;
                font-weight: 400;
                margin: 25px 0;
                opacity: 0.5;
            }
        """)
    
    def change_status_text(self):
        """Change status text and fade back in"""
    def change_status_text(self):
        """Change status text and fade back in"""
        self.current_message_index = (self.current_message_index + 1) % len(self.status_messages)
        self.status_text.setText(self.status_messages[self.current_message_index])
        
        # Fade back in
        self.status_text.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #4a5568;
                font-weight: 400;
                margin: 25px 0;
                opacity: 0.8;
            }
        """)
    
    def robot_clicked(self):
        """Handle robot click"""
        self.update_status("Hello! I'm your AI assistant", "#667eea")
        QTimer.singleShot(2500, lambda: self.update_status("Ready to assist you", "#4a5568"))
    
    def activate_voice(self):
        """Handle voice activation - stay on current page"""
        self.update_status("🎤 Listening...", "#667eea")
        
        # Start voice recognition without switching screens
        if SPEECH_RECOGNITION_AVAILABLE:
            if TTS_AVAILABLE:
                TextToSpeech("I'm listening...")
            
            # Use a timer to simulate async voice recognition
            QTimer.singleShot(100, self.start_voice_recognition)
        else:
            self.update_status("⚠️ Voice recognition not available", "#e53e3e")
            QTimer.singleShot(2000, lambda: self.update_status("Ready to assist you", "#4a5568"))
    
    def start_voice_recognition(self):
        """Start the actual voice recognition process"""
        try:
            query = Listen()  # This will handle the actual listening
            
            if query:
                self.update_status(f"🎤 Heard: '{query}'", "#48bb78")
                # Process the voice command here or pass it to backend
                if TTS_AVAILABLE:
                    TextToSpeech(f"I heard you say: {query}")
                
                # Reset status after processing
                QTimer.singleShot(3000, lambda: self.update_status("Ready to assist you", "#4a5568"))
            else:
                self.update_status("🎤 No speech detected", "#ed8936")
                QTimer.singleShot(2000, lambda: self.update_status("Ready to assist you", "#4a5568"))
                
        except Exception as e:
            self.update_status("🎤 Voice recognition error", "#e53e3e")
            QTimer.singleShot(2000, lambda: self.update_status("Ready to assist you", "#4a5568"))
            print(f"Voice recognition error: {e}")
    
    def open_chat(self):
        """Handle chat opening - show chat input bar"""
        self.update_status("💬 Chat ready", "#764ba2")
        
        # Show the chat input bar
        self.chat_bar.show_chat()
        
        if TTS_AVAILABLE:
            TextToSpeech("Chat is ready. What would you like to talk about?")
    
    def process_chat_message(self, message):
        """Process the chat message"""
        self.update_status(f"💬 You said: '{message}'", "#48bb78")
        
        # Here you can integrate with your existing chat processing
        if TTS_AVAILABLE:
            TextToSpeech(f"You said: {message}")
        
        # Process with your existing system
        try:
            ShowTextToScreen(QueryModifier(message))
        except Exception as e:
            print(f"Error processing message: {e}")
        
        # Reset status after processing
        QTimer.singleShot(3000, lambda: self.update_status("Ready to assist you", "#4a5568"))
    
    def upload_file(self):
        """Handle file upload - stay on current page"""
        self.update_status("📁 File upload ready", "#ed8936")
        
        # Open file dialog directly on this page
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select files for AI analysis",
            "",
            "Data files (*.csv *.xlsx *.xls *.json *.txt *.pdf);;All Files (*)"
        )
        
        if file_paths:
            file_count = len(file_paths)
            self.update_status(f"📁 {file_count} file(s) selected", "#48bb78")
            
            if TTS_AVAILABLE:
                TextToSpeech(f"Successfully selected {file_count} files for analysis")
            
            # Show files briefly then reset
            QTimer.singleShot(3000, lambda: self.update_status("Files ready for processing", "#4a5568"))
            QTimer.singleShot(5000, lambda: self.update_status("Ready to assist you", "#4a5568"))
        else:
            self.update_status("📁 No files selected", "#ed8936")
            QTimer.singleShot(2000, lambda: self.update_status("Ready to assist you", "#4a5568"))
    
    def show_help(self):
        """Handle help request - stay on current page"""
        self.update_status("❓ Help available", "#e53e3e")
        
        help_text = f"I'm {Assistantname}, your AI assistant. I can help you with voice commands, text chat, and data analysis."
        
        if TTS_AVAILABLE:
            TextToSpeech(help_text)
        
        # Show help information in status
        QTimer.singleShot(1500, lambda: self.update_status("🤖 Voice, Chat & File Analysis", "#667eea"))
        QTimer.singleShot(4000, lambda: self.update_status("How can I assist you?", "#4a5568"))
        QTimer.singleShot(6000, lambda: self.update_status("Ready to assist you", "#4a5568"))
    
    def update_status(self, text, color="#4a5568"):
        """Update status text with color"""
        self.status_text.setText(text)
        self.status_text.setStyleSheet(f"""
            QLabel {{
                font-size: 16px;
                color: {color};
                font-weight: 400;
                margin: 25px 0;
            }}
        """)
    
    def paintEvent(self, event):
        """Custom paint event for gradient background"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create gradient background (matches HTML linear-gradient)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(245, 247, 250))  # #f5f7fa
        gradient.setColorAt(1, QColor(195, 207, 226))  # #c3cfe2
        
        painter.fillRect(self.rect(), gradient)
        
        # Draw container background
        container_rect = self.rect().adjusted(80, 80, -80, -80)
        
        # Container gradient and effects
        container_gradient = QRadialGradient(container_rect.center(), min(container_rect.width(), container_rect.height()) / 2)
        container_gradient.setColorAt(0, QColor(255, 255, 255, 242))  # rgba(255, 255, 255, 0.95)
        container_gradient.setColorAt(1, QColor(255, 255, 255, 230))
        
        painter.setBrush(QBrush(container_gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 51), 1))  # rgba(255, 255, 255, 0.2) border
        
        # Draw rounded container
        painter.drawRoundedRect(container_rect, 32, 32)
        
        # Draw subtle top line
        top_line_gradient = QLinearGradient(container_rect.left(), container_rect.top(), 
                                          container_rect.right(), container_rect.top())
        top_line_gradient.setColorAt(0, QColor(99, 102, 241, 0))
        top_line_gradient.setColorAt(0.5, QColor(99, 102, 241, 76))  # rgba(99, 102, 241, 0.3)
        top_line_gradient.setColorAt(1, QColor(99, 102, 241, 0))
        
        painter.setPen(QPen(QBrush(top_line_gradient), 1))
        painter.drawLine(container_rect.left(), container_rect.top() + 1, 
                        container_rect.right(), container_rect.top() + 1)

# Chat section and other components from original code (keeping them as is)
class EnhancedChatSection(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        # Simplified for space - use original implementation
        self.status_label = QLabel("Chat Interface - Original Implementation")
        self.status_label.setStyleSheet("""
            color: #4FACFE;
            font-size: 18px;
            font-weight: 600;
            padding: 15px 25px;
            background: rgba(79, 172, 254, 0.2);
            border-radius: 22px;
            border: 2px solid rgba(79, 172, 254, 0.4);
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)

class DataAnalysisSection(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(28)
        
        # Simplified for space - use original implementation
        header = QLabel("Data Analysis Center - Original Implementation")
        header.setStyleSheet("""
            font-size: 36px;
            font-weight: bold;
            color: #4FACFE;
            margin-bottom: 8px;
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        
        # Initialize screens
        self.elegant_interface = ElegantMainWidget(self)
        self.chat_screen = EnhancedChatSection()
        self.data_screen = DataAnalysisSection()
        
        # Add screens to stack
        self.stacked_widget.addWidget(self.elegant_interface)
        self.stacked_widget.addWidget(self.chat_screen)
        self.stacked_widget.addWidget(self.data_screen)
        
        # Set main widget
        self.setCentralWidget(self.stacked_widget)
        
        # Window settings
        screen = QApplication.primaryScreen()
        geometry = screen.availableGeometry()
        self.setGeometry(geometry)
        
        # Apply elegant styling that matches HTML
        self.setStyleSheet("""
            QMainWindow {
                background: transparent;
            }
            QWidget {
                font-family: 'SF Pro Display', 'Helvetica Neue', 'Segoe UI', Roboto, Arial, sans-serif;
            }
        """)
        
        self.setWindowTitle(f"{Assistantname} - Elegant AI Assistant")
        
        # Set window icon if available
        if os.path.exists(os.path.join(GraphicsDirPath, "icon.png")):
            self.setWindowIcon(QIcon(os.path.join(GraphicsDirPath, "icon.png")))
        
        self.showMaximized()
        
        # Start with elegant interface
        self.current_screen = 0
    
    def switchScreen(self, index):
        """Switch between screens with smooth transition"""
        if self.current_screen != index:
            self.stacked_widget.setCurrentIndex(index)
            self.current_screen = index
    
    def closeEvent(self, event):
        """Handle application close"""
        if TTS_AVAILABLE:
            TextToSpeech(f"Thank you for using {Assistantname}. Goodbye!")
        else:
            print(f"Thank you for using {Assistantname}. Goodbye!")
        event.accept()

# Utility functions (keeping from original)
def Listen():
    if not SPEECH_RECOGNITION_AVAILABLE:
        print("[GUI] Speech recognition not available")
        return None
        
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("[GUI] Listening...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            print("[GUI] Awaiting speech...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            print("[GUI] Processing audio...")
            query = recognizer.recognize_google(audio)
            print(f"[GUI] You said: {query}")
            return query
        except sr.WaitTimeoutError:
            print("[GUI] Listening timed out while waiting for phrase.")
        except sr.UnknownValueError:
            print("[GUI] Could not understand the audio.")
        except sr.RequestError as e:
            print(f"[GUI] API Error: {e}")
    return None

def Speak(text):
    if TTS_AVAILABLE:
        TextToSpeech(text)
    else:
        print(f"[SPEAK]: {text}")

def ShowTextToScreen(text):
    try:
        with open(os.path.join(TempDirPath, "Responses.data"), "w", encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Error writing to file: {e}")

def AnswerModifier(answer):
    return "\n".join([line for line in answer.split("\n") if line.strip()])

def QueryModifier(query):
    q = query.lower().strip()
    if any(q.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "which", "whose", "whom", "can you", "what's", "where's", "how's"]):
        return q.rstrip(".?!") + "?"
    return q.rstrip(".?!") + "."

def GetAssistantStatus():
    try:
        with open(os.path.join(TempDirPath, "Status.data"), "r", encoding='utf-8') as f:
            return f.read().strip()
    except:
        return ""

def SetAssistantStatus(status):
    try:
        with open(os.path.join(TempDirPath, "Status.data"), "w", encoding='utf-8') as f:
            f.write(status)
    except Exception as e:
        print(f"Error setting status: {e}")

def GraphicalUserInterface():
    """Main function to start the elegant GUI application"""
    # Set high DPI scaling for better display
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName(f"{Assistantname} AI Assistant")
    app.setApplicationVersion("2.1")
    app.setOrganizationName("Elegant AI Systems")
    app.setApplicationDisplayName(f"{Assistantname} - Elegant AI Assistant")
    
    # Load system fonts for better rendering
    font_db = QFontDatabase()
    app.setStyleSheet("""
        * {
            font-family: 'SF Pro Display', 'Helvetica Neue', 'Segoe UI', Roboto, Arial, sans-serif;
        }
    """)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Welcome message
    if TTS_AVAILABLE:
        TextToSpeech(f"Welcome to {Assistantname}, your elegant AI assistant!")
    else:
        print(f"Welcome to {Assistantname}, your elegant AI assistant!")
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs(TempDirPath, exist_ok=True)
    os.makedirs(GraphicsDirPath, exist_ok=True)
    
    # Start the elegant GUI
    GraphicalUserInterface()
