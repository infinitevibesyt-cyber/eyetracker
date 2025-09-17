"""
Neural Network Pupil Tracker
Lightweight gaze estimation using small neural networks

Requirements:
    pip install mediapipe opencv-python numpy scikit-learn joblib pyqt6 torch

Usage:
    python neural_pupil_tracker.py
"""

import sys
import os
import time
import json
import math
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QPointF
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QPushButton, QLabel,
                             QCheckBox, QSlider, QProgressBar, QFileDialog,
                             QMessageBox, QTextEdit, QSpinBox)
from PyQt6.QtGui import (QPixmap, QPainter, QPen, QBrush, QColor, QLinearGradient,
                         QImage, QPalette)

import mediapipe as mp

# ==================== Configuration ====================
class Config:
    # Calibration settings
    CAL_POINTS_GRID = (4, 3)  # 12 calibration points
    FRAMES_PER_POINT = 25     # Frames to collect per point
    POINT_DISPLAY_TIME = 2.5  # seconds
    POINT_SETTLE_TIME = 0.5   # seconds to settle
    MIN_FRAMES_PER_POINT = 10 # Minimum frames required

    # Neural network settings
    HIDDEN_SIZE = 32          # Hidden layer size
    LEARNING_RATE = 0.001     # Learning rate
    EPOCHS = 200              # Training epochs
    BATCH_SIZE = 32           # Batch size for training
    DROPOUT_RATE = 0.1        # Dropout for regularization
    
    # Display settings
    HIGHLIGHT_RADIUS = 25
    HIGHLIGHT_COLOR = (0, 255, 150, 150)
    CALIBRATION_DOT_RADIUS = 8
    CALIBRATION_DOT_COLOR = (255, 0, 0)

    # Smoothing
    SMOOTH_WINDOW = 7
    CONFIDENCE_THRESHOLD = 0.6

    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30

    # File paths
    CALIB_DATA_PATH = "data/calibration_neural.json"
    MODEL_PATH = "models/neural_gaze_model.pkl"
    CONFIG_PATH = "data/app_config.json"

config = Config()

# Ensure directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==================== MediaPipe indices ====================
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]    # inner, outer corner
RIGHT_EYE_CORNERS = [362, 263]  # inner, outer corner
FACE_BOUNDS = [10, 152, 234, 454]  # face boundary points

# ==================== Neural Network Model ====================
class GazeNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, output_size=2, dropout_rate=0.1):
        super(GazeNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

# ==================== Eye Tracker ====================
class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        self.initialize()
        self.history = deque(maxlen=5)

    def initialize(self):
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1, 
                refine_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6
            )
        except Exception as e:
            print("MediaPipe initialization failed:", e)
            self.face_mesh = None

    def extract_features(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.face_mesh is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results or not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Convert to numpy array
        lm = np.array([(p.x * w, p.y * h) for p in landmarks.landmark])
        
        try:
            features = self._compute_iris_features(lm, w, h)
            features['confidence'] = self._compute_confidence(lm)
            features['frame'] = frame
            features['landmarks'] = lm
            
            self.history.append(features)
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def _compute_iris_features(self, landmarks, w, h):
        """Extract iris positions relative to eye regions"""
        # Get iris centers
        left_iris = landmarks[LEFT_IRIS[:4]].mean(axis=0)
        right_iris = landmarks[RIGHT_IRIS[:4]].mean(axis=0)
        
        # Get eye corner coordinates for normalization
        left_corners = landmarks[LEFT_EYE_CORNERS]
        right_corners = landmarks[RIGHT_EYE_CORNERS]
        
        # Eye regions (bounding boxes)
        left_eye_width = abs(left_corners[1][0] - left_corners[0][0])
        right_eye_width = abs(right_corners[1][0] - right_corners[0][0])
        
        left_eye_center = left_corners.mean(axis=0)
        right_eye_center = right_corners.mean(axis=0)
        
        # Normalize iris positions relative to eye centers and widths
        left_rel_x = (left_iris[0] - left_eye_center[0]) / max(left_eye_width, 1.0)
        left_rel_y = (left_iris[1] - left_eye_center[1]) / max(left_eye_width, 1.0)
        right_rel_x = (right_iris[0] - right_eye_center[0]) / max(right_eye_width, 1.0)
        right_rel_y = (right_iris[1] - right_eye_center[1]) / max(right_eye_width, 1.0)
        
        return {
            'Lx': float(left_rel_x),
            'Ly': float(left_rel_y), 
            'Rx': float(right_rel_x),
            'Ry': float(right_rel_y)
        }

    def _compute_confidence(self, landmarks):
        """Simple confidence based on face detection stability"""
        try:
            face_pts = landmarks[FACE_BOUNDS]
            face_area = self._compute_area(face_pts)
            
            min_area = 5000
            max_area = 50000
            
            if face_area < min_area:
                return 0.0
            elif face_area > max_area:
                return 0.8
            else:
                return min(1.0, face_area / max_area)
                
        except:
            return 0.5

    def _compute_area(self, points):
        """Compute area of polygon defined by points"""
        if len(points) < 3:
            return 0
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        return (x_coords.max() - x_coords.min()) * (y_coords.max() - y_coords.min())

# ==================== Camera Thread ====================
class CameraThread(QThread):
    frame_ready = pyqtSignal(object)
    status_update = pyqtSignal(str)

    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.eye_tracker = EyeTracker()
        self.running = False
        self.cap = None

    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        except:
            pass

        if not self.cap or not self.cap.isOpened():
            self.status_update.emit("Failed to open camera")
            return

        self.status_update.emit("Camera started")
        self.running = True

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.msleep(10)
                continue
                
            frame = cv2.flip(frame, 1)  # Mirror for user comfort
            features = self.eye_tracker.extract_features(frame)
            self.frame_ready.emit(features)
            self.msleep(33)  # ~30 FPS

        if self.cap:
            self.cap.release()
        self.status_update.emit("Camera stopped")

    def stop(self):
        self.running = False
        self.wait(2000)

# ==================== Neural Network Gaze Model ====================
class NeuralGazeModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        print(f"Using device: {self.device}")
        
    def train(self, calibration_data):
        """Train neural network using calibration data"""
        samples = calibration_data.get('samples', [])
        
        X = []  # features
        y = []  # target coordinates (x, y pairs)
        
        # Extract training data
        for sample in samples:
            target_x, target_y = sample['target']
            frames = sample['features']
            
            for frame_features in frames:
                try:
                    feature_vector = [
                        frame_features['Lx'],
                        frame_features['Ly'], 
                        frame_features['Rx'],
                        frame_features['Ry']
                    ]
                    X.append(feature_vector)
                    y.append([target_x, target_y])
                except KeyError:
                    continue
        
        if len(X) < 20:
            raise ValueError(f"Insufficient training data: {len(X)} samples")
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"Training neural network with {len(X)} samples")
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create model
        self.model = GazeNet(
            input_size=4,
            hidden_size=config.HIDDEN_SIZE,
            output_size=2,
            dropout_rate=config.DROPOUT_RATE
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        # Training loop
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=min(config.BATCH_SIZE, len(X)), 
            shuffle=True
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.EPOCHS):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 30:  # Early stopping
                print(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
            error = np.sqrt(np.mean((predictions - y)**2, axis=1)).mean()
        
        self.is_trained = True
        
        return {
            'num_samples': len(X),
            'num_features': 4,
            'mean_error': float(error),
            'final_loss': float(best_loss)
        }
    
    def predict(self, features):
        """Predict gaze coordinates from iris features"""
        if not self.is_trained:
            return None
            
        try:
            # Extract features
            X = np.array([[
                features['Lx'],
                features['Ly'],
                features['Rx'], 
                features['Ry']
            ]], dtype=np.float32)
            
            # Normalize
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(X_tensor).cpu().numpy()[0]
            
            return (float(prediction[0]), float(prediction[1]))
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def save(self, path):
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Move model to CPU for saving
        cpu_model = self.model.cpu()
        
        data = {
            'model_state_dict': cpu_model.state_dict(),
            'model_config': {
                'input_size': 4,
                'hidden_size': config.HIDDEN_SIZE,
                'output_size': 2,
                'dropout_rate': config.DROPOUT_RATE
            },
            'scaler': self.scaler
        }
        
        torch.save(data, path)
        
        # Move model back to device
        self.model = self.model.to(self.device)
    
    def load(self, path):
        try:
            data = torch.load(path, map_location=self.device)
            
            # Recreate model
            model_config = data['model_config']
            self.model = GazeNet(**model_config).to(self.device)
            self.model.load_state_dict(data['model_state_dict'])
            self.scaler = data['scaler']
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Model load failed: {e}")
            return False

# ==================== Image Display Widget ====================
class ImageDisplayWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 2px solid #444; background-color: #222;")
        self.setMinimumSize(800, 600)
        self._image = None
        self._highlight_pos = None
        self._calibration_points = []
        self._current_calibration_point = None

    def set_image(self, image_path: Optional[str] = None):
        if image_path and os.path.exists(image_path):
            pix = QPixmap(image_path)
        else:
            pix = self._create_default_background()
        self._image = pix
        self._update_display()

    def _create_default_background(self):
        size = QSize(800, 600)
        pix = QPixmap(size)
        painter = QPainter(pix)
        grad = QLinearGradient(0, 0, size.width(), size.height())
        grad.setColorAt(0, QColor(30, 30, 50))
        grad.setColorAt(0.5, QColor(50, 50, 100))
        grad.setColorAt(1, QColor(30, 30, 50))
        painter.fillRect(pix.rect(), grad)
        painter.end()
        return pix

    def get_image_rect(self):
        """Get the rectangle where the scaled image is displayed"""
        if not self._image:
            return QRect(0, 0, self.width(), self.height())
            
        scaled = self._image.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

    def set_highlight(self, x, y):
        self._highlight_pos = (x, y)
        self.update()

    def clear_highlight(self):
        self._highlight_pos = None
        self.update()

    def set_calibration_points(self, points):
        self._calibration_points = points
        self.update()

    def set_current_calibration_point(self, point):
        self._current_calibration_point = point
        self.update()

    def clear_calibration(self):
        self._calibration_points = []
        self._current_calibration_point = None
        self.update()

    def _update_display(self):
        if self._image:
            scaled = self._image.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.setPixmap(scaled)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        img_rect = self.get_image_rect()
        
        # Draw completed calibration points
        for point in self._calibration_points:
            px = img_rect.x() + point[0] * img_rect.width()
            py = img_rect.y() + point[1] * img_rect.height()
            painter.setPen(QPen(QColor(150, 150, 150), 1))
            painter.setBrush(QBrush(QColor(150, 150, 150, 100)))
            painter.drawEllipse(QPointF(px, py), 4, 4)

        # Draw current calibration point
        if self._current_calibration_point:
            point = self._current_calibration_point
            px = img_rect.x() + point[0] * img_rect.width()
            py = img_rect.y() + point[1] * img_rect.height()
            
            color = QColor(*config.CALIBRATION_DOT_COLOR)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(px, py), config.CALIBRATION_DOT_RADIUS, 
                              config.CALIBRATION_DOT_RADIUS)

        # Draw gaze highlight
        if self._highlight_pos:
            hx = img_rect.x() + self._highlight_pos[0]
            hy = img_rect.y() + self._highlight_pos[1]
            
            color = QColor(*config.HIGHLIGHT_COLOR)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(hx, hy), config.HIGHLIGHT_RADIUS, 
                              config.HIGHLIGHT_RADIUS)

        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image:
            self._update_display()

# ==================== Webcam Preview ====================
class WebcamPreview(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(320, 240)
        self.setStyleSheet("border: 1px solid #666; background-color: #111;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Camera Preview")

    def update_frame(self, features):
        if not features or 'frame' not in features:
            return
            
        frame = features['frame'].copy()
        
        # Draw iris centers if available
        if 'landmarks' in features and features['landmarks'] is not None:
            try:
                lm = features['landmarks']
                
                # Draw left iris
                left_iris = lm[LEFT_IRIS[:4]].mean(axis=0).astype(int)
                cv2.circle(frame, tuple(left_iris), 3, (0, 255, 0), -1)
                
                # Draw right iris  
                right_iris = lm[RIGHT_IRIS[:4]].mean(axis=0).astype(int)
                cv2.circle(frame, tuple(right_iris), 3, (0, 255, 0), -1)
                
                # Show confidence
                conf = features.get('confidence', 0)
                cv2.putText(frame, f"Conf: {conf:.2f}", (10, 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
            except Exception as e:
                print(f"Preview error: {e}")

        # Convert to Qt format
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (320, 240))
        h, w, ch = small.shape
        bytes_per_line = ch * w
        img = QImage(small.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(img))

# ==================== Main Application ====================
class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Pupil Tracker")
        self.setGeometry(100, 100, 1200, 800)

        # Core components
        self.camera_thread = None
        self.gaze_model = NeuralGazeModel()
        self.calibration_data = []
        self.recent_predictions = deque(maxlen=config.SMOOTH_WINDOW)
        
        # State
        self.is_calibrating = False
        self.is_highlighting = False
        self.current_features = None
        
        # Calibration state
        self.calibration_points = []
        self.calibration_index = 0
        self.point_frames = []
        self.point_start_time = 0.0
        
        # Timers
        self.calibration_timer = QTimer()
        self.prediction_timer = QTimer()
        
        self.setup_ui()
        self.setup_connections()
        self.start_camera()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left panel
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        self.btn_load_image = QPushButton("Load Image")
        self.btn_use_default = QPushButton("Use Default Background")  
        self.btn_calibrate = QPushButton("Start Calibration")
        self.btn_train = QPushButton("Train Neural Network")
        self.btn_highlight = QPushButton("Start Highlighting")
        self.btn_save_model = QPushButton("Save Model")
        self.btn_load_model = QPushButton("Load Model")
        
        self.btn_train.setEnabled(False)
        self.btn_highlight.setEnabled(False)
        
        for btn in [self.btn_load_image, self.btn_use_default, self.btn_calibrate,
                   self.btn_train, self.btn_highlight, self.btn_save_model, 
                   self.btn_load_model]:
            controls_layout.addWidget(btn)
        
        left_layout.addWidget(controls_group)

        # Settings
        settings_group = QGroupBox("Neural Network Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        self.chk_show_preview = QCheckBox("Show Camera Preview")
        self.chk_show_preview.setChecked(True)
        
        # Neural network parameters
        hidden_layout = QHBoxLayout()
        hidden_layout.addWidget(QLabel("Hidden Size:"))
        self.spin_hidden_size = QSpinBox()
        self.spin_hidden_size.setRange(16, 128)
        self.spin_hidden_size.setValue(config.HIDDEN_SIZE)
        hidden_layout.addWidget(self.spin_hidden_size)
        
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(50, 500)
        self.spin_epochs.setValue(config.EPOCHS)
        epochs_layout.addWidget(self.spin_epochs)
        
        # Grid size
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Calibration Grid:"))
        self.spin_grid_x = QSpinBox()
        self.spin_grid_x.setRange(3, 5)
        self.spin_grid_x.setValue(config.CAL_POINTS_GRID[0])
        self.spin_grid_y = QSpinBox()
        self.spin_grid_y.setRange(3, 4)
        self.spin_grid_y.setValue(config.CAL_POINTS_GRID[1])
        grid_layout.addWidget(self.spin_grid_x)
        grid_layout.addWidget(QLabel("x"))
        grid_layout.addWidget(self.spin_grid_y)
        
        settings_layout.addWidget(self.chk_show_preview)
        settings_layout.addLayout(hidden_layout)
        settings_layout.addLayout(epochs_layout)
        settings_layout.addLayout(grid_layout)
        left_layout.addWidget(settings_group)

        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.lbl_camera_status = QLabel("Camera: Starting...")
        self.lbl_calibration_status = QLabel("Calibration: Not done")
        self.lbl_model_status = QLabel("Model: Not trained")
        self.lbl_accuracy = QLabel("Accuracy: N/A")
        
        for lbl in [self.lbl_camera_status, self.lbl_calibration_status,
                   self.lbl_model_status, self.lbl_accuracy]:
            status_layout.addWidget(lbl)
        
        left_layout.addWidget(status_group)

        # Camera preview
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.webcam_preview = WebcamPreview()
        preview_layout.addWidget(self.webcam_preview)
        left_layout.addWidget(preview_group)
        
        left_layout.addStretch()

        # Right panel
        right_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.image_display = ImageDisplayWidget()
        
        self.info_label = QLabel("Load an image or use default background to begin")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #888; font-size: 14px; padding: 10px;")
        
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)
        self.log_box.setStyleSheet("background: #111; color: #ddd; font-family: monospace;")
        
        right_layout.addWidget(self.progress_bar)
        right_layout.addWidget(self.image_display)
        right_layout.addWidget(self.info_label)
        right_layout.addWidget(QLabel("Log:"))
        right_layout.addWidget(self.log_box)

        layout.addWidget(left_panel)
        layout.addLayout(right_layout, stretch=1)
        
        self.image_display.set_image()  # Default background

    def setup_connections(self):
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_use_default.clicked.connect(self.use_default_background)
        self.btn_calibrate.clicked.connect(self.start_calibration)
        self.btn_train.clicked.connect(self.train_model)
        self.btn_highlight.clicked.connect(self.toggle_highlighting)
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_load_model.clicked.connect(self.load_model)
        
        self.chk_show_preview.stateChanged.connect(self.toggle_preview)
        self.spin_grid_x.valueChanged.connect(self.update_grid_size)
        self.spin_grid_y.valueChanged.connect(self.update_grid_size)
        self.spin_hidden_size.valueChanged.connect(self.update_nn_params)
        self.spin_epochs.valueChanged.connect(self.update_nn_params)
        
        self.calibration_timer.timeout.connect(self.calibration_step)
        self.prediction_timer.timeout.connect(self.update_prediction)

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{timestamp}] {message}")
        print(message)

    def start_camera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.on_frame_ready)
        self.camera_thread.status_update.connect(self.on_camera_status)
        self.camera_thread.start()

    def on_frame_ready(self, features):
        self.current_features = features
        if self.chk_show_preview.isChecked():
            self.webcam_preview.update_frame(features)

    def on_camera_status(self, status):
        self.lbl_camera_status.setText(f"Camera: {status}")
        self.log(f"Camera: {status}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.image_display.set_image(path)
            self.info_label.setText(f"Loaded: {os.path.basename(path)}")
            self.log(f"Loaded image: {path}")

    def use_default_background(self):
        self.image_display.set_image()
        self.info_label.setText("Using default background")
        self.log("Using default background")

    def generate_calibration_points(self):
        """Generate calibration points for neural network training"""
        points = []
        rows, cols = config.CAL_POINTS_GRID
        
        # Use safe margins
        margin = 0.15
        
        for r in range(rows):
            for c in range(cols):
                if cols == 1:
                    x = 0.5
                else:
                    x = margin + (1 - 2*margin) * c / (cols - 1)
                    
                if rows == 1:
                    y = 0.5
                else:
                    y = margin + (1 - 2*margin) * r / (rows - 1)
                
                points.append((x, y))
        
        return points

    def start_calibration(self):
        if not self.current_features:
            QMessageBox.warning(self, "No Camera Input", 
                              "Please ensure camera is working and face is detected.")
            return

        config.CAL_POINTS_GRID = (self.spin_grid_x.value(), self.spin_grid_y.value())
        
        self.calibration_points = self.generate_calibration_points()
        self.calibration_data = []
        self.calibration_index = 0
        self.point_frames = []
        self.point_start_time = time.time()
        self.is_calibrating = True
        
        self.btn_calibrate.setText("Calibrating...")
        self.btn_calibrate.setEnabled(False)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.calibration_points))
        self.progress_bar.setValue(0)
        
        self.image_display.set_calibration_points(self.calibration_points)
        self.image_display.set_current_calibration_point(self.calibration_points[0])
        
        self.calibration_timer.start(100)
        self.info_label.setText("Calibration started - look at the red dots")
        self.log(f"Starting neural network calibration with {len(self.calibration_points)} points")

    def calibration_step(self):
        if not self.is_calibrating:
            return
            
        if self.calibration_index >= len(self.calibration_points):
            self.finish_calibration()
            return
        
        current_point = self.calibration_points[self.calibration_index]
        elapsed = time.time() - self.point_start_time
        
        self.image_display.set_current_calibration_point(current_point)
        
        # Collect features if confidence is good enough
        if (self.current_features and 
            self.current_features.get('confidence', 0) >= config.CONFIDENCE_THRESHOLD):
            
            # Extract the 4 core features
            try:
                features = {
                    'Lx': self.current_features['Lx'],
                    'Ly': self.current_features['Ly'],
                    'Rx': self.current_features['Rx'],
                    'Ry': self.current_features['Ry']
                }
                self.point_frames.append(features)
            except KeyError:
                pass
        
        if elapsed < config.POINT_SETTLE_TIME:
            return
        
        collect_time = elapsed - config.POINT_SETTLE_TIME
        
        if (len(self.point_frames) >= config.FRAMES_PER_POINT or 
            collect_time >= config.POINT_DISPLAY_TIME):
            
            if len(self.point_frames) >= config.MIN_FRAMES_PER_POINT:
                # Convert calibration point to image coordinates
                img_rect = self.image_display.get_image_rect()
                target_x = current_point[0] * img_rect.width()
                target_y = current_point[1] * img_rect.height()
                
                self.calibration_data.append({
                    'target': [target_x, target_y],
                    'features': self.point_frames.copy()
                })
                
                self.info_label.setText(
                    f"Point {self.calibration_index + 1}/{len(self.calibration_points)} completed "
                    f"({len(self.point_frames)} frames)"
                )
                self.log(f"Point {self.calibration_index + 1} collected: {len(self.point_frames)} frames")
            else:
                self.info_label.setText(
                    f"Point {self.calibration_index + 1} failed - insufficient frames"
                )
                self.log(f"Point {self.calibration_index + 1} failed: only {len(self.point_frames)} frames")
            
            self.calibration_index += 1
            self.progress_bar.setValue(self.calibration_index)
            self.point_frames = []
            self.point_start_time = time.time()
            
            if self.calibration_index < len(self.calibration_points):
                self.image_display.set_current_calibration_point(
                    self.calibration_points[self.calibration_index]
                )

    def finish_calibration(self):
        self.is_calibrating = False
        self.calibration_timer.stop()
        self.btn_calibrate.setText("Start Calibration")
        self.btn_calibrate.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.image_display.clear_calibration()
        
        valid_samples = len(self.calibration_data)
        min_required = max(6, len(self.calibration_points) // 2)
        
        if valid_samples >= min_required:
            # Save calibration data
            calib_bundle = {
                'image_size': [
                    self.image_display.get_image_rect().width(),
                    self.image_display.get_image_rect().height()
                ],
                'samples': self.calibration_data
            }
            
            try:
                with open(config.CALIB_DATA_PATH, 'w') as f:
                    json.dump(calib_bundle, f, indent=2)
                
                self.lbl_calibration_status.setText(f"Calibration: {valid_samples} points")
                self.btn_train.setEnabled(True)
                self.info_label.setText(f"Calibration completed with {valid_samples} valid points")
                self.log("Calibration data saved successfully")
                
            except Exception as e:
                self.lbl_calibration_status.setText("Calibration: Save failed")
                QMessageBox.warning(self, "Save Failed", str(e))
                self.log(f"Failed to save calibration: {e}")
        else:
            self.lbl_calibration_status.setText("Calibration: Failed")
            self.info_label.setText("Calibration failed - please retry")
            QMessageBox.warning(self, "Calibration Failed", 
                              f"Only {valid_samples} valid points. Need at least {min_required}.")
            self.log(f"Calibration failed: {valid_samples}/{min_required} points")

    def train_model(self):
        # Update config from UI
        config.HIDDEN_SIZE = self.spin_hidden_size.value()
        config.EPOCHS = self.spin_epochs.value()
        
        # Load calibration data
        calib_data = None
        if self.calibration_data:
            calib_data = {
                'image_size': [
                    self.image_display.get_image_rect().width(),
                    self.image_display.get_image_rect().height()
                ],
                'samples': self.calibration_data
            }
        elif os.path.exists(config.CALIB_DATA_PATH):
            try:
                with open(config.CALIB_DATA_PATH, 'r') as f:
                    calib_data = json.load(f)
            except Exception as e:
                QMessageBox.warning(self, "Training Failed", f"Failed to load calibration: {e}")
                return
        else:
            QMessageBox.warning(self, "No Calibration", "Please run calibration first")
            return

        try:
            self.info_label.setText("Training neural network... This may take a while.")
            QApplication.processEvents()
            
            self.log(f"Starting neural network training (Hidden: {config.HIDDEN_SIZE}, Epochs: {config.EPOCHS})")
            
            metrics = self.gaze_model.train(calib_data)
            
            # Auto-save model
            try:
                self.gaze_model.save(config.MODEL_PATH)
            except Exception as e:
                self.log(f"Auto-save failed: {e}")
            
            self.lbl_model_status.setText("Model: Trained (Neural Net)")
            self.lbl_accuracy.setText(f"Error: {metrics['mean_error']:.1f}px")
            self.btn_highlight.setEnabled(True)
            
            self.info_label.setText(
                f"Neural network trained! Mean error: {metrics['mean_error']:.1f}px, "
                f"Loss: {metrics['final_loss']:.4f}, Samples: {metrics['num_samples']}"
            )
            
            self.log(f"Neural network training completed: {metrics['mean_error']:.1f}px error, "
                    f"Final loss: {metrics['final_loss']:.4f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Training Failed", f"Neural network training failed: {e}")
            self.log(f"Training error: {e}")

    def toggle_highlighting(self):
        if not self.is_highlighting:
            if not self.gaze_model.is_trained:
                if os.path.exists(config.MODEL_PATH):
                    if not self.gaze_model.load(config.MODEL_PATH):
                        QMessageBox.warning(self, "Load Failed", "Could not load saved model")
                        return
                else:
                    QMessageBox.warning(self, "No Model", "Please train a model first")
                    return
            
            self.is_highlighting = True
            self.btn_highlight.setText("Stop Highlighting")
            self.prediction_timer.start(40)  # 25 FPS
            self.info_label.setText("Neural network gaze highlighting active")
            self.log("Started neural network gaze highlighting")
        else:
            self.is_highlighting = False
            self.btn_highlight.setText("Start Highlighting")
            self.prediction_timer.stop()
            self.image_display.clear_highlight()
            self.info_label.setText("Gaze highlighting stopped")
            self.log("Stopped gaze highlighting")

    def update_prediction(self):
        if not self.is_highlighting or not self.current_features:
            return
        
        # Check confidence
        confidence = self.current_features.get('confidence', 0)
        if confidence < config.CONFIDENCE_THRESHOLD:
            return
        
        # Get prediction from neural network
        prediction = self.gaze_model.predict(self.current_features)
        if prediction is None:
            return
        
        pred_x, pred_y = prediction
        
        # Add to smoothing buffer
        self.recent_predictions.append((pred_x, pred_y))
        
        # Apply smoothing
        if len(self.recent_predictions) >= 3:
            # Use exponential moving average for smoother results
            weights = np.exp(np.linspace(-1, 0, len(self.recent_predictions)))
            weights /= weights.sum()
            
            recent_preds = list(self.recent_predictions)
            avg_x = sum(w * p[0] for w, p in zip(weights, recent_preds))
            avg_y = sum(w * p[1] for w, p in zip(weights, recent_preds))
        else:
            avg_x, avg_y = pred_x, pred_y
        
        # Get image rect to clamp coordinates
        img_rect = self.image_display.get_image_rect()
        
        # Clamp to image bounds
        final_x = max(0, min(img_rect.width() - 1, avg_x))
        final_y = max(0, min(img_rect.height() - 1, avg_y))
        
        # Update highlight
        self.image_display.set_highlight(final_x, final_y)

    def save_model(self):
        if not self.gaze_model.is_trained:
            QMessageBox.warning(self, "No Model", "No trained neural network to save")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Neural Network Model", "", "Model files (*.pth)")
        if path:
            try:
                self.gaze_model.save(path)
                QMessageBox.information(self, "Saved", "Neural network model saved successfully")
                self.log(f"Neural network model saved to {path}")
            except Exception as e:
                QMessageBox.warning(self, "Save Failed", str(e))
                self.log(f"Save failed: {e}")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Neural Network Model", "", "Model files (*.pth)")
        if path:
            if self.gaze_model.load(path):
                self.lbl_model_status.setText("Model: Loaded (Neural Net)")
                self.btn_highlight.setEnabled(True)
                self.log(f"Neural network model loaded from {path}")
            else:
                QMessageBox.warning(self, "Load Failed", "Failed to load neural network model")

    def toggle_preview(self, state):
        if state != Qt.CheckState.Checked:
            self.webcam_preview.setText("Camera Preview")
            self.webcam_preview.setPixmap(QPixmap())

    def update_grid_size(self):
        grid_x = self.spin_grid_x.value()
        grid_y = self.spin_grid_y.value()
        config.CAL_POINTS_GRID = (grid_x, grid_y)
        total = grid_x * grid_y
        self.log(f"Calibration grid: {grid_x}x{grid_y} ({total} points)")

    def update_nn_params(self):
        hidden_size = self.spin_hidden_size.value()
        epochs = self.spin_epochs.value()
        self.log(f"Neural network params: Hidden size={hidden_size}, Epochs={epochs}")

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        self.calibration_timer.stop()
        self.prediction_timer.stop()
        event.accept()

# ==================== Main Function ====================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Dark theme
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 48))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 38))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(60, 60, 63))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(60, 60, 63))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    # Check for PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        QMessageBox.critical(None, "Missing Dependency", 
                           "PyTorch is required for neural network functionality.\n"
                           "Install with: pip install torch")
        sys.exit(1)

    window = MainApplication()
    window.show()

    def cleanup():
        if window.camera_thread:
            window.camera_thread.stop()
    
    app.aboutToQuit.connect(cleanup)

    try:
        sys.exit(app.exec())
    except SystemExit:
        pass

if __name__ == "__main__":
    main()
