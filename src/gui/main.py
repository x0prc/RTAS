import sys
import os
import threading
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QPushButton, QComboBox,
                            QProgressBar, QTextEdit, QFileDialog, QMessageBox,
                            QGroupBox, QSlider, QSpinBox)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon
import torch
import torchaudio

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cpu.stego_encode import Encode
from cpu.stego_decode import Decode

class AudioProcessor(QThread):
    """Background thread for audio processing to keep GUI responsive"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, mode, method, input_file, secret_file=None, output_file=None):
        super().__init__()
        self.mode = mode
        self.method = method
        self.input_file = input_file
        self.secret_file = secret_file
        self.output_file = output_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self):
        try:
            if self.mode == 'encode':
                self.encode_audio()
            else:
                self.decode_audio()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def encode_audio(self):
        self.status_updated.emit("Loading cover audio...")
        self.progress_updated.emit(10)

        cover, sr = torchaudio.load(self.input_file)

        self.status_updated.emit("Loading secret audio...")
        self.progress_updated.emit(20)

        secret, sr2 = torchaudio.load(self.secret_file)

        if sr2 != sr:
            self.status_updated.emit("Resampling secret audio...")
            self.progress_updated.emit(30)
            secret = torchaudio.functional.resample(secret, sr2, sr)

        self.status_updated.emit("Moving to GPU...")
        self.progress_updated.emit(40)

        cover, secret = cover.to(self.device), secret.to(self.device)

        self.status_updated.emit(f"Encoding using {self.method.upper()} method...")
        self.progress_updated.emit(60)

        steg = Encode(device=self.device)

        if self.method == 'lsb':
            stego = steg.lsb_embed(cover, secret)
        elif self.method == 'fft':
            stego = steg.fft_embed(cover, secret)
        else:
            stego = steg.echo_hide(cover, secret)

        self.status_updated.emit("Saving stego audio...")
        self.progress_updated.emit(80)

        torchaudio.save(self.output_file, stego.cpu(), sr)

        self.progress_updated.emit(100)
        self.finished.emit(f"Encoding complete! Saved to {self.output_file}")

    def decode_audio(self):
        self.status_updated.emit("Loading stego audio...")
        self.progress_updated.emit(20)

        stego, sr = torchaudio.load(self.input_file)
        stego = stego.to(self.device)

        self.status_updated.emit(f"Decoding using {self.method.upper()} method...")
        self.progress_updated.emit(60)

        decoder = Decode(device=self.device)

        if self.method == 'lsb':
            extracted = decoder.decode_lsb(stego)
        elif self.method == 'fft':
            extracted = decoder.decode_fft(stego)
        else:
            extracted = decoder.decode_echo(stego)

        self.status_updated.emit("Saving extracted audio...")
        self.progress_updated.emit(80)

        torchaudio.save(self.output_file, extracted.cpu(), sr)

        self.progress_updated.emit(100)
        self.finished.emit(f"Decoding complete! Saved to {self.output_file}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.processor = None

    def init_ui(self):
        self.setWindowTitle("Real-Time Audio Steganography")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Title
        title = QLabel("Real-Time Audio Steganography")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title)

        # Mode selection
        mode_group = QGroupBox("Mode Selection")
        mode_layout = QHBoxLayout()

        self.encode_btn = QPushButton("Encode")
        self.decode_btn = QPushButton("Decode")
        self.encode_btn.setCheckable(True)
        self.decode_btn.setCheckable(True)
        self.encode_btn.clicked.connect(self.on_encode_selected)
        self.decode_btn.clicked.connect(self.on_decode_selected)

        mode_layout.addWidget(self.encode_btn)
        mode_layout.addWidget(self.decode_btn)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Method selection
        method_group = QGroupBox("Steganography Method")
        method_layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(["LSB", "FFT", "Echo"])
        self.method_combo.setCurrentText("FFT")

        method_layout.addWidget(self.method_combo)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()

        # Input file
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Cover Audio:")
        self.input_path = QLabel("No file selected")
        self.input_btn = QPushButton("Browse")
        self.input_btn.clicked.connect(self.browse_input_file)

        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path, 1)
        input_layout.addWidget(self.input_btn)
        file_layout.addLayout(input_layout)

        # Secret file (for encoding)
        secret_layout = QHBoxLayout()
        self.secret_label = QLabel("Secret Audio:")
        self.secret_path = QLabel("No file selected")
        self.secret_btn = QPushButton("Browse")
        self.secret_btn.clicked.connect(self.browse_secret_file)

        secret_layout.addWidget(self.secret_label)
        secret_layout.addWidget(self.secret_path, 1)
        secret_layout.addWidget(self.secret_btn)
        file_layout.addLayout(secret_layout)

        # Output file
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output File:")
        self.output_path = QLabel("No file selected")
        self.output_btn = QPushButton("Browse")
        self.output_btn.clicked.connect(self.browse_output_file)

        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path, 1)
        output_layout.addWidget(self.output_btn)
        file_layout.addLayout(output_layout)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.log_text)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Control buttons
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Processing")
        self.stop_btn = QPushButton("Stop")
        self.clear_btn = QPushButton("Clear Log")

        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.clear_btn.clicked.connect(self.clear_log)

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.clear_btn)
        layout.addLayout(control_layout)

        # Device info
        device_label = QLabel(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        device_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(device_label)

        # Initial state
        self.encode_btn.setChecked(True)
        self.on_encode_selected()

    def on_encode_selected(self):
        self.encode_btn.setChecked(True)
        self.decode_btn.setChecked(False)
        self.input_label.setText("Cover Audio:")
        self.secret_label.show()
        self.secret_path.show()
        self.secret_btn.show()

    def on_decode_selected(self):
        self.decode_btn.setChecked(True)
        self.encode_btn.setChecked(False)
        self.input_label.setText("Stego Audio:")
        self.secret_label.hide()
        self.secret_path.hide()
        self.secret_btn.hide()

    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if file_path:
            self.input_path.setText(file_path)

    def browse_secret_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Secret Audio File", "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if file_path:
            self.secret_path.setText(file_path)

    def browse_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Output File", "", "Audio Files (*.wav)"
        )
        if file_path:
            self.output_path.setText(file_path)

    def start_processing(self):
        # Validate inputs
        if self.input_path.text() == "No file selected":
            QMessageBox.warning(self, "Error", "Please select an input file")
            return

        if self.encode_btn.isChecked() and self.secret_path.text() == "No file selected":
            QMessageBox.warning(self, "Error", "Please select a secret file for encoding")
            return

        if self.output_path.text() == "No file selected":
            QMessageBox.warning(self, "Error", "Please select an output file")
            return

        # Start processing
        mode = "encode" if self.encode_btn.isChecked() else "decode"
        method = self.method_combo.currentText().lower()

        self.processor = AudioProcessor(
            mode=mode,
            method=method,
            input_file=self.input_path.text(),
            secret_file=self.secret_path.text() if mode == "encode" else None,
            output_file=self.output_path.text()
        )

        self.processor.progress_updated.connect(self.update_progress)
        self.processor.status_updated.connect(self.update_status)
        self.processor.finished.connect(self.processing_finished)
        self.processor.error_occurred.connect(self.processing_error)

        self.processor.start()
        self.start_btn.setEnabled(False)

    def stop_processing(self):
        if self.processor and self.processor.isRunning():
            self.processor.terminate()
            self.start_btn.setEnabled(True)
            self.update_status("Processing stopped")

    def clear_log(self):
        self.log_text.clear()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)
        self.log_text.append(f"{time.strftime('%H:%M:%S')} - {message}")

    def processing_finished(self, message):
        self.start_btn.setEnabled(True)
        self.update_status("Complete")
        QMessageBox.information(self, "Success", message)

    def processing_error(self, error):
        self.start_btn.setEnabled(True)
        self.update_status("Error occurred")
        QMessageBox.critical(self, "Error", f"Processing failed: {error}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
