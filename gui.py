from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QFont, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDateEdit,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from analytics import build_daily_summary, daily_totals
from config import APP_DIR, AppConfig, save_config
from database import BarkDatabase
from audio_engine import AudioEngine


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(figsize=(5, 3))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self, app: QApplication, config: AppConfig, db: BarkDatabase, engine: AudioEngine):
        super().__init__()
        self.app = app
        self.config = config
        self.db = db
        self.engine = engine
        self.setWindowTitle("Dog Bark Monitor")
        
        # Create dog icon
        icon = self._create_dog_icon()
        self.setWindowIcon(icon)
        app.setWindowIcon(icon)
        
        self.resize(1100, 760)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.main_tab = self._build_main_tab()
        self.analytics_tab = self._build_analytics_tab()
        self.data_tab = self._build_data_tab()

        self.tabs.addTab(self.main_tab, "Monitor")
        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.tabs.addTab(self.data_tab, "Data Management")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(500)

        if config.auto_start:
            self.engine.start()

    def _create_dog_icon(self) -> QIcon:
        """Create a dog emoji icon for the application."""
        pixmap = QPixmap(128, 128)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw dog emoji
        font = QFont("Apple Color Emoji", 96)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🐕")
        
        painter.end()
        return QIcon(pixmap)

    def _build_main_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        status_box = QGroupBox("Status")
        form = QFormLayout(status_box)
        self.status_label = QLabel("Paused")
        self.counter_label = QLabel("0")
        self.conf_meter = QProgressBar()
        self.conf_meter.setRange(0, 100)
        self.conf_meter.setFormat("%p%")
        self.conf_meter.setTextVisible(True)
        self.conf_meter.setMinimumHeight(25)
        form.addRow("State", self.status_label)
        form.addRow("Today's Events", self.counter_label)
        form.addRow("Confidence", self.conf_meter)

        controls = QHBoxLayout()
        start_btn = QPushButton("Start Listening")
        start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        pause_btn = QPushButton("Pause Listening")
        pause_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e68900; }"
        )
        stop_btn = QPushButton("Stop Listening")
        stop_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        start_btn.clicked.connect(self._on_start)
        pause_btn.clicked.connect(self._on_pause)
        stop_btn.clicked.connect(self._on_stop)
        controls.addWidget(start_btn)
        controls.addWidget(pause_btn)
        controls.addWidget(stop_btn)

        cfg_box = QGroupBox("Detection Settings")
        cfg_form = QFormLayout(cfg_box)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(self.config.detection_threshold)
        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.5, 10.0)
        self.cooldown_spin.setSingleStep(0.5)
        self.cooldown_spin.setValue(self.config.cooldown_seconds)
        self.auto_start_check = QCheckBox("Auto-start on launch")
        self.auto_start_check.setChecked(self.config.auto_start)
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        cfg_form.addRow("Threshold", self.threshold_spin)
        cfg_form.addRow("Cooldown (s)", self.cooldown_spin)
        cfg_form.addRow(self.auto_start_check)
        cfg_form.addRow(save_btn)

        layout.addWidget(status_box)
        layout.addLayout(controls)
        layout.addWidget(cfg_box)
        return w

    def _build_analytics_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(date.today())
        refresh_btn = QPushButton("Load Date")
        refresh_btn.clicked.connect(self.load_analytics)
        top.addWidget(QLabel("Date"))
        top.addWidget(self.date_picker)
        top.addWidget(refresh_btn)

        self.summary_label = QLabel("No data")

        charts = QGridLayout()
        self.hist_canvas = MplCanvas()
        self.trend_canvas = MplCanvas()
        self.dist_canvas = MplCanvas()
        charts.addWidget(self.hist_canvas, 0, 0)
        charts.addWidget(self.trend_canvas, 0, 1)
        charts.addWidget(self.dist_canvas, 1, 0, 1, 2)

        layout.addLayout(top)
        layout.addWidget(self.summary_label)
        layout.addLayout(charts)
        return w

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.day_list = QListWidget()
        self.range_start = QDateEdit(); self.range_start.setCalendarPopup(True); self.range_start.setDate(date.today())
        self.range_end = QDateEdit(); self.range_end.setCalendarPopup(True); self.range_end.setDate(date.today())

        btn_row = QHBoxLayout()
        delete_btn = QPushButton("Delete Selected Day")
        delete_btn.clicked.connect(self.delete_selected_day)
        export_day_btn = QPushButton("Export Selected Day")
        export_day_btn.clicked.connect(self.export_selected_day)
        export_range_btn = QPushButton("Export Date Range")
        export_range_btn.clicked.connect(self.export_range)
        clear_btn = QPushButton("Delete All Data")
        clear_btn.clicked.connect(self.delete_all)
        btn_row.addWidget(delete_btn)
        btn_row.addWidget(export_day_btn)
        btn_row.addWidget(export_range_btn)
        btn_row.addWidget(clear_btn)

        layout.addWidget(self.day_list)
        layout.addWidget(QLabel("Range Start"))
        layout.addWidget(self.range_start)
        layout.addWidget(QLabel("Range End"))
        layout.addWidget(self.range_end)
        layout.addLayout(btn_row)
        self.reload_days()
        return w

    def _on_start(self):
        if self.engine.stats.listening:
            self.engine.resume()
        else:
            self.engine.start()

    def _on_pause(self):
        self.engine.pause()

    def _on_stop(self):
        self.engine.stop()

    def _save_settings(self):
        self.config.detection_threshold = self.threshold_spin.value()
        self.config.cooldown_seconds = self.cooldown_spin.value()
        self.config.auto_start = self.auto_start_check.isChecked()
        self.engine.detector.threshold = self.config.detection_threshold
        self.engine.config.cooldown_seconds = self.config.cooldown_seconds
        save_config(self.config)

    def refresh(self):
        if self.engine.stats.listening:
            self.status_label.setText("● LISTENING")
            self.status_label.setStyleSheet(
                "QLabel { "
                "background-color: #4CAF50; "
                "color: white; "
                "padding: 8px; "
                "border-radius: 4px; "
                "font-weight: bold; "
                "font-size: 14px; "
                "}"
            )
        else:
            self.status_label.setText("⏸ PAUSED")
            self.status_label.setStyleSheet(
                "QLabel { "
                "background-color: #FF9800; "
                "color: white; "
                "padding: 8px; "
                "border-radius: 4px; "
                "font-weight: bold; "
                "font-size: 14px; "
                "}"
            )
        self.counter_label.setText(str(self.engine.stats.today_count))
        
        conf_value = int(self.engine.stats.latest_confidence * 100)
        self.conf_meter.setValue(conf_value)
        
        if conf_value >= 80:
            color = "#4CAF50"
        elif conf_value >= 60:
            color = "#8BC34A"
        elif conf_value >= 40:
            color = "#FFC107"
        elif conf_value >= 20:
            color = "#FF9800"
        else:
            color = "#F44336"
        
        self.conf_meter.setStyleSheet(
            f"QProgressBar {{"
            f"border: 2px solid #ddd;"
            f"border-radius: 5px;"
            f"text-align: center;"
            f"background-color: #f0f0f0;"
            f"font-weight: bold;"
            f"}}"
            f"QProgressBar::chunk {{"
            f"background-color: {color};"
            f"border-radius: 3px;"
            f"}}"
        )

    def _events_df(self, start: str, end: str) -> pd.DataFrame:
        rows = self.db.events_for_range(start, end)
        return pd.DataFrame([dict(r) for r in rows])

    def load_analytics(self):
        d = self.date_picker.date().toPyDate().isoformat()
        day_df = self._events_df(d, d)
        summary = build_daily_summary(day_df)
        self.summary_label.setText(
            f"Events: {summary['event_count']} | Total: {summary['total_barking_sec']:.1f}s ({summary['total_barking_min']:.1f}m) "
            f"| Avg: {summary['avg_duration']:.2f}s | Longest: {summary['longest_event']:.2f}s | Common hour: {summary['most_common_hour']}"
        )

        self.hist_canvas.ax.clear()
        self.hist_canvas.ax.bar(range(24), summary["hourly_hist"])
        self.hist_canvas.ax.set_title("Hourly Barking Histogram")
        self.hist_canvas.draw()

        all_days = self._events_df("0001-01-01", "9999-12-31")
        totals = daily_totals(all_days)
        self.trend_canvas.ax.clear()
        if not totals.empty:
            self.trend_canvas.ax.plot(totals["day"], totals["duration_sec"], marker="o")
            self.trend_canvas.ax.tick_params(axis="x", rotation=45)
        self.trend_canvas.ax.set_title("Total Barking Time / Day")
        self.trend_canvas.draw()

        self.dist_canvas.ax.clear()
        if not day_df.empty:
            self.dist_canvas.ax.hist(day_df["duration_sec"], bins=10)
        self.dist_canvas.ax.set_title("Event Duration Distribution")
        self.dist_canvas.draw()

    def reload_days(self):
        self.day_list.clear()
        self.day_list.addItems(self.db.list_days())

    def delete_selected_day(self):
        item = self.day_list.currentItem()
        if not item:
            return
        if QMessageBox.question(self, "Confirm", f"Delete data for {item.text()}?") == QMessageBox.StandardButton.Yes:
            self.db.delete_day(item.text())
            self.reload_days()

    def delete_all(self):
        if QMessageBox.question(self, "Confirm", "Delete ALL bark data?") == QMessageBox.StandardButton.Yes:
            self.db.delete_all()
            self.reload_days()

    def export_selected_day(self):
        item = self.day_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Selection", "Please select a day to export.")
            return
        day = item.text()
        rows = self.db.events_for_range(day, day)
        
        if not rows:
            QMessageBox.information(self, "No Data", f"No bark events found for {day}.")
            return
        
        try:
            stamp = day.replace("-", "")
            json_path = APP_DIR / "exports" / f"barks_{stamp}.json"
            csv_path = APP_DIR / "exports" / f"barks_{stamp}.csv"
            
            self.db.export_json(rows, json_path)
            self.db.export_csv(rows, csv_path)
            
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("Export Successful")
            msg.setText(f"Exported {len(rows)} events for {day}")
            msg.setInformativeText(f"Files saved to:\n{APP_DIR / 'exports'}")
            msg.setStandardButtons(QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Ok)
            msg.setDefaultButton(QMessageBox.StandardButton.Ok)
            
            result = msg.exec()
            if result == QMessageBox.StandardButton.Open:
                self._open_exports_folder()
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{str(exc)}")

    def export_range(self):
        start = self.range_start.date().toPyDate().isoformat()
        end = self.range_end.date().toPyDate().isoformat()
        
        if start > end:
            QMessageBox.warning(self, "Invalid Range", "Start date must be before or equal to end date.")
            return
        
        rows = self.db.events_for_range(start, end)
        
        if not rows:
            QMessageBox.information(self, "No Data", f"No bark events found between {start} and {end}.")
            return
        
        try:
            stamp = f"{start}_to_{end}".replace("-", "")
            json_path = APP_DIR / "exports" / f"barks_{stamp}.json"
            csv_path = APP_DIR / "exports" / f"barks_{stamp}.csv"
            
            self.db.export_json(rows, json_path)
            self.db.export_csv(rows, csv_path)
            
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("Export Successful")
            msg.setText(f"Exported {len(rows)} events from {start} to {end}")
            msg.setInformativeText(f"Files saved to:\n{APP_DIR / 'exports'}")
            msg.setStandardButtons(QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Ok)
            msg.setDefaultButton(QMessageBox.StandardButton.Ok)
            
            result = msg.exec()
            if result == QMessageBox.StandardButton.Open:
                self._open_exports_folder()
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{str(exc)}")

    def _open_exports_folder(self):
        """Open the exports folder in the system file manager."""
        import subprocess
        import sys
        
        exports_path = APP_DIR / "exports"
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(exports_path)])
            elif sys.platform == "win32":
                subprocess.run(["explorer", str(exports_path)])
            else:
                subprocess.run(["xdg-open", str(exports_path)])
        except Exception as exc:
            QMessageBox.warning(self, "Cannot Open Folder", f"Could not open exports folder:\n{str(exc)}")

    def closeEvent(self, event):
        self.engine.stop()
        self.db.close()
        event.accept()
