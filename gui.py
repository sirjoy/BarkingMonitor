from __future__ import annotations

from datetime import date

import pandas as pd
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from analytics import build_daily_summary, build_thunder_summary, analyze_bark_thunder_correlation, daily_totals
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

        status_box = QGroupBox("Barking Status")
        form = QFormLayout(status_box)
        self.status_label = QLabel("Paused")
        self.counter_label = QLabel("0")
        self.conf_meter = QProgressBar()
        self.conf_meter.setRange(0, 100)
        self.conf_meter.setFormat("%p%")
        self.conf_meter.setTextVisible(True)
        self.conf_meter.setMinimumHeight(25)
        self.conf_debug = QLabel("0.000")
        self.conf_debug.setStyleSheet("font-size: 10px; color: gray;")
        form.addRow("State", self.status_label)
        form.addRow("Today's Barking Events", self.counter_label)
        form.addRow("Confidence", self.conf_meter)
        form.addRow("Raw Confidence", self.conf_debug)

        thunder_status_box = QGroupBox("Thunder Status")
        thunder_form = QFormLayout(thunder_status_box)
        self.thunder_counter_label = QLabel("0")
        self.thunder_conf_meter = QProgressBar()
        self.thunder_conf_meter.setRange(0, 100)
        self.thunder_conf_meter.setFormat("%p%")
        self.thunder_conf_meter.setTextVisible(True)
        self.thunder_conf_meter.setMinimumHeight(25)
        self.thunder_conf_debug = QLabel("0.000")
        self.thunder_conf_debug.setStyleSheet("font-size: 10px; color: gray;")
        thunder_form.addRow("Today's Thunder Events", self.thunder_counter_label)
        thunder_form.addRow("Confidence", self.thunder_conf_meter)
        thunder_form.addRow("Raw Confidence", self.thunder_conf_debug)

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
        
        self.thunder_threshold_spin = QDoubleSpinBox()
        self.thunder_threshold_spin.setRange(0.1, 1.0)
        self.thunder_threshold_spin.setSingleStep(0.01)
        self.thunder_threshold_spin.setValue(self.config.thunder_detection_threshold)
        self.thunder_cooldown_spin = QDoubleSpinBox()
        self.thunder_cooldown_spin.setRange(0.5, 10.0)
        self.thunder_cooldown_spin.setSingleStep(0.5)
        self.thunder_cooldown_spin.setValue(self.config.thunder_cooldown_seconds)
        
        self.auto_start_check = QCheckBox("Auto-start on launch")
        self.auto_start_check.setChecked(self.config.auto_start)
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        cfg_form.addRow("Bark Threshold", self.threshold_spin)
        cfg_form.addRow("Bark Cooldown (s)", self.cooldown_spin)
        cfg_form.addRow("Thunder Threshold", self.thunder_threshold_spin)
        cfg_form.addRow("Thunder Cooldown (s)", self.thunder_cooldown_spin)
        cfg_form.addRow(self.auto_start_check)
        cfg_form.addRow(save_btn)

        layout.addWidget(status_box)
        layout.addWidget(thunder_status_box)
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
        
        self.event_type_combo = QComboBox()
        self.event_type_combo.addItems(["Barking Events", "Thunder Events", "Correlation Analysis"])
        
        refresh_btn = QPushButton("Load Date")
        refresh_btn.clicked.connect(self.load_analytics)
        top.addWidget(QLabel("Date"))
        top.addWidget(self.date_picker)
        top.addWidget(QLabel("Event Type"))
        top.addWidget(self.event_type_combo)
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
        
        bark_group = QGroupBox("Bark Events Data")
        bark_layout = QVBoxLayout(bark_group)
        self.day_list = QListWidget()
        bark_layout.addWidget(QLabel("Days with bark events:"))
        bark_layout.addWidget(self.day_list)
        
        thunder_group = QGroupBox("Thunder Events Data")
        thunder_layout = QVBoxLayout(thunder_group)
        self.thunder_day_list = QListWidget()
        thunder_layout.addWidget(QLabel("Days with thunder events:"))
        thunder_layout.addWidget(self.thunder_day_list)
        
        self.range_start = QDateEdit()
        self.range_start.setCalendarPopup(True)
        self.range_start.setDate(date.today())
        self.range_end = QDateEdit()
        self.range_end.setCalendarPopup(True)
        self.range_end.setDate(date.today())

        bark_btn_row = QHBoxLayout()
        delete_btn = QPushButton("Delete Selected Bark Day")
        delete_btn.clicked.connect(self.delete_selected_day)
        export_day_btn = QPushButton("Export Selected Bark Day")
        export_day_btn.clicked.connect(self.export_selected_day)
        export_range_btn = QPushButton("Export Bark Range")
        export_range_btn.clicked.connect(self.export_range)
        clear_btn = QPushButton("Delete All Bark Data")
        clear_btn.clicked.connect(self.delete_all)
        bark_btn_row.addWidget(delete_btn)
        bark_btn_row.addWidget(export_day_btn)
        bark_btn_row.addWidget(export_range_btn)
        bark_btn_row.addWidget(clear_btn)
        
        thunder_btn_row = QHBoxLayout()
        delete_thunder_btn = QPushButton("Delete Selected Thunder Day")
        delete_thunder_btn.clicked.connect(self.delete_selected_thunder_day)
        export_thunder_day_btn = QPushButton("Export Selected Thunder Day")
        export_thunder_day_btn.clicked.connect(self.export_selected_thunder_day)
        export_thunder_range_btn = QPushButton("Export Thunder Range")
        export_thunder_range_btn.clicked.connect(self.export_thunder_range)
        clear_thunder_btn = QPushButton("Delete All Thunder Data")
        clear_thunder_btn.clicked.connect(self.delete_all_thunder)
        thunder_btn_row.addWidget(delete_thunder_btn)
        thunder_btn_row.addWidget(export_thunder_day_btn)
        thunder_btn_row.addWidget(export_thunder_range_btn)
        thunder_btn_row.addWidget(clear_thunder_btn)

        layout.addWidget(bark_group)
        layout.addWidget(thunder_group)
        layout.addWidget(QLabel("Range Start"))
        layout.addWidget(self.range_start)
        layout.addWidget(QLabel("Range End"))
        layout.addWidget(self.range_end)
        layout.addLayout(bark_btn_row)
        layout.addLayout(thunder_btn_row)
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
        self.config.thunder_detection_threshold = self.thunder_threshold_spin.value()
        self.config.thunder_cooldown_seconds = self.thunder_cooldown_spin.value()
        self.config.auto_start = self.auto_start_check.isChecked()
        self.engine.detector.threshold = self.config.detection_threshold
        if self.engine.thunder_detector:
            self.engine.thunder_detector.threshold = self.config.thunder_detection_threshold
        self.engine.config.cooldown_seconds = self.config.cooldown_seconds
        self.engine.config.thunder_cooldown_seconds = self.config.thunder_cooldown_seconds
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
        self.conf_debug.setText(f"{self.engine.stats.latest_confidence:.3f}")
        
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
        
        self.thunder_counter_label.setText(str(self.engine.stats.today_thunder_count))
        thunder_conf_value = int(self.engine.stats.latest_thunder_confidence * 100)
        self.thunder_conf_meter.setValue(thunder_conf_value)
        self.thunder_conf_meter.setFormat(f"{thunder_conf_value}%")
        self.thunder_conf_debug.setText(f"{self.engine.stats.latest_thunder_confidence:.3f}")
        
        if thunder_conf_value >= 80:
            thunder_color = "#4CAF50"
        elif thunder_conf_value >= 60:
            thunder_color = "#8BC34A"
        elif thunder_conf_value >= 40:
            thunder_color = "#FFC107"
        elif thunder_conf_value >= 20:
            thunder_color = "#FF9800"
        else:
            thunder_color = "#F44336"
        
        self.thunder_conf_meter.setStyleSheet(
            f"QProgressBar {{"
            f"border: 2px solid #ddd;"
            f"border-radius: 5px;"
            f"text-align: center;"
            f"background-color: #f0f0f0;"
            f"font-weight: bold;"
            f"}}"
            f"QProgressBar::chunk {{"
            f"background-color: {thunder_color};"
            f"border-radius: 3px;"
            f"}}"
        )

    def _events_df(self, start: str, end: str) -> pd.DataFrame:
        rows = self.db.events_for_range(start, end)
        return pd.DataFrame([dict(r) for r in rows])
    
    def _thunder_events_df(self, start: str, end: str) -> pd.DataFrame:
        """Get thunder events as DataFrame for date range.
        
        Args:
            start: Start date in ISO format
            end: End date in ISO format
            
        Returns:
            DataFrame with thunder event data
        """
        rows = self.db.events_for_thunder_range(start, end)
        return pd.DataFrame([dict(r) for r in rows])

    def load_analytics(self):
        d = self.date_picker.date().toPyDate().isoformat()
        event_type = self.event_type_combo.currentText()
        
        if event_type == "Barking Events":
            self._load_bark_analytics(d)
        elif event_type == "Thunder Events":
            self._load_thunder_analytics(d)
        else:
            self._load_correlation_analytics(d)
    
    def _load_bark_analytics(self, d: str):
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
    
    def _load_thunder_analytics(self, d: str):
        day_df = self._thunder_events_df(d, d)
        summary = build_thunder_summary(day_df)
        self.summary_label.setText(
            f"Thunder Events: {summary['event_count']} | Total: {summary['total_thunder_sec']:.1f}s ({summary['total_thunder_min']:.1f}m) "
            f"| Avg: {summary['avg_duration']:.2f}s | Longest: {summary['longest_event']:.2f}s | Common hour: {summary['most_common_hour']}"
        )

        self.hist_canvas.ax.clear()
        self.hist_canvas.ax.bar(range(24), summary["hourly_hist"])
        self.hist_canvas.ax.set_title("Hourly Thunder Histogram")
        self.hist_canvas.draw()

        all_days = self._thunder_events_df("0001-01-01", "9999-12-31")
        totals = daily_totals(all_days)
        self.trend_canvas.ax.clear()
        if not totals.empty:
            self.trend_canvas.ax.plot(totals["day"], totals["duration_sec"], marker="o", color="purple")
            self.trend_canvas.ax.tick_params(axis="x", rotation=45)
        self.trend_canvas.ax.set_title("Total Thunder Time / Day")
        self.trend_canvas.draw()

        self.dist_canvas.ax.clear()
        if not day_df.empty:
            self.dist_canvas.ax.hist(day_df["duration_sec"], bins=10, color="purple")
        self.dist_canvas.ax.set_title("Thunder Duration Distribution")
        self.dist_canvas.draw()
    
    def _load_correlation_analytics(self, d: str):
        bark_df = self._events_df(d, d)
        thunder_df = self._thunder_events_df(d, d)
        correlation = analyze_bark_thunder_correlation(bark_df, thunder_df)
        
        self.summary_label.setText(
            f"Correlation Analysis for {d}\n"
            f"Bark events near thunder: {correlation['overlapping_events']} "
            f"({correlation['bark_during_thunder_ratio']*100:.1f}% of all barks) | "
            f"Avg barks per thunder: {correlation['avg_barks_per_thunder']:.2f} | "
            f"Bark reduction during thunder: {correlation['bark_frequency_change']:.1f}%"
        )
        
        self.hist_canvas.ax.clear()
        if not bark_df.empty and not thunder_df.empty:
            bark_summary = build_daily_summary(bark_df)
            thunder_summary = build_thunder_summary(thunder_df)
            x = range(24)
            width = 0.35
            self.hist_canvas.ax.bar([i - width/2 for i in x], bark_summary["hourly_hist"], width, label="Barking")
            self.hist_canvas.ax.bar([i + width/2 for i in x], thunder_summary["hourly_hist"], width, label="Thunder", color="purple")
            self.hist_canvas.ax.legend()
        self.hist_canvas.ax.set_title("Hourly Comparison: Bark vs Thunder")
        self.hist_canvas.draw()
        
        self.trend_canvas.ax.clear()
        all_barks = self._events_df("0001-01-01", "9999-12-31")
        all_thunder = self._thunder_events_df("0001-01-01", "9999-12-31")
        bark_totals = daily_totals(all_barks)
        thunder_totals = daily_totals(all_thunder)
        if not bark_totals.empty:
            self.trend_canvas.ax.plot(bark_totals["day"], bark_totals["duration_sec"], marker="o", label="Barking")
        if not thunder_totals.empty:
            self.trend_canvas.ax.plot(thunder_totals["day"], thunder_totals["duration_sec"], marker="s", color="purple", label="Thunder")
        self.trend_canvas.ax.legend()
        self.trend_canvas.ax.tick_params(axis="x", rotation=45)
        self.trend_canvas.ax.set_title("Daily Trends: Bark vs Thunder")
        self.trend_canvas.draw()
        
        self.dist_canvas.ax.clear()
        if correlation['temporal_matches']:
            self.dist_canvas.ax.bar([0, 1], [correlation['overlapping_events'], len(bark_df) - correlation['overlapping_events']], 
                                   tick_label=['Barks near thunder', 'Barks away from thunder'])
        self.dist_canvas.ax.set_title("Bark Distribution Relative to Thunder")
        self.dist_canvas.draw()

    def reload_days(self):
        self.day_list.clear()
        self.day_list.addItems(self.db.list_days())
        self.thunder_day_list.clear()
        self.thunder_day_list.addItems(self.db.list_thunder_days())

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
    
    def delete_selected_thunder_day(self):
        """Delete selected thunder event day from database."""
        item = self.thunder_day_list.currentItem()
        if not item:
            return
        if QMessageBox.question(self, "Confirm", f"Delete thunder data for {item.text()}?") == QMessageBox.StandardButton.Yes:
            self.db.delete_thunder_day(item.text())
            self.reload_days()

    def delete_all_thunder(self):
        """Delete all thunder events from database."""
        if QMessageBox.question(self, "Confirm", "Delete ALL thunder data?") == QMessageBox.StandardButton.Yes:
            self.db.delete_all_thunder()
            self.reload_days()

    def export_selected_thunder_day(self):
        """Export selected thunder event day to JSON and CSV."""
        item = self.thunder_day_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Selection", "Please select a day to export.")
            return
        day = item.text()
        rows = self.db.events_for_thunder_range(day, day)
        
        if not rows:
            QMessageBox.information(self, "No Data", f"No thunder events found for {day}.")
            return
        
        try:
            stamp = day.replace("-", "")
            json_path = APP_DIR / "exports" / f"thunder_{stamp}.json"
            csv_path = APP_DIR / "exports" / f"thunder_{stamp}.csv"
            
            self.db.export_json(rows, json_path)
            self.db.export_csv(rows, csv_path)
            
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("Export Successful")
            msg.setText(f"Exported {len(rows)} thunder events for {day}")
            msg.setInformativeText(f"Files saved to:\n{APP_DIR / 'exports'}")
            msg.setStandardButtons(QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Ok)
            msg.setDefaultButton(QMessageBox.StandardButton.Ok)
            
            result = msg.exec()
            if result == QMessageBox.StandardButton.Open:
                self._open_exports_folder()
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{str(exc)}")

    def export_thunder_range(self):
        """Export thunder events for date range to JSON and CSV."""
        start = self.range_start.date().toPyDate().isoformat()
        end = self.range_end.date().toPyDate().isoformat()
        
        if start > end:
            QMessageBox.warning(self, "Invalid Range", "Start date must be before or equal to end date.")
            return
        
        rows = self.db.events_for_thunder_range(start, end)
        
        if not rows:
            QMessageBox.information(self, "No Data", f"No thunder events found between {start} and {end}.")
            return
        
        try:
            stamp = f"{start}_to_{end}".replace("-", "")
            json_path = APP_DIR / "exports" / f"thunder_{stamp}.json"
            csv_path = APP_DIR / "exports" / f"thunder_{stamp}.csv"
            
            self.db.export_json(rows, json_path)
            self.db.export_csv(rows, csv_path)
            
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("Export Successful")
            msg.setText(f"Exported {len(rows)} thunder events from {start} to {end}")
            msg.setInformativeText(f"Files saved to:\n{APP_DIR / 'exports'}")
            msg.setStandardButtons(QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Ok)
            msg.setDefaultButton(QMessageBox.StandardButton.Ok)
            
            result = msg.exec()
            if result == QMessageBox.StandardButton.Open:
                self._open_exports_folder()
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{str(exc)}")

    def closeEvent(self, event):
        self.engine.stop()
        self.db.close()
        event.accept()
