import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert, correlate
from scipy import interpolate
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QPushButton, QFileDialog, QMessageBox, QLabel,
    QSplitter, QTextBrowser
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PhaseAnalyzer(QMainWindow):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
        self.selected_files = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('开搞！')
        self.setGeometry(100, 100, 1400, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        self.list_widget = QListWidget()
        self.list_widget.addItems(self.csv_files)
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        control_layout.addWidget(QLabel('选择两个要分析的CSV文件:'))
        control_layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        self.analyze_btn = QPushButton('分析相位差')
        self.analyze_btn.clicked.connect(self.analyze_phase_difference)
        self.clear_btn = QPushButton('清空选择')
        self.clear_btn.clicked.connect(self.clear_selection)
        button_layout.addWidget(self.analyze_btn)
        button_layout.addWidget(self.clear_btn)
        control_layout.addLayout(button_layout)

        folder_layout = QHBoxLayout()
        self.folder_label = QLabel(f'当前文件夹: {self.folder}')
        self.folder_label.setWordWrap(True)
        change_folder_btn = QPushButton('更改文件夹')
        change_folder_btn.clicked.connect(self.change_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(change_folder_btn)
        control_layout.insertLayout(0, folder_layout)

        self.info_browser = QTextBrowser()
        control_layout.addWidget(QLabel('文件信息:'))
        control_layout.addWidget(self.info_browser)

        self.metrics_label = QLabel('评价指标:')
        self.metrics_label.setAlignment(Qt.AlignTop)
        control_layout.addWidget(self.metrics_label)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        layout.addWidget(control_panel, stretch=1)
        layout.addWidget(plot_widget, stretch=4)

    def clear_selection(self):
        self.list_widget.clearSelection()
        self.info_browser.clear()
        self.metrics_label.setText('评价指标:')
        self.figure.clear()
        self.canvas.draw()

    def change_folder(self):
        new_folder = QFileDialog.getExistingDirectory(self, "选择包含CSV文件的文件夹")
        if new_folder:
            self.folder = new_folder
            self.csv_files = [f for f in os.listdir(new_folder) if f.endswith('.csv')]
            self.list_widget.clear()
            self.list_widget.addItems(self.csv_files)
            self.folder_label.setText(f'当前文件夹: {self.folder}')
            self.clear_selection()

    def validate_and_load_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"文件 {os.path.basename(file_path)} 的列名: {df.columns.tolist()}")
            print(f"数据形状: {df.shape}")
            if df.empty:
                raise ValueError("文件中没有数据")
            df.columns = df.columns.str.strip()
            time_column = next((col for col in df.columns if 'Time (s)' in col), None)
            if time_column is None:
                raise ValueError(f"找不到'Time (s)'列，当前列名: {', '.join(df.columns)}")
            u3_column = next((col for col in df.columns if 'U3' in col.strip()), None)
            if u3_column is None:
                raise ValueError(f"找不到U3列，当前列名: {', '.join(df.columns)}")
            time_data = df[time_column].values
            signal_data = df[u3_column].values
            if len(time_data) == 0 or len(signal_data) == 0:
                raise ValueError("时间或信号数据为空")
            if np.any(np.isnan(time_data)) or np.any(np.isnan(signal_data)):
                raise ValueError("数据中包含NaN值")
            print(f"成功加载文件 {os.path.basename(file_path)}")
            print(f"时间范围: [{time_data.min():.2f}, {time_data.max():.2f}]")
            print(f"数据点数: {len(time_data)}")
            return time_data, signal_data
        except Exception as e:
            raise ValueError(f"读取文件 {os.path.basename(file_path)} 时出错: {str(e)}")

    def resample_signals(self, time1, signal1, time2, signal2):
        try:
            if len(time1) == 0 or len(time2) == 0:
                raise ValueError("输入时间数组为空")
            if len(signal1) == 0 or len(signal2) == 0:
                raise ValueError("输入信号数组为空")
            start_time = max(np.min(time1), np.min(time2))
            end_time = min(np.max(time1), np.max(time2))
            if start_time >= end_time:
                raise ValueError("信号时间范围不重叠")
            num_points = max(len(time1), len(time2))
            new_time = np.linspace(start_time, end_time, num_points)
            f1 = interpolate.interp1d(time1, signal1, kind='linear', bounds_error=False, fill_value="extrapolate")
            f2 = interpolate.interp1d(time2, signal2, kind='linear', bounds_error=False, fill_value="extrapolate")
            new_signal1 = f1(new_time)
            new_signal2 = f2(new_time)
            if np.any(np.isnan(new_signal1)) or np.any(np.isnan(new_signal2)):
                raise ValueError("重采样后的数据包含NaN值")
            return new_time, new_signal1, new_signal2
        except Exception as e:
            raise ValueError(f"重采样信号时出错: {str(e)}")

    def analyze_phase_difference(self):
        selected_items = self.list_widget.selectedItems()
        if len(selected_items) != 2:
            QMessageBox.warning(self, "选择错误", "请精确选择两个文件进行分析！")
            return

        try:
            file1_path = os.path.join(self.folder, selected_items[0].text())
            file2_path = os.path.join(self.folder, selected_items[1].text())
            time1, signal1 = self.validate_and_load_data(file1_path)
            time2, signal2 = self.validate_and_load_data(file2_path)

            info_text = (
                f"文件1: {selected_items[0].text()}\n"
                f"- 采样点数: {len(time1)}\n"
                f"- 时间范围: [{np.min(time1):.2f}, {np.max(time1):.2f}] s\n"
                f"- U3范围: [{np.min(signal1):.4f}, {np.max(signal1):.4f}]\n\n"
                f"文件2: {selected_items[1].text()}\n"
                f"- 采样点数: {len(time2)}\n"
                f"- 时间范围: [{np.min(time2):.2f}, {np.max(time2):.2f}] s\n"
                f"- U3范围: [{np.min(signal2):.4f}, {np.max(signal2):.4f}]"
            )
            self.info_browser.setText(info_text)

            if len(time1) != len(time2) or not np.array_equal(time1, time2):
                print(f"正在重采样信号... (文件1: {len(time1)}点, 文件2: {len(time2)}点)")
                time, signal1, signal2 = self.resample_signals(time1, signal1, time2, signal2)
                print(f"重采样完成，统一为 {len(time)} 个点")
            else:
                time = time1

            analytic_signal1 = hilbert(signal1)
            analytic_signal2 = hilbert(signal2)
            if len(analytic_signal1) == 0 or len(analytic_signal2) == 0:
                raise ValueError("希尔伯特变换结果为空")
            phase1 = np.angle(analytic_signal1)
            phase2 = np.angle(analytic_signal2)
            phase_diff = phase1 - phase2
            unwrapped_phase_diff = np.unwrap(phase_diff)

            self.figure.clear()
            ax1 = self.figure.add_subplot(221)
            ax1.plot(time, signal1, label='Signal 1')
            ax1.plot(time, signal2, label='Signal 2')
            ax1.set_title('Original Signals')
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')

            ax2 = self.figure.add_subplot(222)
            ax2.plot(time, phase1, label='Phase 1')
            ax2.plot(time, phase2, label='Phase 2')
            ax2.set_title('Instantaneous Phases')
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Phase (rad)')

            ax3 = self.figure.add_subplot(223)
            ax3.plot(time, phase_diff)
            ax3.set_title('Phase Difference')
            ax3.grid(True)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Phase Difference (rad)')

            ax4 = self.figure.add_subplot(224)
            ax4.plot(time, unwrapped_phase_diff)
            ax4.set_title('Unwrapped Phase Difference')
            ax4.grid(True)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Unwrapped Phase (rad)')

            self.figure.tight_layout()
            self.canvas.draw()

            valid_mask = time >= 0.00006
            mean_phase_diff = np.mean(unwrapped_phase_diff[valid_mask])
            std_phase_diff = np.std(unwrapped_phase_diff[valid_mask])
            phase_correlation = np.corrcoef(phase1[valid_mask], phase2[valid_mask])[0,1]
            print(np.corrcoef(phase1[valid_mask], phase2[valid_mask]))

            # 均方根误差 (RMSE)
            rmse = np.sqrt(np.mean((signal1[valid_mask] - signal2[valid_mask]) ** 2))
            # 平均绝对误差 (MAE)
            mae = np.mean(np.abs(signal1[valid_mask] - signal2[valid_mask]))
            # 互相关系数
            cross_correlation = correlate(signal1[valid_mask], signal2[valid_mask], mode='full')
            cross_correlation = cross_correlation[len(cross_correlation)//2] / (np.std(signal1[valid_mask]) * np.std(signal2[valid_mask]) * len(signal1[valid_mask]))
            # 综合相似度得分
            weight_phase = 0.6
            weight_rmse = 0.2
            weight_cross = 0.2
            max_signal = max(np.max(np.abs(signal1)), np.max(np.abs(signal2)))
            normalized_rmse = 1 / (1 + rmse/max_signal) if max_signal > 0 else 0  # 归一化，值越小越好
            composite_score = (weight_phase * phase_correlation + weight_rmse * normalized_rmse + weight_cross * cross_correlation)

            stability_index = std_phase_diff / abs(mean_phase_diff) if abs(mean_phase_diff) > 1e-10 else float('inf')
            phase_diff_range = np.max(unwrapped_phase_diff[valid_mask]) - np.min(unwrapped_phase_diff[valid_mask])

            metrics_text = (
                "评价指标 (计算范围: Time ≥ 0.00006s):\n\n"
                f"1. 平均相位差: {mean_phase_diff:.4f} rad\n"
                f"2. 相位差标准差: {std_phase_diff:.4f} rad\n"
                f"3. 相位相关系数: {phase_correlation:.4f}\n"
                f"4. 均方根误差 (RMSE): {rmse:.4f}\n"
                f"5. 平均绝对误差 (MAE): {mae:.4f}\n"
                f"6. 互相关系数: {cross_correlation:.4f}\n"
                f"7. 相位差稳定性: {stability_index:.4f}\n"
                f"8. 相位差范围: {phase_diff_range:.4f} rad\n"
                f"9. 综合相似度得分: {composite_score:.4f}\n\n"
                "信号相似度评估:\n"
                f"{'非常相似' if composite_score > 0.5 else '相似' if composite_score > 0.3 else '一般相似' if composite_score > 0.1 else '差异较大'}\n\n"
                "相位相关系数评估:\n"
                f"{'非常相似' if phase_correlation > 0.4 else '相似' if phase_correlation > 0.25 else '一般相似' if phase_correlation > 0.1 else '差异较大'}\n\n"
                "注释:\n"
                "- 相位相关系数越接近1，信号相位越相似\n"
                "- RMSE和MAE越小，幅度差异越小\n"
                "- 互相关系数越接近1，时间域相似度越高\n"
                "- 综合相似度得分越高，信号越接近\n"
            )
            self.metrics_label.setText(metrics_text)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"分析过程中出现错误：{str(e)}\n\n"
                               "请检查：\n"
                               "1. CSV文件是否包含有效数据\n"
                               "2. 列名是否正确（应为'Time (s)'和'U3'）\n"
                               "3. 数据是否包含无效值（如NaN）")
            print(f"详细错误信息：{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    folder = QFileDialog.getExistingDirectory(None, "选择包含CSV文件的文件夹")
    if folder:
        window = PhaseAnalyzer(folder)
        window.show()
        sys.exit(app.exec_())