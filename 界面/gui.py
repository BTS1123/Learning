import os
import sys
import time  
import tkinter as tk
import batch_process
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from vtk_viewer import open_vtk_viewer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from stl_processing import load_stl, save_stl, split_model
from section_analysis import compute_long_axis, find_max_section, plot_heatmap_on_section

class STLProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("STL File Processing Tool")
        self.root.geometry("1080x800")
        self.root.resizable(True, True)

        # 确保关闭界面时结束进程
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 变量：存储当前模式（单独处理 / 批量处理）
        self.mode = "single"  # 默认是单独处理模式

        # 文件地址变量（单独 / 批量独立）
        self.selected_file = None
        self.selected_folder = None
        self.stl_save_directory_single = None
        self.stl_save_directory_batch = None
        self.heatmap_save_directory_single = None
        self.heatmap_save_directory_batch = None

        self.vtk_widget = None
        self.create_widgets()
        
    def switch_mode(self):
        """切换处理模式"""
        if self.mode == "single":
            self.mode = "batch"
            messagebox.showinfo("模式切换", "已切换到【批量处理模式】")
        else:
            self.mode = "single"
            messagebox.showinfo("模式切换", "已切换到【单独处理模式】")

        # 更新按钮
        self.update_buttons()

    def update_buttons(self):
        """更新按钮布局"""
        # 清除原有按钮
        for widget in self.control_frame.winfo_children():
            widget.destroy()

        # 滑动开关按钮
        self.switch_var = tk.IntVar(value=0 if self.mode == "single" else 1)
        self.switch_btn = tk.Checkbutton(
            self.control_frame, text="单独处理 / 批量处理",
            variable=self.switch_var, command=self.switch_mode,
            indicatoron=False, selectcolor="lightblue", height=2, width=20
        )
        self.switch_btn.pack(pady=10, fill=tk.X, padx=10)

        # 切换按钮布局
        if self.mode == "single":
            self.btn_select_file = tk.Button(self.control_frame, text="请选择文件处理", command=self.select_stl_file)
            self.btn_select_file.pack(pady=10, fill=tk.X, padx=10)
        else:  # 批量模式
            self.btn_batch_process = tk.Button(self.control_frame, text="请选择文件夹批量处理", command=self.batch_process)
            self.btn_batch_process.pack(pady=10, fill=tk.X, padx=10)

        # 选择STL存储位置
        self.btn_select_stl_save = tk.Button(self.control_frame, text="选择STL文件存储位置", command=self.select_stl_save_directory)
        self.btn_select_stl_save.pack(pady=10, fill=tk.X, padx=10)

        # 选择热力图存储位置
        self.btn_select_heatmap_save = tk.Button(self.control_frame, text="选择热力图生成位置", command=self.select_heatmap_save_directory)
        self.btn_select_heatmap_save.pack(pady=10, fill=tk.X, padx=10)

        # 处理文件
        self.btn_process_file = tk.Button(self.control_frame, text="处理文件", command=self.process_file)
        self.btn_process_file.pack(pady=10, fill=tk.X, padx=10)

        # 打开投影图
        self.btn_open_heatmap = tk.Button(self.control_frame, text="打开投影图", command=self.open_heatmap)
        self.btn_open_heatmap.pack(pady=10, fill=tk.X, padx=10)
        
        self.btn_view_stl = tk.Button(self.control_frame, text="STL文件三维展示", command=self.select_and_view_stl)
        self.btn_view_stl.pack(pady=10, fill=tk.X, padx=10)

    def create_widgets(self):
        """创建 GUI 界面组件"""
        self.canvas_frame = tk.Frame(self.root, bg="white", width=800, height=800)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(self.root, bg="lightgray", width=280)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        
        # 初始加载单独处理模式
        self.update_buttons()

    def select_and_view_stl(self):
        """选择 STL 文件（用于三维展示）"""
        self.selected_view_file = filedialog.askopenfilename(filetypes=[("STL Files", "*.stl")])
        if self.selected_view_file:
            open_vtk_viewer(self.selected_view_file)

    def select_stl_file(self):
        """选择 STL 文件"""
        self.selected_file = filedialog.askopenfilename(filetypes=[("STL Files", "*.stl")])
        if self.selected_file:
            messagebox.showinfo("文件已选择", f"已选择文件: {self.selected_file}")

    def select_stl_save_directory(self):
        """选择 STL 文件存储路径"""
        folder = filedialog.askdirectory()
        if folder:
            if self.mode == "single":
                self.stl_save_directory_single = folder
                messagebox.showinfo("保存路径", f"单独处理的 STL 存储路径: {folder}")
            else:
                self.stl_save_directory_batch = folder
                messagebox.showinfo("保存路径", f"批量处理的 STL 存储路径: {folder}")

    def select_heatmap_save_directory(self):
        """选择热力图存储路径"""
        folder = filedialog.askdirectory()
        if folder:
            if self.mode == "single":
                self.heatmap_save_directory_single = folder
                messagebox.showinfo("保存路径", f"单独处理的热力图存储路径: {folder}")
            else:
                self.heatmap_save_directory_batch = folder
                messagebox.showinfo("保存路径", f"批量处理的热力图存储路径: {folder}")
    def open_heatmap(self):
        """打开并显示热力图"""
        heatmap_file = filedialog.askopenfilename(filetypes=[("PNG Files", "*.png")])
        if not heatmap_file:
            messagebox.showwarning("警告", "请先选择投影图文件！")
            return

        try:
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()

            fig, ax = plt.subplots(figsize=(6, 6))
            img = plt.imread(heatmap_file)
            ax.imshow(img)
            ax.axis('off')

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开投影图: {str(e)}")

    import time  # **导入 time 模块用于模拟进度更新**

    def process_file(self):
        """处理 STL 文件（单独 / 批量），带进度窗口"""
        if self.mode == "single":
            if not self.selected_file:
                messagebox.showwarning("警告", "请先选择 STL 文件！")
                return
            if not self.stl_save_directory_single or not self.heatmap_save_directory_single:
                messagebox.showwarning("警告", "请先选择 STL 和 热力图的存储位置！")
                return

            # **创建进度窗口**
            progress_window = tk.Toplevel(self.root)
            progress_window.title("处理进度")
            progress_window.geometry("400x150")

            progress_label = tk.Label(progress_window, text="正在处理，请稍候...")
            progress_label.pack(pady=10)

            progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
            progress_bar.pack(pady=10, fill=tk.X, padx=20)

            progress_bar["value"] = 0
            self.root.update_idletasks()

            # **处理 STL 文件，模拟进度更新**
            for progress in range(1, 100, 2):  # **逐步增加进度**
                time.sleep(0.5)  # **模拟处理时间**
                progress_bar["value"] = progress
                progress_label.config(text=f"进度: {progress}%")
                self.root.update_idletasks()

            batch_process.process_single_stl(self.selected_file, self.stl_save_directory_single, self.heatmap_save_directory_single)

            # **确保最终 100%**
            progress_bar["value"] = 100
            progress_label.config(text="处理完成！")
            self.root.update_idletasks()

            messagebox.showinfo("完成", "单个文件处理完成！")
            progress_window.destroy()

        else:
            if not self.selected_folder:
                messagebox.showwarning("警告", "请先选择批量处理的文件夹！")
                return
            if not self.stl_save_directory_batch or not self.heatmap_save_directory_batch:
                messagebox.showwarning("警告", "请先选择 STL 和 热力图的存储位置！")
                return

            # **获取待处理的 STL 文件数**
            stl_files = [f for f in os.listdir(self.selected_folder) if f.endswith(".stl")]
            total_files = len(stl_files)
            if total_files == 0:
                messagebox.showwarning("警告", "未找到 STL 文件！")
                return

            # **创建进度窗口**
            progress_window = tk.Toplevel(self.root)
            progress_window.title("处理进度")
            progress_window.geometry("400x150")

            progress_label = tk.Label(progress_window, text="正在批量处理，请稍候...")
            progress_label.pack(pady=10)

            progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
            progress_bar.pack(pady=10, fill=tk.X, padx=20)

            progress_bar["value"] = 0
            self.root.update_idletasks()

            for idx, file_name in enumerate(stl_files):
                file_path = os.path.join(self.selected_folder, file_name)

                # **处理 STL 文件**
                batch_process.process_single_stl(file_path, self.stl_save_directory_batch, self.heatmap_save_directory_batch)

                # **进度条分阶段更新**
                for progress in range(int((idx / total_files) * 100), int(((idx + 1) / total_files) * 100), 20):
                    time.sleep(0.3)  # **模拟处理时间**
                    progress_bar["value"] = progress
                    progress_label.config(text=f"进度: {progress}%")
                    self.root.update_idletasks()

            # **确保最终 100%**
            progress_bar["value"] = 100
            progress_label.config(text="处理完成！")
            self.root.update_idletasks()

            messagebox.showinfo("完成", "批量处理已完成！")
            progress_window.destroy()


    def batch_process(self):
        """选择文件夹进行批量处理"""
        folder = filedialog.askdirectory(title="选择包含 STL 文件的文件夹")
        if folder:
            self.selected_folder = folder
            messagebox.showinfo("文件夹已选择", f"批量处理目标文件夹: {folder}")

    def on_close(self):
        """关闭窗口并终止程序"""
        self.root.destroy()
        os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = STLProcessingApp(root)
    root.mainloop()
