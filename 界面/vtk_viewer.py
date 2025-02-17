#实现了一个 3D VTK 交互式查看器，用于展示 STL 文件
# 读取 STL 文件
# 创建 3D 物体、渲染器和交互窗口
# 启动 PyQt VTK 交互窗口
import vtkmodules.all as vtk
import tkinter as tk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget
import sys

class VTKViewer(QMainWindow):
    """
    3D VTK 交互式查看器，嵌入 Tkinter 的 Toplevel 窗口
    """
    def __init__(self, stl_file):
        super().__init__()

        self.setWindowTitle("STL 三维展示")
        self.setGeometry(100, 100, 800, 800)

        # 1️ **创建 VTK 窗口**
        self.frame = QWidget()
        self.layout = QVBoxLayout()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        # 2️ **读取 STL 文件**
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_file)
        reader.Update()

        # 3️ **创建 3D 物体**
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.8, 0.8)

        # 4️ **创建渲染器**
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.1, 0.1)

        # 5️ **创建 VTK 交互窗口**
        self.vtk_widget = QVTKRenderWindowInteractor(self.frame)
        self.layout.addWidget(self.vtk_widget)
        self.vtk_widget.GetRenderWindow().AddRenderer(renderer)

        # 6️ **交互控制**
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)
        self.iren.Initialize()
        self.iren.Start()

def open_vtk_viewer(stl_file):
    """
    启动 PyQt VTK 交互窗口
    """
    app = QApplication(sys.argv)
    viewer = VTKViewer(stl_file)
    viewer.show()
    app.exec_()
