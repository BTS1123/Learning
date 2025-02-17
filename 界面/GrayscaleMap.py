import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from stl import mesh
import os
import time

###  加载 STL 文件
def load_stl(file_path):
    """加载 STL 文件"""
    return mesh.Mesh.from_file(file_path)

###  计算长轴
def compute_long_axis(model):
    vertices = model.vectors.reshape(-1, 3)
    centroid = np.mean(vertices, axis=0)  # 计算质心
    inertia_tensor = np.zeros((3, 3))

    for vertex in vertices:
        relative_position = vertex - centroid
        inertia_tensor[0, 0] += (relative_position[1] ** 2 + relative_position[2] ** 2)
        inertia_tensor[1, 1] += (relative_position[0] ** 2 + relative_position[2] ** 2)
        inertia_tensor[2, 2] += (relative_position[0] ** 2 + relative_position[1] ** 2)
        inertia_tensor[0, 1] -= relative_position[0] * relative_position[1]
        inertia_tensor[1, 0] = inertia_tensor[0, 1]
        inertia_tensor[0, 2] -= relative_position[0] * relative_position[2]
        inertia_tensor[2, 0] = inertia_tensor[0, 2]
        inertia_tensor[1, 2] -= relative_position[1] * relative_position[2]
        inertia_tensor[2, 1] = inertia_tensor[1, 2]

    eigvals, eigvecs = np.linalg.eig(inertia_tensor)
    long_axis = eigvecs[:, np.argmax(eigvals)]  # 最大特征值对应的特征向量

    return centroid, long_axis

###  计算模型与平面的相交点
def get_intersection_section(model, plane_point, plane_normal):
    """获取 STL 牙齿模型的最大横向截面点"""
    section_points = []
    for triangle in model.vectors:
        intersection = []
        for i in range(3):
            p1 = triangle[i]
            p2 = triangle[(i + 1) % 3]
            t = np.dot(plane_point - p1, plane_normal) / np.dot(p2 - p1, plane_normal)
            if 0 <= t <= 1:
                intersection.append(p1 + t * (p2 - p1))
        if len(intersection) == 2:
            section_points.extend(intersection)

    return np.array(section_points)

###  计算截面面积（凸包计算）
def compute_section_area(section_points):
    """计算最大横截面的面积"""
    if len(section_points) < 3:
        return 0
    try:
        hull = ConvexHull(section_points)
        return hull.volume  # 近似为面积
    except:
        return 0  # 避免异常导致程序中断

###  查找最大横截面
def find_max_section(model, center, long_axis):
    """在不同 Z 轴高度寻找最大横截面"""
    z_min, z_max = np.min(model.vectors[:, :, 2]), np.max(model.vectors[:, :, 2])
    max_area = 0
    max_section_points = []
    max_plane_point = None

    for z in np.linspace(z_min, z_max, 100):  
        plane_point = center + z * long_axis
        section_points = get_intersection_section(model, plane_point, long_axis)
        if len(section_points) < 3:
            continue

        section_area = compute_section_area(section_points)
        if section_area > max_area:
            max_area = section_area
            max_section_points = section_points
            max_plane_point = plane_point

    if max_plane_point is None:
        max_plane_point = center  

    return max_section_points, max_plane_point

###  分割 STL 模型
def split_model(model, plane_point, plane_normal):
    """按最大横截面分割 STL 模型"""
    above_vertices, below_vertices = [], []

    for triangle in model.vectors:
        above, below = [], []
        for vertex in triangle:
            if np.dot(vertex - plane_point, plane_normal) > 0:
                above.append(vertex)
            else:
                below.append(vertex)

        if len(above) == 3:
            above_vertices.append(triangle)
        elif len(below) == 3:
            below_vertices.append(triangle)

    return np.array(above_vertices), np.array(below_vertices)

###  生成灰度热力图
def plot_gray_heatmap(vertices, section_point, long_axis, output_path, label):
    """生成未附着色彩的灰度热力图"""
    long_axis = long_axis / np.linalg.norm(long_axis)
    distances = np.dot(vertices - section_point, long_axis)
    min_distance, max_distance = distances.min(), distances.max()
    normalized_distances = (distances - min_distance) / (max_distance - min_distance)

    levels = np.linspace(0, 1, 100)

    arbitrary_vec = np.array([1, 0, 0]) if abs(long_axis[0]) < 0.9 else np.array([0, 1, 0])
    v1 = np.cross(long_axis, arbitrary_vec)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(long_axis, v1)

    projection_points = np.array([[np.dot(v - section_point, v1), np.dot(v - section_point, v2)] for v in vertices])

    x, y = projection_points[:, 0], projection_points[:, 1]
    z = normalized_distances

    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 500), np.linspace(y.min(), y.max(), 500))
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(8, 8))
    plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap="gray")

    plt.title(f"{label} Model Heatmap (Grayscale)")
    plt.xlabel("X-axis (projected)")
    plt.ylabel("Y-axis (projected)")
    plt.axis("equal")
    plt.grid(False)

    plt.savefig(os.path.join(output_path, f"{label}_gray_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

### 处理单个 STL 文件
def process_stl(file_path, output_heatmap_folder):
    """按照 `process_single_stl()` 逻辑处理 STL 并生成灰度图"""
    file_name_prefix = os.path.splitext(os.path.basename(file_path))[0]

    try:
        model = load_stl(file_path)
        center, long_axis = compute_long_axis(model)
        max_section_points, max_plane_point = find_max_section(model, center, long_axis)

        if max_plane_point is None:
            print(f"⚠️ 无法找到有效的最大截面，跳过: {file_path}")
            return

        upper, below = split_model(model, max_plane_point, long_axis)
        if len(upper) == 0 or len(below) == 0:
            print("❌ STL 分割失败，检查模型")
            return

        plot_gray_heatmap(upper.reshape(-1, 3), max_plane_point, long_axis, output_heatmap_folder, f"{file_name_prefix}_upper")
        plot_gray_heatmap(below.reshape(-1, 3), max_plane_point, long_axis, output_heatmap_folder, f"{file_name_prefix}_below")

        print(f"✅ 处理完成: {file_path}")
    except Exception as e:
        print(f"❌ 处理失败: {file_path}, 错误: {str(e)}")

# === 输入 & 输出路径定义 ===
INPUT_FOLDER = r"F:\【00002】24下\【0000】项目与比赛\国创\【000】代码实现部分\数据与存储\20岁年龄组36号牙"         # STL 文件所在文件夹
OUTPUT_FOLDER = r"F:\【00002】24下\【0000】项目与比赛\国创\【000】代码实现部分\数据与存储\存储生成图\灰度图"   # 生成的灰度图存放文件夹

# === 处理整个文件夹中的 STL 文件 ===
def batch_process_stl():
    """批量处理文件夹内的所有 STL 文件"""
    if not os.path.exists(INPUT_FOLDER):
        print("❌ 输入文件夹不存在，请检查路径")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    stl_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".stl")]
    total_files = len(stl_files)

    if not stl_files:
        print("⚠️ 输入文件夹中没有 STL 文件")
        return

    print(f"🔄 开始处理 {total_files} 个 STL 文件...")
    for idx, file_name in enumerate(stl_files, 1):
        file_path = os.path.join(INPUT_FOLDER, file_name)
        process_stl(file_path, OUTPUT_FOLDER)
        print(f"📌 进度: {idx}/{total_files} 文件处理完成... 剩余 {total_files - idx}")

    print("🎉 批量处理完成")

# === 运行脚本 ===
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # 确保输出目录存在
    batch_process_stl()
