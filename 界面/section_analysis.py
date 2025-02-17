import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from stl_processing import load_stl, save_stl, split_model, classify_parts,compute_surface_roughness
import matplotlib.pyplot as plt
import os

# 计算牙齿的惯性矩和纵向长轴
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

# 计算模型与平面相交的截面
def get_intersection_section(model, plane_point, plane_normal):
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

# 计算截面面积（凸包）
def compute_section_area(section_points):
    if len(section_points) < 3:
        return 0
    hull = ConvexHull(section_points, qhull_options='QJ')  # 解决共面问题
    return hull.volume

# 找最大截面
def find_max_section(model, center, long_axis):
    z_min = np.min(model.vectors[:, :, 2])
    z_max = np.max(model.vectors[:, :, 2])
    max_area = 0
    max_section_points = None
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

    # **如果 max_plane_point 为空，则使用模型中心点**
    if max_plane_point is None:
        max_plane_point = center

    return max_section_points, max_plane_point

# 分割模型
def split_model(model, plane_point, plane_normal, long_axis):
    """
    按照最大横向截面分割 STL 模型。
    - 计算模型与平面的交点，将其分为上下两部分。
    - 调用 `classify_parts()` 使得返回的 upper 始终是牙冠，below 始终是牙根。
    """
    above_vertices = []
    below_vertices = []

    for triangle in model.vectors:
        above = []
        below = []

        for vertex in triangle:
            if np.dot(vertex - plane_point, plane_normal) > 0:
                above.append(vertex)
            else:
                below.append(vertex)

        if len(above) == 3:
            above_vertices.append(triangle)
        elif len(below) == 3:
            below_vertices.append(triangle)

    upper, below = np.array(above_vertices), np.array(below_vertices)
    upper, below = classify_parts(upper, below)  

    return upper, below

# 绘制热力图
def plot_heatmap_on_section(vertices, section_point, long_axis, output_path, label):
    long_axis = long_axis / np.linalg.norm(long_axis)
    distances = np.dot(vertices - section_point, long_axis)
    min_distance, max_distance = distances.min(), distances.max()
    normalized_distances = (distances - min_distance) / (max_distance - min_distance)

    levels = np.linspace(0, 1, 100)
    cmap = plt.cm.get_cmap('RdYlBu_r')

    arbitrary_vec = np.array([1, 0, 0]) if abs(long_axis[0]) < 0.9 else np.array([0, 1, 0])
    v1 = np.cross(long_axis, arbitrary_vec)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(long_axis, v1)

    projection_points = []
    for vertex in vertices:
        relative_position = vertex - section_point
        x = np.dot(relative_position, v1)
        y = np.dot(relative_position, v2)
        projection_points.append([x, y])
    projection_points = np.array(projection_points)

    x, y = projection_points[:, 0], projection_points[:, 1]
    z = normalized_distances

    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 500), np.linspace(y.min(), y.max(), 500))
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(8, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap)
    plt.colorbar(contour, label="Normalized Distance to Section Plane (Z-axis)")
    plt.title(f"{label} Model Heatmap")
    plt.xlabel("X-axis (projected)")
    plt.ylabel("Y-axis (projected)")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"{label}_heatmap.png"))
    plt.close()
