#包含加载 STL 文件和计算表面粗糙度的函数
import numpy as np
from scipy.spatial import KDTree
from stl import mesh

# 加载 STL 模型
def load_stl(file_path):
    """加载 STL 文件"""
    return mesh.Mesh.from_file(file_path)

# 保存 STL 模型
def save_stl(vertices, file_path):
    """保存 STL 文件"""
    output_mesh = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    output_mesh.vectors = vertices
    output_mesh.save(file_path)

# 计算表面粗糙度（法向量角度变化）
def compute_surface_roughness(part):
    """计算 STL 片段的表面粗糙度（基于相邻法向量的夹角变化）"""
    if part.shape[0] == 0:
        return 0  # 避免空输入

    normals = np.cross(part[:, 1] - part[:, 0], part[:, 2] - part[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # 单位化法向量

    roughness = 0
    count = 0
    for i in range(len(normals) - 1):
        angle = np.arccos(np.clip(np.dot(normals[i], normals[i + 1]), -1.0, 1.0))
        roughness += angle
        count += 1

    return roughness / count if count > 0 else 0

# 计算 Z 轴高度变化（标准差）
def compute_height_variation(part):
    """计算 STL 片段的 Z 轴坐标标准差（表面起伏度）"""
    if part.shape[0] == 0:
        return 0  # 避免空输入

    z_values = part.reshape(-1, 3)[:, 2]
    return np.std(z_values)

# 计算 STL 片段的平均曲率
def compute_curvature(part):
    """计算 STL 片段的平均曲率（基于法向量夹角平方）"""
    if part.shape[0] == 0:
        return 0  # 避免空输入

    normals = np.cross(part[:, 1] - part[:, 0], part[:, 2] - part[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    curvature = 0
    count = 0
    for i in range(len(normals) - 1):
        angle = np.arccos(np.clip(np.dot(normals[i], normals[i + 1]), -1.0, 1.0))
        curvature += angle ** 2
        count += 1

    return curvature / count if count > 0 else 0

# 计算 STL 片段的边缘密度（平均三角形边长）
def compute_edge_density(part):
    """计算 STL 片段的边缘密度（牙冠的 STL 边缘通常比牙根复杂）"""
    if part.shape[0] == 0:
        return 0  # 避免空输入

    edges = np.linalg.norm(part[:, 1] - part[:, 0], axis=1) + \
            np.linalg.norm(part[:, 2] - part[:, 1], axis=1) + \
            np.linalg.norm(part[:, 0] - part[:, 2], axis=1)
    
    return np.mean(edges)  # 计算平均边长

# 通过多个几何特征判断牙冠和牙根
def classify_parts(upper, below):
    """
    结合多个特征判断牙冠和牙根：
    1. 计算法向量角度变化
    2. 计算 Z 轴坐标标准差
    3. 计算平均曲率
    4. 计算边缘密度
    5. 综合评分进行最终分类
    """
    roughness_upper_normal = compute_surface_roughness(upper)
    roughness_below_normal = compute_surface_roughness(below)

    roughness_upper_height = compute_height_variation(upper)
    roughness_below_height = compute_height_variation(below)

    curvature_upper = compute_curvature(upper)
    curvature_below = compute_curvature(below)

    edge_density_upper = compute_edge_density(upper)
    edge_density_below = compute_edge_density(below)

    # 综合计算最终粗糙度（加权计算）
    roughness_upper = (0.3 * roughness_upper_normal + 
                       0.3 * roughness_upper_height + 
                       0.2 * curvature_upper + 
                       0.2 * edge_density_upper)
    
    roughness_below = (0.3 * roughness_below_normal + 
                       0.3 * roughness_below_height + 
                       0.2 * curvature_below + 
                       0.2 * edge_density_below)

    # 打印分类信息
    print("=== 牙冠与牙根分类信息 ===")
    print(f"🔹 upper 法向量粗糙度: {roughness_upper_normal:.5f}, 高度变化: {roughness_upper_height:.5f}, 曲率: {curvature_upper:.5f}, 边缘密度: {edge_density_upper:.5f}, 综合: {roughness_upper:.5f}")
    print(f"🔹 below 法向量粗糙度: {roughness_below_normal:.5f}, 高度变化: {roughness_below_height:.5f}, 曲率: {curvature_below:.5f}, 边缘密度: {edge_density_below:.5f}, 综合: {roughness_below:.5f}")

    # 设定阈值，防止小范围误差导致分类错误
    THRESHOLD = 0.02  

    if roughness_upper - roughness_below > THRESHOLD:
        print("✅ 分类结果: upper 为牙冠，below 为牙根")
        return upper, below  # 牙冠是 upper，牙根是 below
    else:
        print("🔄 交换分类: below 为牙冠，upper 为牙根")
        return below, upper  # 交换，使牙冠始终是 upper，牙根是 below

# 分割模型
def split_model(model, plane_point, plane_normal):
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
    upper, below = classify_parts(upper, below)  # 重新分类

    return upper, below
