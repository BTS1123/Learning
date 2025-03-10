import os
import time
from stl_processing import load_stl, save_stl, split_model
from section_analysis import compute_long_axis, find_max_section, plot_heatmap_on_section

def process_single_stl(file_path, output_stl_folder, output_heatmap_folder):
    """处理单个 STL 文件"""
    file_name_prefix = os.path.splitext(os.path.basename(file_path))[0]

    try:
        # **1. 加载 STL 文件**
        model = load_stl(file_path)

        # **2. 计算牙齿中心和主轴**
        center, long_axis = compute_long_axis(model)

        # **3. 计算最大截面**
        max_section_points, max_plane_point = find_max_section(model, center, long_axis)

        if max_plane_point is None:
            print(f"⚠️ 无法找到有效的最大截面，跳过: {file_path}")
            return

        # **4. 切割模型**
        upper, below = split_model(model, max_plane_point, long_axis)

        # **5. 保存 STL 文件**
        upper_stl_path = os.path.join(output_stl_folder, f"{file_name_prefix}_upper.stl")
        below_stl_path = os.path.join(output_stl_folder, f"{file_name_prefix}_below.stl")
        save_stl(upper, upper_stl_path)
        save_stl(below, below_stl_path)

        # **6. 生成并保存热力图**
        plot_heatmap_on_section(upper.reshape(-1, 3), max_plane_point, long_axis, output_heatmap_folder, f"{file_name_prefix}_upper")
        plot_heatmap_on_section(below.reshape(-1, 3), max_plane_point, long_axis, output_heatmap_folder, f"{file_name_prefix}_below")

        print(f"✅ 单个 STL 处理完成: {file_path}")
    except Exception as e:
        print(f"❌ 处理失败: {file_path}, 错误: {str(e)}")

def batch_process_stl(input_folder, output_stl_folder, output_heatmap_folder):
    """批量处理 STL 文件"""
    if not os.path.exists(input_folder):
        print("❌ 输入文件夹不存在，请检查路径")
        return

    os.makedirs(output_stl_folder, exist_ok=True)
    os.makedirs(output_heatmap_folder, exist_ok=True)

    stl_files = [f for f in os.listdir(input_folder) if f.endswith(".stl")]
    if not stl_files:
        print("⚠️ 输入文件夹中没有 STL 文件")
        return

    print(f"🔄 开始处理 {len(stl_files)} 个 STL 文件...")
    start_time = time.time()

    for file_name in stl_files:
        file_path = os.path.join(input_folder, file_name)
        process_single_stl(file_path, output_stl_folder, output_heatmap_folder)

    end_time = time.time()
    print(f"🎉 批量处理完成，总耗时: {end_time - start_time:.2f} 秒")
