import os
import sys

sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
import numpy as np

def write_array_to_file(field, file_handle):
    """
    将NumPy数组每5个元素为一组写入文件，每组占一行。在文件末尾添加一个换行符。
    
    参数:
    - field: NumPy数组，维度为[10000]，类型为np.float32。
    - filename: 要写入数据的文件名。
    """
    # 确保数组是一维的
    assert field.ndim == 1, "数组必须是一维的"

    # 每5个元素为一组，格式化为字符串并写入一行
    for i in range(0, len(field), 5):
        # 提取当前组的元素，并将其转换为字符串列表
        line = ' '.join(map(str, field[i:i+5]))
        # 写入一行数据
        file_handle.write(line + '\n')
    
    # 在文件最后写入一个换行符
    file_handle.write('\n')
        



def formatnp(data, file_handle, amounts_per_line=3):
    """
    Write formatted numpy array data to a file, with each line containing a specified number of elements.

    Arguments:
        - data: a list or numpy array of data to write.
        - file_handle: an open file handle for writing.
        - amounts_per_line: the number of data elements per line (default is 3).
    """
    for i in range(len(data)):
        if np.issubdtype(data[i], np.integer):
            file_handle.write(" {:d}".format(data[i].item()))
        else:
            file_handle.write(" {:e}".format(data[i].item()))
        if (i + 1) % amounts_per_line == 0:
            file_handle.write("\n")
    
    # Ensure the file ends with a newline character
    if len(data) % amounts_per_line != 0:
        file_handle.write("\n")


def has_more_than_three_duplicates(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return np.any(counts > 3)


def count_cells_num_node(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return counts


def write_cell_index(Cells, Cells_index, writer):
    """
    将单元格节点索引写入文件。
    
    参数:
        Cells: flattened 节点索引数组，形状为 [总节点数]
        Cells_index: 每个节点属于哪个单元的索引，形状为 [总节点数]
        writer: 文件写入器
    
    说明:
        Cells 和 Cells_index 都是被 flattened 的一维数组。
        Cells_index[i] 表示 Cells[i] 属于第几个单元。
        例如：
            Cells = [1, 2, 3, 4, 5, 6, 7, 8, ...]
            Cells_index = [0, 0, 0, 0, 1, 1, 1, 1, ...]  （每个单元4个节点）
        应输出为：
            1 2 3 4
            5 6 7 8
    """
    if len(Cells) == 0:
        return
    
    # 确保 Cells 和 Cells_index 都是 NumPy arrays（可能是 PyTorch tensors）
    if hasattr(Cells, 'cpu'):
        Cells = Cells.cpu().numpy()
    elif not isinstance(Cells, np.ndarray):
        Cells = np.asarray(Cells)
    
    if hasattr(Cells_index, 'cpu'):
        Cells_index = Cells_index.cpu().numpy()
    elif not isinstance(Cells_index, np.ndarray):
        Cells_index = np.asarray(Cells_index)
    
    # 获取唯一的单元索引，用于确定单元数和每个单元的节点数
    unique_cell_indices = np.unique(Cells_index)
    num_cells = len(unique_cell_indices)
    
    # 检查每个单元有多少个节点
    cell_node_counts = {}
    for cell_idx in unique_cell_indices:
        mask = Cells_index == cell_idx
        # 确保 mask 是 numpy 数组
        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)
        node_count = np.sum(mask)
        cell_node_counts[cell_idx] = node_count
    
    # 检查是否所有单元的节点数相同
    node_counts_set = set(cell_node_counts.values())
    if len(node_counts_set) == 1:
        nodes_per_cell = list(node_counts_set)[0]
        print(f"每个单元有 {nodes_per_cell} 个节点")
    else:
        print(f"警告：单元节点数不一致: {node_counts_set}")
        nodes_per_cell = max(node_counts_set)
    
    # 逐个单元输出
    for cell_idx in unique_cell_indices:
        mask = Cells_index == cell_idx
        cell_nodes = Cells[mask]
        
        # 确保 cell_nodes 是 NumPy array（可能是 PyTorch tensor）
        if hasattr(cell_nodes, 'cpu'):
            cell_nodes = cell_nodes.cpu().numpy()
        elif not isinstance(cell_nodes, np.ndarray):
            cell_nodes = np.asarray(cell_nodes)
        
        # 检查节点数
        if len(cell_nodes) < 3:
            print(f"警告：单元 {cell_idx} 节点数过少 ({len(cell_nodes)} < 3)，跳过")
            continue
        
        # 检查是否有无效的节点索引（0或负数）
        if np.any(cell_nodes <= 0):
            print(f"警告：单元 {cell_idx} 包含无效节点索引 (<=0): {cell_nodes[cell_nodes <= 0]}，跳过")
            continue
        
        # 如果节点数不足（例如三角形被标记为四边形），复制最后一个节点
        if len(cell_nodes) < nodes_per_cell and nodes_per_cell == 4:
            if len(cell_nodes) == 3:
                cell_nodes = np.append(cell_nodes, cell_nodes[-1])
        
        # 写入单元格定义
        cell_str = " ".join([str(int(node)) for node in cell_nodes])
        writer.write(f" {cell_str}\n")
            
def write_face_index(faces, writer):
    for index in range(faces.shape[0]):
        formatnp(faces[index], writer, amounts_per_line=2)

def write_uvp_tecplotzone(
    filename="flowcfdgcn.dat",
    datasets=None,
    time_step_length=100,

):
    interior_zone = datasets[0]
    mu = interior_zone["mu"]
    rho = interior_zone["rho"]
    dt  = interior_zone["dt"]

    with open(filename, "w") as f:
        f.write('TITLE = "FDTO solution"\n')
        f.write('VARIABLES = "X"\n"Y"\n"U"\n"V"\n"P"\n')
        f.write('DATASETAUXDATA Common.Incompressible="TRUE"\n')
        f.write('DATASETAUXDATA Common.VectorVarsAreVelocity="TRUE"\n')
        f.write(f'DATASETAUXDATA Common.Viscosity="{mu}"\n')
        f.write(f'DATASETAUXDATA Common.Density="{rho}"\n')
        f.write(f'DATASETAUXDATA Common.UVar="3"\n')
        f.write(f'DATASETAUXDATA Common.VVar="4"\n')
        f.write(f'DATASETAUXDATA Common.PressureVar="5"\n')

        for i in range(time_step_length):
            for zone in datasets:
                zonename = zone["zonename"]
                if zonename == "Fluid":
                    f.write('ZONE T="{0}"\n'.format(zonename))

                    X = zone["mesh_pos"][i, :, 0]
                    Y = zone["mesh_pos"][i, :, 1]
                    U = zone["velocity"][i, :, 0]
                    V = zone["velocity"][i, :, 1]
                    P = zone["pressure"][i, :, 0]

                    field = np.concatenate((X, Y, U, V, P), axis=0)
                    Cells = zone["cells"] + 1
                    Cells_index = zone["cells_index"]
                    face_node = zone["face_node"]
          
                    f.write(" STRANDID=1, SOLUTIONTIME={0}\n".format(dt * i))
                    counts = count_cells_num_node(Cells_index)
                    # 计算实际的唯一元素数量，而不是使用max()+1
                    unique_elements = len(np.unique(Cells_index))
                    write_face = False
                    if counts.max() <= 3:
                        f.write(
                            f" Nodes={X.size}, Elements={unique_elements}, "
                            "ZONETYPE=FETRIANGLE\n"
                        )
                        write_face = False
                    elif 3 < counts.max() <= 4:
                        f.write(
                            f" Nodes={X.size}, Elements={unique_elements}, "
                            "ZONETYPE=FEQuadrilateral\n"
                        )
                        write_face = False
                    elif counts.max() > 4:
                        f.write(
                            f" Nodes={X.size}, Faces={face_node.shape[1]},Elements={unique_elements}, "
                            "ZONETYPE=FEPolygon\n"
                        )
                        f.write(
                            f"NumConnectedBoundaryFaces=0, TotalNumBoundaryConnections=0\n"
                        )
                        write_face = True

                    f.write(" DATAPACKING=BLOCK\n")
                    data_packing_type = zone["data_packing_type"]
                    if data_packing_type == "cell":
                        f.write(" VARLOCATION=([3,4,5]=CELLCENTERED)\n")
                    elif data_packing_type == "node":
                        f.write(" VARLOCATION=([3,4,5]=NODAL)\n")
                    f.write(" DT=(SINGLE SINGLE SINGLE SINGLE SINGLE)\n")
                    try:
                        print(f"start writing interior field data, size in {field.size},shape in {field.shape}")
                        write_array_to_file(field, f)
                    except Exception as e:
                        print(f"Error formatting data: {e}")
                        
                    print("Start writing interior cell")
                    if not write_face:
                        write_cell_index(Cells, Cells_index, f)
     
                elif (
                    zonename == "OBSTACLE_BOUNDARY" or zonename.find("BOUNDARY") != -1
                ):
                    f.write('ZONE T="{0}"\n'.format(zonename))
                    X = zone["mesh_pos"][i, :, 0].astype(np.float32)
                    Y = zone["mesh_pos"][i, :, 1].astype(np.float32)
                    U = zone["velocity"][i, :, 0]
                    V = zone["velocity"][i, :, 1]
                    P = zone["pressure"][i, :, 0]
                    field = np.concatenate((X, Y, U, V, P), axis=0)
                    faces = zone["face_node"]+ 1
                    f.write(" STRANDID=3, SOLUTIONTIME={0}\n".format(dt * i))
                    f.write(
                        f" Nodes={X.size}, Elements={faces.shape[0]}, "
                        "ZONETYPE=FELineSeg\n"
                    )
                    f.write(" DATAPACKING=BLOCK\n")
                    f.write('AUXDATA Common.BoundaryCondition="Wall"\n')
                    f.write('AUXDATA Common.IsBoundaryZone="TRUE"\n')
                    data_packing_type = zone["data_packing_type"]
                    if data_packing_type == "cell":
                        f.write(" VARLOCATION=([3,4,5]=CELLCENTERED)\n")
                    elif data_packing_type == "node":
                        f.write(" VARLOCATION=([3,4,5]=NODAL)\n")
                    f.write(" DT=(SINGLE SINGLE SINGLE SINGLE SINGLE)\n")
                    
                    print("start writing boundary field data")
                    write_array_to_file(field, f)
                    
                    print("start writing boundary face")
                    write_face_index(faces, f)
                    
    print("saved tecplot file at " + filename)

def write_u_tecplotzone(
    filename="flowcfdgcn.dat",
    datasets=None,
    time_step_length=100,

):
    interior_zone = datasets[0]
    mu = interior_zone["mu"]
    rho = interior_zone["rho"]
    dt  = interior_zone["dt"]

    with open(filename, "w") as f:
        f.write('TITLE = "FOGN solution"\n')
        f.write('VARIABLES = "X"\n"Y"\n"U"\n')
        f.write('DATASETAUXDATA Common.Incompressible="TRUE"\n')
        f.write('DATASETAUXDATA Common.VectorVarsAreVelocity="TRUE"\n')
        f.write(f'DATASETAUXDATA Common.Viscosity="{mu}"\n')
        f.write(f'DATASETAUXDATA Common.Density="{rho}"\n')
        f.write(f'DATASETAUXDATA Common.UVar="3"\n')

        for i in range(time_step_length):
            for zone in datasets:
                zonename = zone["zonename"]
                if zonename == "Fluid":
                    f.write('ZONE T="{0}"\n'.format(zonename))

                    X = zone["mesh_pos"][i, :, 0]
                    Y = zone["mesh_pos"][i, :, 1]
                    
                    # Handle both 2D and 3D velocity arrays
                    velocity = zone["velocity"]
                    if velocity.ndim == 3:
                        U = velocity[i, :, 0]
                    else:  # 2D array [time, nodes]
                        U = velocity[i, :]
                    

                    field = np.concatenate((X, Y, U), axis=0)
                    Cells = zone["cells"] + 1
                    Cells_index = zone["cells_index"]
                    face_node = zone["face_node"]
          
                    f.write(" STRANDID=1, SOLUTIONTIME={0}\n".format(dt * i))
                    counts = count_cells_num_node(Cells_index)
                    # 计算实际的唯一元素数量，而不是使用max()+1
                    unique_elements = len(np.unique(Cells_index))
                    write_face = False
                    if counts.max() <= 3:
                        f.write(
                            f" Nodes={X.size}, Elements={unique_elements}, "
                            "ZONETYPE=FETRIANGLE\n"
                        )
                        write_face = False
                    elif 3 < counts.max() <= 4:
                        f.write(
                            f" Nodes={X.size}, Elements={unique_elements}, "
                            "ZONETYPE=FEQuadrilateral\n"
                        )
                        write_face = False
                    elif counts.max() > 4:
                        f.write(
                            f" Nodes={X.size}, Faces={face_node.shape[1]},Elements={unique_elements}, "
                            "ZONETYPE=FEPolygon\n"
                        )
                        f.write(
                            f"NumConnectedBoundaryFaces=0, TotalNumBoundaryConnections=0\n"
                        )
                        write_face = True

                    f.write(" DATAPACKING=BLOCK\n")
                    data_packing_type = zone["data_packing_type"]
                    if data_packing_type == "cell":
                        f.write(" VARLOCATION=([3,4]=CELLCENTERED)\n")
                    elif data_packing_type == "node":
                        f.write(" VARLOCATION=([3,4]=NODAL)\n")
                    f.write(" DT=(SINGLE SINGLE SINGLE)\n")
                    try:
                        print(f"start writing interior field data, size in {field.size},shape in {field.shape}")
                        write_array_to_file(field, f)
                    except Exception as e:
                        print(f"Error formatting data: {e}")
                        
                    print("Start writing interior cell")
                    if not write_face:
                        write_cell_index(Cells, Cells_index, f)
     
                elif (
                    zonename == "OBSTACLE_BOUNDARY" or zonename.find("BOUNDARY") != -1
                ):
                    f.write('ZONE T="{0}"\n'.format(zonename))
                    X = zone["mesh_pos"][i, :, 0].astype(np.float32)
                    Y = zone["mesh_pos"][i, :, 1].astype(np.float32)
                    U = zone["velocity"][i, :, 0]

                    field = np.concatenate((X, Y, U), axis=0)
                    faces = zone["face_node"]+ 1
                    f.write(" STRANDID=3, SOLUTIONTIME={0}\n".format(dt * i))
                    f.write(
                        f" Nodes={X.size}, Elements={faces.shape[0]}, "
                        "ZONETYPE=FELineSeg\n"
                    )
                    f.write(" DATAPACKING=BLOCK\n")
                    f.write('AUXDATA Common.BoundaryCondition="Wall"\n')
                    f.write('AUXDATA Common.IsBoundaryZone="TRUE"\n')
                    data_packing_type = zone["data_packing_type"]
                    if data_packing_type == "cell":
                        f.write(" VARLOCATION=([3,4]=CELLCENTERED)\n")
                    elif data_packing_type == "node":
                        f.write(" VARLOCATION=([3,4]=NODAL)\n")
                    f.write(" DT=(SINGLE SINGLE SINGLE)\n")

                    print("start writing boundary field data")
                    write_array_to_file(field, f)
                    
                    print("start writing boundary face")
                    write_face_index(faces, f)
                    
    print("saved tecplot file at " + filename)