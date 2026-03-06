
import numpy as np
import torch
from torch_scatter import scatter
from utils.utilities import NodeType
from torch_geometric import utils as pyg_utils

def find_pos(mesh_point, mesh_pos_sp1):
    for k in range(mesh_pos_sp1.shape[0]):
        if (mesh_pos_sp1[k] == mesh_point).all():
            print("found{}".format(k))
            return k
    return False


def convert_to_tensors(input_dict):
    # 遍历字典中的所有键
    for key in input_dict.keys():
        # 检查值的类型
        value = input_dict[key]
        if isinstance(value, np.ndarray):
            # 如果值是一个Numpy数组，使用torch.from_numpy进行转换
            input_dict[key] = torch.from_numpy(value)
        elif not isinstance(value, torch.Tensor):
            # 如果值不是一个PyTorch张量，使用torch.tensor进行转换
            input_dict[key] = torch.tensor(value)
        # 如果值已经是一个PyTorch张量，不进行任何操作

    # 返回已更新的字典
    return input_dict


def polygon_area(vertices):
    """
    使用shoelace formula（鞋带公式）来计算多边形的面积。
    :param vertices: 多边形的顶点坐标，一个二维numpy数组。
    :return: 多边形的面积。
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def find_max_distance(points):
    # 获取点的数量
    n_points = points.size(0)

    # 初始化最大距离为0
    max_distance = 0

    # 遍历每一对点
    for i in range(n_points):
        for j in range(i + 1, n_points):
            # 计算两点之间的欧几里得距离
            distance = torch.norm(points[i] - points[j])

            # 更新最大距离
            max_distance = max(max_distance, distance)

    # 返回最大距离
    return max_distance


def compose_support_face_node_x(cells_type, cells_node):
    """
    Composes the unique connections between nodes that form the faces of each cell.

    Parameters:
    - cells_type (int): The number of nodes per cell (e.g., 3 for triangles, 4 for quadrilaterals).
    - cells_node (torch.Tensor): Tensor containing the indices of nodes for each cell, flattened.

    Returns:
    - torch.Tensor: A tensor of shape [2, num_faces], where each column represents a unique face defined by two node indices.
    """
    
    face_node_x=[]
    origin_cells_node = cells_node.clone()
    for _ in range(cells_type-1):
        cells_node = torch.roll(cells_node.reshape(-1,cells_type), shifts=1, dims=1).reshape(-1)
        face_node_x.append(torch.stack((origin_cells_node, cells_node), dim=0))

    return torch.unique(torch.cat(face_node_x, dim=1).sort(dim=0)[0],dim=1)

def compose_support_edge_to_node(cells_type, cells_face, cells_node, offset=None):
    """
    Constructs the mapping between faces and nodes, indicating which nodes belong to each face.

    Parameters:
    - cells_type (int): The number of nodes per cell.
    - cells_face (torch.Tensor): Tensor containing the indices of faces for each cell.
    - cells_node (torch.Tensor): Tensor containing the indices of nodes for each face.
    - offset (int, optional): An optional offset to be added to the face indices.

    Returns:
    - torch.Tensor: A tensor of shape [2, num_edges], representing the unique connections between faces and nodes.
    """
    if offset is not None:
        cells_face += offset
        
    support_edge_to_node=[]
    for _ in range(cells_type):
        support_edge_to_node.append(torch.stack((cells_face, cells_node), dim=0))
        cells_node = torch.roll(cells_node.reshape(-1,cells_type), shifts=1, dims=1).reshape(-1)
    return torch.unique(torch.cat(support_edge_to_node, dim=1).sort(dim=0)[0],dim=1)

def compose_support_cell_to_node(cells_type, cells_index, cells_node, offset=None):
    """
    Constructs the mapping between cells and nodes, indicating which nodes belong to each cell.

    Parameters:
    - cells_type (int): The number of nodes per cell.
    - cells_index (torch.Tensor): Tensor containing the indices of cells.
    - cells_node (torch.Tensor): Tensor containing the indices of nodes for each cell.
    - offset (int, optional): An optional offset to be added to the cell indices.

    Returns:
    - torch.Tensor: A tensor of shape [2, num_edges], representing the unique connections between cells and nodes.
    """
    if offset is not None:
        cells_index += offset
        
    support_cell_to_node=[]
    for _ in range(cells_type):
        support_cell_to_node.append(torch.stack((cells_index, cells_node), dim=0))
        cells_node = torch.roll(cells_node.reshape(-1,cells_type), shifts=1, dims=1).reshape(-1)
    return torch.unique(torch.cat(support_cell_to_node, dim=1).sort(dim=0)[0],dim=1)

def seperate_domain(cells_node, cells_face, cells_index):
    """
    Separates the domain into different regions based on cell types (e.g., triangular, quadrilateral and polygons).

    Parameters:
    - cells_node (torch.Tensor): Tensor containing the node indices for each cell.
    - cells_face (torch.Tensor): Tensor containing the face indices for each cell.
    - cells_index (torch.Tensor): Tensor containing the cell indices.

    Returns:
    - list: A list of tuples, each containing:
        - ct (int): The cell type (number of nodes per cell).
        - cells_node_sub (torch.Tensor): Subset of cells_node for the cell type.
        - cells_face_sub (torch.Tensor): Subset of cells_face for the cell type.
        - cells_index_sub (torch.Tensor): Subset of cells_index for the cell type.
    """
    cells_type_ex = scatter(src=torch.ones_like(cells_index), 
        index=cells_index, 
        dim=0, 
    )
    
    cells_type = torch.unique(cells_type_ex, dim=0)
    
    domain_list = []
    for ct in cells_type:
        mask = (cells_type_ex==ct)[cells_index]
        domain_list.append((ct, cells_node[mask], cells_face[mask], cells_index[mask]))
        
    return domain_list

def build_k_hop_edge_index(edge_index, k):
    """
    用PyG的to_torch_coo_tensor和稀疏矩阵运算计算k跳邻居的连接关系。

    Parameters:
    - mesh_pos: [N, 2] 每个节点的坐标。
    - edge_index: [2, E] 原始的边索引，两个节点之间的连通关系。
    - num_nodes: 节点总数 N。
    - k: 跳数，表示距离多少跳的邻居。

    Returns:
    - new_edge_index: 新的边索引数组, 包含距离当前节点k跳以外的邻居的连通关系。
    """
    # 将edge_index转换为稀疏矩阵 (COO 格式)
    sparse_adj = pyg_utils.to_torch_coo_tensor(edge_index)

    # 初始化邻接矩阵为一跳邻居
    adj_k = sparse_adj

    # 进行k-1次邻接矩阵自乘，得到k跳邻居
    for _ in range(k - 1):
        adj_k = torch.sparse.mm(adj_k, sparse_adj)

    # 从稀疏矩阵中提取新的edge_index (两跳或k跳邻居)
    new_edge_index = adj_k.coalesce().indices()

    return new_edge_index


def extract_mesh_state(
    dataset,
    path=None,
):
    """
    face_center_pos, centroid, face_type, neighbour_cell, face_node_x
    """
    dataset = convert_to_tensors(dataset)

    """>>> prepare for converting >>>"""

    cells_node = dataset["cells_node_unique"]
    cells_index = dataset["cells_index"]
    cells_face = dataset["cells_face_unique"]
    """<<< prepare for converting <<<"""


    ''' >>> compute face_node_x <<< '''
    domain_list = seperate_domain(
        cells_node=cells_node, 
        cells_face=cells_face, 
        cells_index=cells_index
    )
    
    face_node_x=[]
    for domain in domain_list:
        
        _ct, _cells_node, _cells_face, _cells_index = domain
        
        face_node_x.append(
            compose_support_face_node_x(cells_type=_ct, cells_node=_cells_node)
        )
    face_node_x = torch.cat(face_node_x, dim=1)
    dataset["face_node_x"] = face_node_x
    ''' >>> compute face_node_x <<< '''
    
    # print(f"{path['case_name']}mesh has been extracted")

    return dataset
