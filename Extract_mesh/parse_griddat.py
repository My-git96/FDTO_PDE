import sys
import os
file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(file_dir)
import numpy as np
import multiprocessing
import torch
import re
from Extract_mesh.to_h5 import (
    extract_mesh_state,
    NodeType,
)
import os
import h5py
import sys
import torch.nn.functional as F
from utils.utilities import *
from torch_scatter import scatter_mean
# 将输出缓冲区设置为0
sys.stdout.flush()

def np2torch(np_array):
    return torch.from_numpy(np_array)



class TopologyManager:
    """
    多块结构网格管理器，支持 ghost layer。
      A. 全局单元/节点编号只统计真实单元/节点
      B. 连接面 ghost layer 用 (R,t) 从相邻块拷贝
      C. 物理边界使用edge padding填充ghost层
    """
    def __init__(self, top_file, ghost_layer=None):
        self.top_file = top_file
        self.ghost_layer = ghost_layer
        self.block_topology = self._create_topology(top_file)


    @staticmethod
    def _has_reverse_coordinates(coords):
        arr = np.array(coords)
        return np.any(arr[..., 0] > arr[..., 1])

    @staticmethod
    def _range_to_idx(start, end, max_dim):
        """TOP 文件坐标→0-based 半开区间索引。[start,end)；当 start==end 表示单层。"""
        if start == end:
            start_idx = max(0, start - 1)
            end_idx = start_idx + 1
        elif start > end:
            start_idx = max(0, start - 1)   
            end_idx = max(0, end - 1)       
        else:
            start_idx = max(0, start - 1)
            end_idx = end
        return start_idx, end_idx    

    def _create_topology(self, top_path):
        original_block_data = []
        padded_block_data = []
        mapped_block_data = []
        block_topology = []
        global_cell_offset = 0
        global_node_offset = 0

        with open(top_path, 'r') as file:
            solveid = int(file.readline().strip())
            nblocks = int(file.readline().strip())
            self.nblocks = nblocks    

            for nb in range(nblocks):               
                ni, nj = map(int, file.readline().strip().split())
                block_id = nb
                # ------- 真实单元/节点编号（不含 ghost） -------
                nci, ncj = ni-1, nj-1
                block_cell_offset = global_cell_offset
                block_node_offset = global_node_offset
                block_cells_index = (np.arange(nci*ncj, dtype=np.int64)
                                     .reshape(nci, ncj) + block_cell_offset)
                global_cell_offset += nci*ncj
                block_nodes_index = (np.arange(ni*nj, dtype=np.int64)
                                     .reshape(ni, nj) + block_node_offset)
                global_node_offset += ni*nj

                block_nodes_type = np.full((ni, nj), NodeType.NORMAL, dtype=np.int32)

                block_name = file.readline().strip()
                bc_region_num = int(file.readline().strip())

                connection_faces = []
                bc_faces = []
                
                for bc_region in range(bc_region_num):
                    xs, xe, ys, ye, bc_type = map(int, file.readline().strip().split())
                    xs_idx, xe_idx = self._range_to_idx(xs, xe, ni)
                    ys_idx, ye_idx = self._range_to_idx(ys, ye, nj)
                    bc_coords = np.array([[xs_idx, xe_idx],[ys_idx, ye_idx]], dtype=np.int64)                    

                    if bc_type > 0:
                        bc_faces.append({'bc_type':bc_type,'bc_coords':bc_coords})
                        if bc_type == 2:  # 壁面
                            block_nodes_type[xs_idx:xe_idx, ys_idx:ye_idx] = NodeType.WALL

                        elif bc_type == 3:  # 对称边界
                            block_nodes_type[xs_idx:xe_idx, ys_idx:ye_idx] = NodeType.SYMMETRY
                            
                        elif bc_type == 4:  # 远场边界
                            block_nodes_type[xs_idx:xe_idx, ys_idx:ye_idx] = NodeType.FARFIELD

                        elif bc_type == 5:  # 入口边界
                            block_nodes_type[xs_idx:xe_idx, ys_idx:ye_idx] = NodeType.INFLOW

                        elif bc_type == 6:  # 出口边界
                            block_nodes_type[xs_idx:xe_idx, ys_idx:ye_idx] = NodeType.OUTFLOW
                        else:
                            print(f"    未知边界类型: {bc_type}")
                            pass                        
                    elif bc_type == -1:
                        region_mask = block_nodes_type[xs_idx:xe_idx, ys_idx:ye_idx]
                        # 检查是否存在边界类型节点
                        if (region_mask != NodeType.NORMAL).any():
                            # 创建布尔掩码，标识哪些节点是BC_NORMAL类型
                            normal_node_mask = (region_mask == NodeType.NORMAL)
                            # 只对BC_NORMAL类型的节点设置为连接面类型
                            region_mask[normal_node_mask] = NodeType.CUT1TO1
                            # 将修改后的区域掩码赋值回原数组
                            block_nodes_type[xs_idx:xe_idx, ys_idx:ye_idx] = region_mask
                        else:
                            # 如果区域内都是BC_NORMAL类型，则全部设置为连接面类型
                            block_nodes_type[xs_idx:xe_idx, ys_idx:ye_idx] = NodeType.CUT1TO1

                        txs, txe, tys, tye, tbid = map(int, file.readline().strip().split())           
                        txs_i, txe_i = self._range_to_idx(txs, txe, ni)
                        tys_i, tye_i = self._range_to_idx(tys, tye, nj)

                        target_coords = np.array([[txs_i, txe_i],[tys_i, tye_i]], dtype=np.int64)               
                        connection_faces.append({                   
                            'source_coordinates': bc_coords,
                            'target_coordinates': target_coords,
                            'target_block_id': int(tbid-1),
                            'source_block_id': block_id,
                        })
                    else:
                        print(f"    未知边界类型: {bc_type}")
                        pass
                
                connection_faces_dict = {}
                for face in connection_faces:
                    target_block_id = face['target_block_id']
                    if target_block_id in connection_faces_dict:
                        if isinstance(connection_faces_dict[target_block_id], list):
                            connection_faces_dict[target_block_id].append(face)
                        else:
                            connection_faces_dict[target_block_id] = [connection_faces_dict[target_block_id], face]
                    else:
                        connection_faces_dict[target_block_id] = face
                
                original_block_data.append({
                    'block_id': block_id,
                    'cells_shape': (nci, ncj),
                    'nodes_shape': (ni, nj),
                    'block_cells_index': block_cells_index,
                    'block_nodes_index': block_nodes_index,
                    'block_nodes_type': block_nodes_type,
                    'block_cell_offset': block_cell_offset,
                    'block_node_offset': block_node_offset,
                    'connection_faces': connection_faces,
                    'target_face_map': connection_faces_dict,
                    'bc_faces': bc_faces, 
                })

                # padding ghost layer
                g = self.ghost_layer
                padw = ((g, g), (g, g))
                cells_pad = np.pad(block_cells_index, padw, mode='edge')
                nodes_pad = np.pad(block_nodes_index, padw, mode='edge')
                node_type_pad = np.pad(block_nodes_type, padw, mode='edge')

                padded_block_data.append({
                    'block_id': block_id,
                    'original_cells_shape': (nci, ncj),
                    'original_nodes_shape': (ni, nj),
                    'cells_shape': (nci+2*g, ncj+2*g),
                    'nodes_shape': (ni+2*g, nj+2*g),
                    'block_cells_index': cells_pad,
                    'block_nodes_index': nodes_pad,
                    "block_nodes_type_pad" :node_type_pad,
                    'block_nodes_type': block_nodes_type,
                    'block_cell_offset': block_cell_offset,
                    'block_node_offset': block_node_offset,
                    'connection_faces': connection_faces,
                    'target_face_map': connection_faces_dict,
                    'bc_faces': bc_faces, 
                })

        mapped_block_data = self._block_ghost_mapping(padded_block_data, original_block_data)

        for original, mapped in zip(original_block_data, mapped_block_data):

            block_data_dict = {}

            block_data_dict["block_id"] = original["block_id"]
            block_data_dict["cells_shape"] = original["cells_shape"]

            original_node_type = original["block_nodes_type"].copy()

            original_node_type[original_node_type==NodeType.CUT1TO1] = NodeType.NORMAL

         
            block_data_dict["nodes_shape"] = original["nodes_shape"]
            block_data_dict["block_cells_index"] = original["block_cells_index"]
            block_data_dict["block_nodes_index"] = original["block_nodes_index"]
            block_data_dict["block_nodes_type"] = original_node_type

            block_data_dict["extended_cells_index"] = mapped["block_cells_index"]
            block_data_dict["extended_nodes_index"] = mapped["block_nodes_index"]
            block_data_dict["extended_cells_shape"] = mapped["cells_shape"]
            block_data_dict["extended_nodes_shape"] = mapped["nodes_shape"]

            block_nodes_type_pad = mapped["block_nodes_type_pad"]

            block_nodes_type_pad[1:-1,1:-1] = NodeType.NORMAL
            #block_nodes_type_pad[block_nodes_type_pad==NodeType.CUT1TO1] = NodeType.NORMAL

            block_data_dict["extended_node_type"] = block_nodes_type_pad
            
            # 保存拓扑信息供可视化使用
            block_data_dict["connection_faces"] = original["connection_faces"]
            block_data_dict["bc_faces"] = original["bc_faces"]
            block_data_dict["target_face_map"] = original["target_face_map"]

            block_topology.append(block_data_dict)

   

        return block_topology

    def _get_face_start_end(self,coords,direction):
        i_range = coords[0]
        j_range = coords[1]
        normal_axis = direction[0]

        start_point = [0, 0]
        end_point = [0, 0]

        if normal_axis == 0: # 法线是 i 轴
            start_point[0] = i_range[0]
            end_point[0]   = i_range[0]
            start_point[1] = j_range[0]
            end_point[1]   = j_range[1]
        else: # 法线是 j 轴
            start_point[0] = i_range[0]
            end_point[0]   = i_range[1]
            start_point[1] = j_range[0]
            end_point[1]   = j_range[0]
            
        return np.array(start_point), np.array(end_point)

    def _get_connection_face_mapmat(self,face,src_blk_node_shape,tgt_blk_node_shape):
        mide = np.eye(2, dtype=np.int32)
        src_side,src_direction = self._get_face_side(face['source_coordinates'],src_blk_node_shape)
        tgt_side,tgt_direction = self._get_face_side(face['target_coordinates'],tgt_blk_node_shape)
        src_start,src_end = self._get_face_start_end(face['source_coordinates'],src_direction)
        tgt_start,tgt_end = self._get_face_start_end(face['target_coordinates'],tgt_direction)

        # 建立轴映射关系
        axis_map = np.zeros(2, dtype=np.int32)
        axis_map[tgt_direction[0]] = src_direction[0] # 法线 -> 法线
        axis_map[tgt_direction[1]] = src_direction[1] # 切线 -> 切线

        tgt_tang_axis = tgt_direction[1]

        #计算目标符号：包括法向和切向
        tgt_sign = np.zeros(2,dtype=np.int32)
        tang_diff = tgt_end[tgt_tang_axis] - tgt_start[tgt_tang_axis]
        tgt_sign[tgt_tang_axis] = np.sign(tang_diff)

        side_map = {'lower': -1, 'upper': 1}
        src_side_val = side_map[src_side]
        tgt_side_val = side_map[tgt_side]
        normal_sign = src_side_val*tgt_side_val
        tgt_sign[tgt_direction[0]] = normal_sign
    
        #计算旋转矩阵R
        rotation_matrix = tgt_sign[:,np.newaxis]*mide[:,axis_map]
        #计算偏移量t 目标起点=R*源起点+t t=目标起点-R*源起点
        offset = tgt_start - np.sum(rotation_matrix*src_start,axis=1)
        mapmat = np.column_stack([rotation_matrix,offset])
        return mapmat


    def _get_connection_face_coords(self, face_coords, face_id, coord_type='nodes'):
        i_range = face_coords[0]
        j_range = face_coords[1]

        # 取起止值和方向符号
        i_start, i_end = i_range[0], i_range[1]
        j_start, j_end = j_range[0], j_range[1]

        # 确定切向范围和点数
        if face_id in ['i_min', 'i_max']: # 切向是 j
            tang_start, tang_end = j_start, j_end
            normal_val_node = i_start
        else: # 切向是 i
            tang_start, tang_end = i_start, i_end
            normal_val_node = j_start
        
        step = 1 if tang_start < tang_end else -1
        tang_indices_node = np.arange(tang_start, tang_end, step, dtype=np.int32)

        if coord_type == 'nodes':
            i_indices = np.full_like(tang_indices_node, normal_val_node) if face_id in ['i_min', 'i_max'] else tang_indices_node
            j_indices = tang_indices_node if face_id in ['i_min', 'i_max'] else np.full_like(tang_indices_node, normal_val_node)
            return np.stack([i_indices, j_indices], axis=-1)
        elif coord_type == 'cells':
            tang_indices_cell = tang_indices_node[:-1] if tang_start < tang_end else tang_indices_node[1:]
            # 单元的法向索引
            if 'max' in face_id:
                normal_val_cell = normal_val_node - 1
            else:
                normal_val_cell = normal_val_node
        
        normal_indices_cell = np.full_like(tang_indices_cell, normal_val_cell)

        if face_id in ['i_min', 'i_max']:
            i_indices, j_indices = normal_indices_cell, tang_indices_cell
        else:
            i_indices, j_indices = tang_indices_cell, normal_indices_cell
        return np.stack([i_indices, j_indices], axis=-1)

    def _get_ghost_indices(self, face_coords, face_id, coord_type='nodes'):
        i_range, j_range = face_coords
        # 确定法向和切向的范围
        if face_id in ['i_min', 'i_max']: # 切向是 j
            tang_start, tang_end = j_range
            normal_boundary_coord = i_range[0]
        else: # 切向是 i
            tang_start, tang_end = i_range
            normal_boundary_coord = j_range[0]
        
        num_tang_pts = abs(tang_end - tang_start)
        if coord_type == 'nodes':
            num_tang_pts += 1
        # 生成切向索引
        if num_tang_pts > 0:
            tangent_indices = np.linspace(tang_start, tang_end, num=num_tang_pts, dtype=np.int32)
        else: # 如果面上只有一个点
            tangent_indices = np.array([tang_start], dtype=np.int32)

        g = self.ghost_layer
        # 生成法向（ghost）索引
        if 'min' in face_id:
            normal_indices = np.arange(-g, 0, dtype=np.int32)
        else: # 'max'
            normal_indices = np.arange(normal_boundary_coord + 1, normal_boundary_coord + 1 + g, dtype=np.int32)

        if face_id in ['i_min', 'i_max']: # 法向是i, 切向是j
            i_indices, j_indices = np.meshgrid(normal_indices, tangent_indices, indexing='ij')
        else: # 法向是j, 切向是i
            j_indices, i_indices = np.meshgrid(normal_indices, tangent_indices, indexing='ij')

        return (i_indices.ravel(), j_indices.ravel())

    def _get_face_side(self,face_coords, blk_node_shape):
        ni, nj = blk_node_shape
        direction = [None, None]
        side = None
        normal_axis = None
        tang_axis = None
        # 判断法线方向 (轴坐标恒定)
        if face_coords[0,0] + 1 == face_coords[0,1]:
            normal_axis = 0
            tang_axis = 1
            coord_val = face_coords[0,0]
            max_coord = ni - 1
        elif face_coords[1,0] + 1 == face_coords[1,1]:
            normal_axis = 1
            tang_axis = 0
            coord_val = face_coords[1,0]
            max_coord = nj - 1
        if coord_val == 0:
            side = 'lower'
        elif coord_val == max_coord:
            side = 'upper'
        else:
            side = 'lower' if face_coords[tang_axis, 0] < face_coords[tang_axis, 1] else 'upper'
        direction = [normal_axis, tang_axis]
        return side, direction

    def _map_node_indices_for_face(self, face, org_blk_dict, padded_blk_dict):
        g = self.ghost_layer
        src_blk_id = face['source_block_id']
        tgt_blk_id = face['target_block_id']

        src_blk_data = org_blk_dict[src_blk_id]
        src_blk_node_shape = src_blk_data['nodes_shape']
        tgt_blk_node_shape = org_blk_dict[tgt_blk_id]['nodes_shape']

        src_side, src_direction = self._get_face_side(face['source_coordinates'], src_blk_node_shape)
        tgt_side, tgt_direction = self._get_face_side(face['target_coordinates'], tgt_blk_node_shape)

        mapmat = self._get_connection_face_mapmat(face, src_blk_node_shape, tgt_blk_node_shape)
        rotation = mapmat[:, :2]
        offset = mapmat[:, 2]

        src_face_id = f"{ {0:'i',1:'j'}[src_direction[0]] }_{ {'lower':'min','upper':'max'}[src_side] }"
        src_phys_nodes = self._get_connection_face_coords(face['source_coordinates'], src_face_id, 'nodes')
        tgt_phys_nodes_mapped = np.round(np.einsum('...i,ji->...j', src_phys_nodes, rotation) + offset).astype(np.int32)

        src_layers = np.arange(g)
        normal_vec_src = np.zeros(2, dtype=np.int32)
        normal_axis_src = src_direction[0]
        normal_vec_src[normal_axis_src] = 1
        delta_src_inward_vec = src_layers[:, np.newaxis] * normal_vec_src
        if src_side == 'upper':
            normal_axis_tgt = tgt_direction[0]
            correction = -1 if tgt_side == 'lower' else 1
            tgt_phys_nodes_mapped[:, normal_axis_tgt] += correction
            delta_src_inward_vec *= -1

        tgt_offset_layers = np.arange(1, g + 1)
        normal_vec_tgt = np.zeros(2, dtype=np.int32)
        normal_vec_tgt[tgt_direction[0]] = 1
        if tgt_side == 'lower':
            delta_tgt_ghost_vec = -tgt_offset_layers[:, np.newaxis] * normal_vec_tgt
        else:
            delta_tgt_ghost_vec = tgt_offset_layers[:, np.newaxis] * normal_vec_tgt

        #更新节点索引
        inward_offset_vec = np.zeros(2, dtype=np.int32)
        if src_side == 'lower':
            inward_offset_vec[src_direction[0]] = 1
        else:
            inward_offset_vec[src_direction[0]] = -1

        src_final_nodes = src_phys_nodes[np.newaxis, :, :] + inward_offset_vec[np.newaxis, np.newaxis, :] + delta_src_inward_vec[:, np.newaxis, :]
        src_final_nodes = src_final_nodes.reshape(-1, 2)
        tgt_ghost_nodes = tgt_phys_nodes_mapped[np.newaxis, :, :] + delta_tgt_ghost_vec[:, np.newaxis, :]
        tgt_ghost_nodes += g
        tgt_ghost_nodes = tgt_ghost_nodes.reshape(-1, 2)
        src_nodes_shape = src_blk_data['block_nodes_index'].shape
        src_final_nodes_clipped = np.clip(src_final_nodes, [0, 0], [src_nodes_shape[0]-1, src_nodes_shape[1]-1])
        normal_axis_tgt = tgt_direction[0]
        if (src_side == 'upper' and tgt_side == 'lower') or (normal_axis_src != normal_axis_tgt and tgt_side == 'lower'):  # upper-low或法向相反：flip
            src_final_nodes_clipped = src_final_nodes_clipped.reshape(g, -1, 2)[::-1].reshape(-1, 2)
        padded_tgt_data = padded_blk_dict[tgt_blk_id]
        padded_tgt_data = padded_blk_dict[tgt_blk_id]
        tgt_nodes_shape = padded_tgt_data['block_nodes_index'].shape
        tgt_ghost_nodes_clipped = np.clip(tgt_ghost_nodes, [0, 0], [tgt_nodes_shape[0]-1, tgt_nodes_shape[1]-1])
        src_nodes_values = src_blk_data['block_nodes_index'][(src_final_nodes_clipped[:, 0], src_final_nodes_clipped[:, 1])]
        padded_tgt_data['block_nodes_index'][tgt_ghost_nodes_clipped[:, 0], tgt_ghost_nodes_clipped[:, 1]] = src_nodes_values
        return
    
    def _map_cell_indices_for_face(self, face, org_blk_dict, padded_blk_dict):
        g = self.ghost_layer
        src_blk_id = face['source_block_id']
        tgt_blk_id = face['target_block_id']

        src_blk_data = org_blk_dict[src_blk_id]
        tgt_blk_data = org_blk_dict[tgt_blk_id]

        src_side, src_direction = self._get_face_side(face['source_coordinates'], src_blk_data['nodes_shape'])
        tgt_side, tgt_direction = self._get_face_side(face['target_coordinates'], tgt_blk_data['nodes_shape'])

        mapmat = self._get_connection_face_mapmat(face, src_blk_data['nodes_shape'], tgt_blk_data['nodes_shape'])
        rotation = mapmat[:, :2]
        offset = mapmat[:, 2]

        src_face_id = f"{ {0:'i',1:'j'}[src_direction[0]] }_{ {'lower':'min','upper':'max'}[src_side] }"
        src_phys_cells = self._get_connection_face_coords(face['source_coordinates'], src_face_id, 'cells')
        
        tgt_phys_cells_mapped = np.round(np.einsum('...i,ji->...j', src_phys_cells, rotation) + offset).astype(np.int32)

        # 1. 校正切向反向
        tang_axis_tgt = tgt_direction[1]
        if face['target_coordinates'][tang_axis_tgt, 0] > face['target_coordinates'][tang_axis_tgt, 1]:
            tgt_phys_cells_mapped[:, tang_axis_tgt] -= 1

        # 2. 校正法向不匹配
        normal_axis_tgt = tgt_direction[0]
        src_layers = np.arange(g)
        normal_vec_src = np.zeros(2, dtype=np.int32)
        normal_axis_src = src_direction[0]
        normal_vec_src[normal_axis_src] = 1
        delta_src_inward_vec = src_layers[:, np.newaxis] * normal_vec_src
        if src_side == 'upper':
            delta_src_inward_vec *= -1
            correction = -1 if tgt_side == 'lower' else 1
            tgt_phys_cells_mapped[:, normal_axis_tgt] += correction

        if tgt_side == 'lower':
            scaling_factors = -np.arange(1, g + 1)
        else:  # 'upper'
            scaling_factors = np.arange(g)

        normal_vec_tgt = np.zeros(2, dtype=np.int32)
        normal_vec_tgt[tgt_direction[0]] = 1
        delta_tgt_ghost_vec = scaling_factors[:, np.newaxis] * normal_vec_tgt

        src_final_cells = src_phys_cells[np.newaxis, :, :] + delta_src_inward_vec[:, np.newaxis, :]
        src_final_cells = src_final_cells.reshape(-1, 2)

        tgt_ghost_cells = (tgt_phys_cells_mapped + g)[np.newaxis, :, :] + delta_tgt_ghost_vec[:, np.newaxis, :]
        tgt_ghost_cells = tgt_ghost_cells.reshape(-1, 2)

        src_shape = src_blk_data['block_cells_index'].shape
        src_final_cells_clipped = np.clip(src_final_cells, [0, 0], [src_shape[0]-1, src_shape[1]-1])
        normal_axis_tgt = tgt_direction[0]
        if (src_side == 'upper' and tgt_side == 'lower') or (normal_axis_src != normal_axis_tgt and tgt_side == 'lower'):  # upper-low或法向相反：flip
            src_final_cells_clipped = src_final_cells_clipped.reshape(g, -1, 2)[::-1].reshape(-1, 2)
        padded_tgt_data = padded_blk_dict[tgt_blk_id]
        tgt_shape = padded_tgt_data['block_cells_index'].shape
        tgt_ghost_cells_clipped = np.clip(tgt_ghost_cells, [0, 0], [tgt_shape[0]-1, tgt_shape[1]-1])
        
        src_cells_values = src_blk_data['block_cells_index'][(src_final_cells_clipped[:, 0], src_final_cells_clipped[:, 1])]
        padded_tgt_data['block_cells_index'][tgt_ghost_cells_clipped[:, 0], tgt_ghost_cells_clipped[:, 1]] = src_cells_values
        return

    def _map_connection_face_to_ghost(self, face, org_blk_dict, padded_blk_dict):
        self._map_cell_indices_for_face(face, org_blk_dict, padded_blk_dict)
        self._map_node_indices_for_face(face, org_blk_dict, padded_blk_dict) 

    def _block_ghost_mapping(self, padded_blocks, original_blocks):
        if self.ghost_layer == 0:
            return padded_blocks
        
        org_blk_dict = {blk['block_id']: blk for blk in original_blocks}
        padded_blk_dict = {blk['block_id']: blk for blk in padded_blocks}

        for blk_data in original_blocks:
            for face in blk_data['connection_faces']:
                self._map_connection_face_to_ghost(face, org_blk_dict, padded_blk_dict)
                
        return list(padded_blk_dict.values())



class GeometryManager():

    def __init__(self,grid_file,topology ,ghost_layer):

        self.grid_file = grid_file

        self.ghost_layer = ghost_layer

        self.topology  = topology

        self._create_geometry(grid_file)

    def _create_geometry(self,grid_file):

        self.block_geometry = {}

        self.read_global_mesh_pos(grid_file)

        for x, (key, value) in zip(self.topology.block_topology, self.block_geometry.items()):


            row,col = value["nodes_shape"]

            current_block_pos = value["mesh_pos"].copy().reshape(row,col,2)

            extended_node_type = x["extended_node_type"].reshape(value["nodes_shape"][0]+2*self.ghost_layer, value["nodes_shape"][1]+2*self.ghost_layer)

            current_block_extened_node_index = x["extended_nodes_index"]

            exchange_mask = (extended_node_type==NodeType.CUT1TO1)[..., None]  

            extended_block_pos_extro = current_block_pos.copy()

            for _ in range(self.ghost_layer):

                extended_block_pos_extro = self.extend_pos(extended_block_pos_extro)
            

            extended_block_pos_interface  = self.mesh_pos.copy().reshape(-1,2)[current_block_extened_node_index].reshape(row+self.ghost_layer*2,col+self.ghost_layer*2,2)

            extended_pos = np.where(exchange_mask,extended_block_pos_interface,extended_block_pos_extro)

            original_block_metrics =  self.cal_metrics(current_block_pos)


            extended_block_metrics_topo_based = self.cal_metrics(extended_pos)

            #extended_block_metrics_interpo_based = self.cal_metrics(extended_block_pos_extro)
            

            self.block_geometry[key]["extended_block_pos"] = extended_block_pos_extro
            self.block_geometry[key]["extended_block_metrics"] = extended_block_metrics_topo_based

            self.block_geometry[key]["original_block_metrics"] = original_block_metrics

        
    def read_global_mesh_pos(self,filepath):
        """Reads an interleaved 2D PLOT3D ASCII mesh file."""
        line_n = 0

        mesh_pos = []
        with open(filepath, "r") as f:
            # 1. Read number of blocks
            try:
                nblocks = int(f.readline().strip())
                line_n += 1
            except (ValueError, IndexError):
                raise ValueError("Could not read number of blocks from first line.")

            
            # 2. Loop for each block to read its dimensions and then its data
            for block_num in range(nblocks):
                # Read dimensions for this block (ni, nj for 2D)
                try:
                    line = f.readline().strip().split()
                    line_n += 1
                    ni, nj,nk = map(int, line)
                except (ValueError, IndexError):
                    raise ValueError(f"Could not read dimensions for block {block_num + 1}.")

                npts = ni * nj * nk
                if npts == 0:
                    continue

                # Read X coordinates for this block
                x_flat = []
                while len(x_flat) < npts:
                    line = f.readline()
                    line_n += 1
                    if not line: raise IOError(f"Unexpected end of file while reading X for block {block_num + 1}.")
                    x_flat.extend(line.strip().split())
                # Read Y coordinates for this block
                y_flat = []
                while len(y_flat) < npts:
                    line = f.readline()
                    line_n += 1
                    if not line: raise IOError(f"Unexpected end of file while reading Y for block {block_num + 1}.")
                    y_flat.extend(line.strip().split())

                z_flat = []
                while len(z_flat) < npts:
                    line = f.readline()
                    line_n += 1
                    if not line: raise IOError(f"Unexpected end of file while reading Y for block {block_num + 1}.")
                    z_flat.extend(line.strip().split())

       
                x_coords = np.array(x_flat, dtype=float)
                y_coords = np.array(y_flat, dtype=float)
                
                if  "NACA" in filepath or "RAE"  in filepath:
                    block_coords = (np.column_stack((x_coords, y_coords)).reshape((ni,nj,2),order='F').reshape(-1,2))/10
                else:
                    block_coords = (np.column_stack((x_coords, y_coords)).reshape((ni,nj,2),order='F').reshape(-1,2))
                mesh_pos.append(block_coords)
        
                self.block_geometry[str(block_num)]={"mesh_pos":block_coords,"nodes_shape":(ni,nj)}
        self.mesh_pos = np.concatenate(mesh_pos)        

    def extend_pos(self,pos: np.ndarray) -> np.ndarray:
        """
        pos: (Ni, Nj, D)
        return: (Ni+2, Nj+2, D)
        规则：
        - 顶/底/左/右边（不含角）：一维线性外推 2*p0 - p1
        - 四个角：沿 x、y 两方向各做一次 1D 外推的组合
        """
        Ni, Nj, D = pos.shape
        if Ni < 2 or Nj < 2:
            raise ValueError("pos 至少需要 2x2 才能外推。")

        out = np.empty((Ni+2, Nj+2, D), dtype=pos.dtype)

        # 中心
        out[1:-1, 1:-1, :] = pos

        # 顶/底（用整行与 out 的 1:-1 对齐）
        out[0,      1:-1, :] = 2*pos[0,   :, :] - pos[1,   :, :]
        out[-1,     1:-1, :] = 2*pos[-1,  :, :] - pos[-2,  :, :]

        # 左/右（用整列与 out 的 1:-1 对齐）
        out[1:-1,   0,    :] = 2*pos[:,   0, :] - pos[:,   1, :]
        out[1:-1,  -1,    :] = 2*pos[:,  -1, :] - pos[:,  -2, :]


        out[0,   0,   :] = 2*pos[0,   0,   :] - pos[1,   1,   :]   # 左上
        out[0,  -1,   :] = 2*pos[0,  -1,   :] - pos[1,  -2,   :]   # 右上
        out[-1,  0,   :] = 2*pos[-1,  0,   :] - pos[-2,  1,   :]   # 左下
        out[-1, -1,   :] = 2*pos[-1, -1,   :] - pos[-2, -2,   :]   # 右下

        return out


    def cal_metrics(self,pos):

        X = pos[:,:,0]
        Y = pos[:,:,1]

        X_t =  torch.from_numpy(X).unsqueeze(0).unsqueeze(0)
        Y_t =  torch.from_numpy(Y).unsqueeze(0).unsqueeze(0)

        kernel_xi = torch.Tensor( [-0.5,0,0.5]  ).cpu().unsqueeze(0).unsqueeze(1).unsqueeze(2).to(torch.float64)
        kernel_eta = torch.Tensor( [0.5,0,-0.5]  ).cpu().unsqueeze(0).unsqueeze(1).unsqueeze(3).to(torch.float64)

        dx_dxi = torch.zeros_like(torch.from_numpy(X).unsqueeze(0).unsqueeze(0)).to(torch.float64)
        dy_dxi = torch.zeros_like(torch.from_numpy(X).unsqueeze(0).unsqueeze(0)).to(torch.float64)
        dx_deta =torch.zeros_like(torch.from_numpy(X).unsqueeze(0).unsqueeze(0)).to(torch.float64)
        dy_deta = torch.zeros_like(torch.from_numpy(X).unsqueeze(0).unsqueeze(0)).to(torch.float64)

        dx_dxi_ = F.conv2d(torch.from_numpy(X).unsqueeze(0).unsqueeze(0).to(torch.float64), kernel_xi, padding=(0,0))
        dy_dxi_ = F.conv2d(torch.from_numpy(Y).unsqueeze(0).unsqueeze(0).to(torch.float64), kernel_xi, padding=(0,0))
        dx_deta_ = F.conv2d(torch.from_numpy(X).unsqueeze(0).unsqueeze(0).to(torch.float64), kernel_eta, padding=(0,0))
        dy_deta_ = F.conv2d(torch.from_numpy(Y).unsqueeze(0).unsqueeze(0).to(torch.float64), kernel_eta, padding=(0,0))
        
        dx_dxi[:,:,:,1:-1] = dx_dxi_
   
        
        dx_dxi[:,:,:,-1] = (3*X_t[:,:,:,-1]-4*X_t[:,:,:,-2]+X_t[:,:,:,-3])/2
        dx_dxi[:,:,:,0]  = (-3*X_t[:,:,:,0]+4*X_t[:,:,:,1]-X_t[:,:,:,2])/2

        dy_dxi[:,:,:,1:-1] = dy_dxi_

        dy_dxi[:,:,:,-1] = (3*Y_t[:,:,:,-1]-4*Y_t[:,:,:,-2]+Y_t[:,:,:,-3])/2
        dy_dxi[:,:,:,0]  = (-3*Y_t[:,:,:,0]+4*Y_t[:,:,:,1]-Y_t[:,:,:,2])/2

        dx_deta[:,:,1:-1,:] = dx_deta_

        dx_deta[:,:,-1,:] = (-3*X_t[:,:,-1,:]+4*X_t[:,:,-2,:]-X_t[:,:,-3,:])/2
        dx_deta[:,:,0,:] = (3*X_t[:,:,0,:]-4*X_t[:,:,1,:]+X_t[:,:,2,:])/2


        dy_deta[:,:,1:-1,:] = dy_deta_

        dy_deta[:,:,-1,:] = (-3*Y_t[:,:,-1,:]+4*Y_t[:,:,-2,:]-Y_t[:,:,-3,:])/2
        dy_deta[:,:,0,:] = (3*Y_t[:,:,0,:]-4*Y_t[:,:,1,:]+Y_t[:,:,2,:])/2        


        det_inverse = (dx_dxi*dy_deta-dy_dxi*dx_deta)
        
        det = (1/det_inverse)
        dxi_dx = (det*dy_deta)
        dxi_dy = (-det*dx_deta)
        deta_dx = (-det*dy_dxi)
        deta_dy = (det*dx_dxi)
     
        dxi_dx = dxi_dx.to(torch.float32).reshape(-1,1).numpy()
        dxi_dy = dxi_dy.to(torch.float32).reshape(-1,1).numpy()
        deta_dx = deta_dx.to(torch.float32).reshape(-1,1).numpy()
        deta_dy = deta_dy.to(torch.float32).reshape(-1,1).numpy()
        J = det.to(torch.float32).reshape(-1,1).numpy()

        return np.concatenate((dxi_dx,dxi_dy,deta_dx,deta_dy,J),axis=1)
    


class StructuredGrid_Transformer():
    def __init__(self,grid_file=None,top_file=None,file_dir=None,case_name=None,path=None):
        
        self.path = path

        self.ghost_layer = 1

        self.topology = TopologyManager(top_file,self.ghost_layer)

        self.geometry = GeometryManager(grid_file,self.topology ,self.ghost_layer)

        self._create_GNN_elements_FD_stencils()

    def _create_GNN_elements_FD_stencils(self):



        extended_block_metrics=[]

        original_block_metrics=[]

        extend_index = []

        node_type = []

        extended_node_type = [] 

        block_shape = []


        original_index = []

        boundary_ghost_stencil = []
        
        # 保存扩展网格的全局索引（用于内部计算的连续编号）
        extended_global_index = []

        accumulate = 0
        
        for i,value in enumerate(self.topology.block_topology):

            extend_index.append(value["extended_nodes_index"].reshape(-1))

            node_type.append(value["block_nodes_type"].reshape(-1))
            extended_nodes_index = value["extended_nodes_index"]
            extended_node_type.append(value["extended_node_type"].reshape(-1))
      
            block_shape.append(value["nodes_shape"])

            original_index.append(value["block_nodes_index"].reshape(-1))

            H,W = value["extended_nodes_shape"]

            extended_idx = np.arange(H*W).reshape(H,W)+accumulate

            extended_node_type_block = value["extended_node_type"].reshape(H, W)

            # 处理ghost层中的WALL节点：只修改连接面方向拓展的
            g = self.ghost_layer
            
            print(f"\n块 {i} WALL边界检测:")
            
            # 上边ghost层 [0:g, :]（不含角点）
            top_ghost = extended_node_type_block[0:g, g:-g]
            has_top_cut = np.any(top_ghost == NodeType.CUT1TO1)
            has_top_wall = np.any(top_ghost == NodeType.WALL)
            is_top_mixed = has_top_cut and has_top_wall  # 混合边界，需要修改
            print(f"  上边ghost层: CUT1TO1={has_top_cut}, WALL={has_top_wall}, 混合={is_top_mixed}")
            
            # 下边ghost层 [-g:, :]（不含角点）
            bottom_ghost = extended_node_type_block[-g:, g:-g]
            has_bottom_cut = np.any(bottom_ghost == NodeType.CUT1TO1)
            has_bottom_wall = np.any(bottom_ghost == NodeType.WALL)
            is_bottom_mixed = has_bottom_cut and has_bottom_wall
            print(f"  下边ghost层: CUT1TO1={has_bottom_cut}, WALL={has_bottom_wall}, 混合={is_bottom_mixed}")
            
            # 左边ghost层 [:, 0:g]（不含角点）
            left_ghost = extended_node_type_block[g:-g, 0:g]
            has_left_cut = np.any(left_ghost == NodeType.CUT1TO1)
            has_left_wall = np.any(left_ghost == NodeType.WALL)
            is_left_mixed = has_left_cut and has_left_wall
            print(f"  左边ghost层: CUT1TO1={has_left_cut}, WALL={has_left_wall}, 混合={is_left_mixed}")
            
            # 右边ghost层 [:, -g:]（不含角点）
            right_ghost = extended_node_type_block[g:-g, -g:]
            has_right_cut = np.any(right_ghost == NodeType.CUT1TO1)
            has_right_wall = np.any(right_ghost == NodeType.WALL)
            is_right_mixed = has_right_cut and has_right_wall
            print(f"  右边ghost层: CUT1TO1={has_right_cut}, WALL={has_right_wall}, 混合={is_right_mixed}")
            
            # 创建需要修改的mask：找到WALL节点中与CUT1TO1节点直接相邻的点（向量化方式）
            changed_mask = np.zeros((H, W), dtype=bool)
            
            # 找到所有WALL节点的位置
            wall_positions = np.argwhere(extended_node_type_block == NodeType.WALL)
            
            if len(wall_positions) > 0:
                # 提取所有WALL节点的i, j坐标
                wall_i = wall_positions[:, 0]
                wall_j = wall_positions[:, 1]
                
                # 计算4个邻居的坐标（上下左右）
                neighbors_i = np.stack([wall_i-1, wall_i+1, wall_i, wall_i], axis=1)  # shape: (N, 4)
                neighbors_j = np.stack([wall_j, wall_j, wall_j-1, wall_j+1], axis=1)  # shape: (N, 4)
                
                # 边界检查mask
                valid_mask = (neighbors_i >= 0) & (neighbors_i < H) & (neighbors_j >= 0) & (neighbors_j < W)
                
                # 检查有效邻居是否为CUT1TO1
                for k in range(4):  # 遍历4个方向
                    valid_k = valid_mask[:, k]
                    if np.any(valid_k):
                        neighbor_i_k = neighbors_i[valid_k, k]
                        neighbor_j_k = neighbors_j[valid_k, k]
                        is_cut1to1 = extended_node_type_block[neighbor_i_k, neighbor_j_k] == NodeType.CUT1TO1
                        # 将满足条件的WALL节点位置标记为True
                        changed_mask[wall_i[valid_k][is_cut1to1], wall_j[valid_k][is_cut1to1]] = True
            
            # 修改节点类型
            extended_node_type_block[changed_mask] = NodeType.NORMAL
            
            # 打印被修改的点
            if np.any(changed_mask):
                changed_indices = np.argwhere(changed_mask)
                print(f"  将 {len(changed_indices)} 个点从 WALL 改为 NORMAL")
                print(f"  被修改的点坐标 (i, j): {changed_indices.tolist()}")
            else:
                print(f"  没有需要修改的点")

            _boundary_ghost_stencil = self.get_boundary_layers_np(extended_idx,extended_node_type_block)

            boundary_ghost_stencil.append(_boundary_ghost_stencil)



            accumulate += H*W
        


        cells_node = self.get_cells_node( [arr.reshape(r, c) for arr, (r, c) in zip(original_index, block_shape)]).reshape(-1,4)
        edge_pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        edges = cells_node[:, edge_pairs].reshape(-1,2)
        sorted_edges = np.sort(edges, axis=1)
        unique_edges = np.unique(sorted_edges, axis=0)  
        self.edge_index = unique_edges.T
        self.cells_node = cells_node.reshape(-1)

        self.extend_index = np.concatenate(extend_index,axis=0)

        self.node_type = np.concatenate(node_type,axis=0)

        self.extended_node_type = np.concatenate(extended_node_type,axis=0)     

        self.boundary_ghost_stencil_index = np.concatenate(boundary_ghost_stencil,axis=0) 

        # 使用统一的索引构造函数
        indices = self._construct_all_indices(block_shape)
        
        # 设置类属性
        self.block_cells_node = indices['block_cells_node'].T
        self.edge_node_xi = indices['edge_node_xi'].T
        self.edge_node_eta = indices['edge_node_eta'].T
        self.neighbor_edge_xi = indices['neighbor_edge_xi'].T
        self.neighbor_edge_eta = indices['neighbor_edge_eta'].T
        self.neighbor_cell_xi = indices['neighbor_cell_xi'].T
        self.neighbor_cell_eta = indices['neighbor_cell_eta'].T       

        mesh_pos_unique = []

        unique_pos_dict={}

        extended_block_pos = []

        for k,v in self.geometry.block_geometry.items():

            extended_block_metrics.append(v["extended_block_metrics"].reshape(-1,5))

            original_block_metrics.append(v["original_block_metrics"].reshape(-1,5))
            extended_block_pos.append(v["extended_block_pos"].reshape(-1,2))
        
        self.mesh_pos = self.geometry.mesh_pos

        self.extended_block_pos = np.concatenate(extended_block_pos,axis=0)

        self.extended_block_metrics = np.concatenate(extended_block_metrics,axis=0)

        self.original_block_metrics = np.concatenate(original_block_metrics,axis=0)
        num_cells = self.cells_node.reshape(-1,4).shape[0]                
        for pos_i,pos in enumerate(self.mesh_pos):
            if not (str(pos)  in unique_pos_dict.keys()):
                mesh_pos_unique.append(pos)
                unique_pos_dict[str(pos)]=len(mesh_pos_unique)-1
            else:
                # where interface exists TODO:

                continue
        self.mesh_pos_unique = np.asarray(mesh_pos_unique)

        reduce_index = []
        block_mask_unique_list = []

        for k,v in self.geometry.block_geometry.items():
     
            block_mask_unique = np.zeros_like(self.mesh_pos[:,0:1])
     
            for block_i, block_pos in enumerate(v['mesh_pos'].reshape(-1,2)):
                if str(block_pos) in unique_pos_dict.keys():
                    block_mask_unique[block_i]=unique_pos_dict[str(block_pos)]
            block_mask_unique_list.append(block_mask_unique)
            block_index_unique = block_mask_unique[:,0].astype(np.int64)[0:v['nodes_shape'][0]*v['nodes_shape'][1]]

            reduce_index.append(block_index_unique)

        self.cells_node_unique = self.get_cells_node( [arr.reshape(r, c) for arr, (r, c) in zip(reduce_index, block_shape)]).reshape(-1,1)

        num_cells = self.cells_node_unique.reshape(-1,4).shape[0]

        self.cells_index = np.repeat(np.arange(num_cells, dtype=np.int64), 4)[:, None]
        
        edge_pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        edges = self.cells_node_unique.reshape(-1,4)[:, edge_pairs].reshape(-1,2)
        sorted_edges = np.sort(edges, axis=1)
        unique_edges = np.unique(sorted_edges, axis=0)  
        self.edge_index_unique = unique_edges

        # prepare for cells_face
        cells_face_node_cell_index_dict = {}
        face_list = self.edge_index_unique
        face_index = {}
        for i in range(face_list.shape[0]):
            face_index[str(face_list[i])] = i 
        cells_face = np.zeros_like(edges)[:,0:1] 
        

        for edges_node_index_i, edges_node_index in enumerate(edges):
            
            if str(np.sort(edges_node_index,axis=-1)) in cells_face_node_cell_index_dict.keys():
                cells_face_node_cell_index_dict[str(np.sort(edges_node_index,axis=-1))].append(self.cells_index[edges_node_index_i])
                
            else:
                cells_face_node_cell_index_dict[str(np.sort(edges_node_index,axis=-1))] = [self.cells_index[edges_node_index_i]]
            cells_face[edges_node_index_i] = face_index[str(np.sort(edges_node_index,axis=-1))]

      
        self.cells_face = cells_face 
        self.cells_face_unique = cells_face 
        
        self.reduce_index = np.concatenate(reduce_index,axis=0)

        self.node_type_unique = scatter_mean(np2torch(self.node_type),np2torch(self.reduce_index),dim=0).numpy()




    def get_boundary_layers_np(self,a: np.ndarray,extended_node_type):
        """
        a: (N, M)，按行优先存 0 ~ N*M-1
        返回:
            result: (K, 3)，每行 [边界编号, 第一层编号, 第二层编号]
        """
        assert a.ndim == 2
        N, M = a.shape

        boundary_list = []
        inner1_list = []
        inner2_list = []

        mask_bc_extended = (extended_node_type==NodeType.WALL)| (extended_node_type==NodeType.INFLOW)| (extended_node_type==NodeType.OUTFLOW)

        # -------- 顶边 i = 0, j = 0..M-1 --------
        j = np.arange(M)
        i = np.zeros_like(j)

        # 默认向内：往下走
        i1 = np.ones_like(j)        # 1
        i2 = np.full_like(j, 2)     # 2
        j1 = j.copy()
        j2 = j.copy()

        # 顶边角点：用对角线向内
        j0 = (j == 0)
        jL = (j == M - 1)
        # 左上角 (0,0) -> (1,1), (2,2)
        j1[j0] = 1
        j2[j0] = 2
        # 右上角 (0,M-1) -> (1,M-2), (2,M-3)
        j1[jL] = M - 2
        j2[jL] = M - 3

        # boundary_list.append(a[i, j].ravel())

        # inner1_list.append(a[i1, j1].ravel())
        # inner2_list.append(a[i2, j2].ravel())

        mask_bc = mask_bc_extended[i, j]
        if np.any(mask_bc):
            boundary_list.append(a[i[mask_bc], j[mask_bc]].ravel())
            inner1_list.append(a[i1[mask_bc], j1[mask_bc]].ravel())
            inner2_list.append(a[i2[mask_bc], j2[mask_bc]].ravel())
        # -------- 底边 i = N-1, j = 0..M-1 --------
        j = np.arange(M)
        i = np.full_like(j, N - 1)

        # 默认向内：往上走
        i1 = np.full_like(j, N - 2)
        i2 = np.full_like(j, N - 3)
        j1 = j.copy()
        j2 = j.copy()

        # 底边角点：对角线向内
        j0 = (j == 0)
        jL = (j == M - 1)
        # 左下角 (N-1,0) -> (N-2,1), (N-3,2)
        j1[j0] = 1
        j2[j0] = 2
        # 右下角 (N-1,M-1) -> (N-2,M-2), (N-3,M-3)
        j1[jL] = M - 2
        j2[jL] = M - 3

        # boundary_list.append(a[i, j].ravel())
        # inner1_list.append(a[i1, j1].ravel())
        # inner2_list.append(a[i2, j2].ravel())
        mask_bc = mask_bc_extended[i, j]
        if np.any(mask_bc):
            boundary_list.append(a[i[mask_bc], j[mask_bc]].ravel())
            inner1_list.append(a[i1[mask_bc], j1[mask_bc]].ravel())
            inner2_list.append(a[i2[mask_bc], j2[mask_bc]].ravel())

        # -------- 左边（去掉角点） j = 0, i = 1..N-2 --------
        if N > 2:
            i = np.arange(1, N - 1)
            j = np.zeros_like(i)

            # 向内：往右走
            i1 = i.copy()
            i2 = i.copy()
            j1 = np.ones_like(i)        # 1
            j2 = np.full_like(i, 2)     # 2

            # boundary_list.append(a[i, j].ravel())
            # inner1_list.append(a[i1, j1].ravel())
            # inner2_list.append(a[i2, j2].ravel())
            mask_bc = mask_bc_extended[i, j]
            if np.any(mask_bc):
                boundary_list.append(a[i[mask_bc], j[mask_bc]].ravel())
                inner1_list.append(a[i1[mask_bc], j1[mask_bc]].ravel())
                inner2_list.append(a[i2[mask_bc], j2[mask_bc]].ravel())

        # -------- 右边（去掉角点） j = M-1, i = 1..N-2 --------
        if N > 2:
            i = np.arange(1, N - 1)
            j = np.full_like(i, M - 1)

            # 向内：往左走
            i1 = i.copy()
            i2 = i.copy()
            j1 = np.full_like(i, M - 2)   # 第一层
            j2 = np.full_like(i, M - 3)   # 第二层

  
            mask_bc = mask_bc_extended[i, j]
            if np.any(mask_bc):
                boundary_list.append(a[i[mask_bc], j[mask_bc]].ravel())
                inner1_list.append(a[i1[mask_bc], j1[mask_bc]].ravel())
                inner2_list.append(a[i2[mask_bc], j2[mask_bc]].ravel())


        # 拼接并组装成 (K, 3)
        if len(boundary_list) > 0:
            boundary_idx = np.concatenate(boundary_list)
            inner1_idx = np.concatenate(inner1_list)
            inner2_idx = np.concatenate(inner2_list)
            result = np.stack([boundary_idx, inner1_idx, inner2_idx], axis=1)
        else:
            # 如果没有边界条件点，返回空的 (0, 3) 数组
            result = np.empty((0, 3), dtype=np.int64)
        return result

        

 
    def _construct_all_indices(self, block_shape):
        """
        统一的索引构造函数，处理所有类型的邻居关系索引
        返回包含所有索引的字典
        """
        # 1. 构造节点索引和block_cells_node
        node_indices = self._construct_node_indices(block_shape)
        
        # 2. 构造边的邻居索引
        edge_indices = self._construct_edge_neighbor_indices(node_indices['edge_node_xi'], node_indices['edge_node_eta'], block_shape)
        
        # 3. 构造单元的邻居索引  
        cell_indices = self._construct_cell_neighbor_indices(node_indices['block_cells_node'], block_shape)
        
        # 返回所有索引
        return {
            **node_indices,
            **edge_indices,
            **cell_indices
        }
    
    def _construct_node_indices(self, block_shape):
        """构造节点相关的索引"""
        idx = np.arange(self.extend_index.shape[0])
        
        # 定义需要应用的操作配置
        node_operations = [
            {'name': 'l_node', 'func': get_left, 'slice_type': 'xi'},
            {'name': 'r_node', 'func': get_right, 'slice_type': 'xi'}, 
            {'name': 'd_node', 'func': get_down, 'slice_type': 'eta'},
            {'name': 'u_node', 'func': get_up, 'slice_type': 'eta'}
        ]
        
        cell_operations = [
            {'name': 'p1', 'func': get_id1},
            {'name': 'p2', 'func': get_id2},
            {'name': 'p3', 'func': get_id3}, 
            {'name': 'p4', 'func': get_id4}
        ]
        
        results = {op['name']: [] for op in node_operations}
        cell_results = []
        
        start_idx = 0
        for row, col in block_shape:
            num_elements = (row+2*self.ghost_layer) * (col+2*self.ghost_layer)
            elements = idx[start_idx:start_idx + num_elements]
            start_idx += num_elements
            
            elements = elements.reshape((row+2*self.ghost_layer), (col+2*self.ghost_layer))
            
            # 处理节点操作
            central_xi_idx = elements[self.ghost_layer:-self.ghost_layer, :]
            central_eta_idx = elements[:, self.ghost_layer:-self.ghost_layer]
            
            for op in node_operations:
                if op['slice_type'] == 'xi':
                    result = op['func'](central_xi_idx)
                else:  # eta
                    result = op['func'](central_eta_idx)
                results[op['name']].append(result)
            
            # 处理单元操作
            cell_values = [op['func'](elements) for op in cell_operations]
            cell_results.append(np.stack(cell_values, axis=0).T)
        
        # 返回索引字典
        return {
            'block_cells_node': np.concatenate(cell_results, axis=0),
            'edge_node_xi': np.stack((np.concatenate(results['l_node']), np.concatenate(results['r_node'])), axis=0).T,
            'edge_node_eta': np.stack((np.concatenate(results['d_node']), np.concatenate(results['u_node'])), axis=0).T
        }
    
    def _construct_edge_neighbor_indices(self, edge_node_xi, edge_node_eta, block_shape):
        """构造边的邻居索引"""
        operations = [
            {'indices': np.arange(edge_node_xi.shape[0]), 'shapes': [(row, col+self.ghost_layer) for row, col in block_shape], 
             'funcs': [get_left, get_right], 'names': ['left_edge', 'right_edge']},
            {'indices': np.arange(edge_node_eta.shape[0]), 'shapes': [(row+self.ghost_layer, col) for row, col in block_shape], 
             'funcs': [get_down, get_up], 'names': ['down_edge', 'up_edge']}
        ]
        
        all_results = {}
        for op_config in operations:
            results = self._apply_block_operations(op_config['indices'], op_config['shapes'], op_config['funcs'])
            for name, result in zip(op_config['names'], results):
                all_results[name] = result
        
        return {
            'neighbor_edge_xi': np.stack((all_results['left_edge'], all_results['right_edge']), axis=0).T,
            'neighbor_edge_eta': np.stack((all_results['down_edge'], all_results['up_edge']), axis=0).T
        }
    
    def _construct_cell_neighbor_indices(self, block_cells_node, block_shape):
        """构造单元的邻居索引"""
        # block_cells_node的形状是(总单元数, 4)，所以单元总数是shape[0]
        total_cells = block_cells_node.shape[0]
        cells_idx = np.arange(total_cells)
        shapes = [(row+self.ghost_layer, col+self.ghost_layer) for row, col in block_shape]
        funcs = [get_left, get_right, get_up, get_down]
        
        left_cell, right_cell, up_cell, down_cell = self._apply_block_operations(cells_idx, shapes, funcs)
        
        return {
            'neighbor_cell_xi': np.stack((left_cell, right_cell), axis=0).T,
            'neighbor_cell_eta': np.stack((down_cell, up_cell), axis=0).T
        }
    
    def _apply_block_operations(self, indices, shapes, functions):
        """
        通用的块操作应用函数
        
        Args:
            indices: 全局索引数组
            shapes: 每个块的形状列表
            functions: 要应用的函数列表
            
        Returns:
            每个函数的结果列表
        """
        results = [[] for _ in functions]
        start_idx = 0
        
        for shape in shapes:
            num_elements = np.prod(shape)
            elements = indices[start_idx:start_idx + num_elements]
            start_idx += num_elements
            
            reshaped_elements = elements.reshape(shape)
                # 对于直接使用numpy的函数
            for i, func in enumerate(functions):
                result = func(reshaped_elements)
                results[i].append(result)
        
        return [np.concatenate(result_list) for result_list in results]
    


    def get_cells_node(self,block_index_list):
        cells_node  = []
        for id,i in enumerate(block_index_list):
            mask = i
            cells_node_0 = get_id1(mask)
            cells_node_1 = get_id2(mask)
            cells_node_2 = get_id3(mask)
            cells_node_3 = get_id4(mask)

            block_cells_node = np.stack((cells_node_0,cells_node_1,cells_node_2,cells_node_3),axis=-1).reshape(-1,4)
            cells_node.append(block_cells_node)
        return np.concatenate(cells_node,axis=0)
    


    def extract_mesh(self,):

        mesh = {"mesh_pos":torch.from_numpy(self.mesh_pos),
                "mesh_pos_unique":torch.from_numpy(self.mesh_pos_unique),
                "extended_block_pos":torch.from_numpy(self.extended_block_pos),
                "cells_node":torch.from_numpy(self.cells_node).to(torch.long),
                "cells_node_unique":torch.from_numpy(self.cells_node_unique).squeeze(-1).to(torch.long),
                "cells_face_unique": torch.from_numpy(self.cells_face_unique).squeeze(-1).long(),
                "cells_index":torch.from_numpy(self.cells_index).squeeze(-1).to(torch.long),
                "cells_face":torch.from_numpy(self.cells_face).squeeze(-1).to(torch.long),
                "node_type_unique":torch.from_numpy(self.node_type_unique).squeeze(-1).to(torch.long),
                "node_type":torch.from_numpy(self.node_type).squeeze(-1).to(torch.long),
                "extended_node_type":torch.from_numpy(self.extended_node_type).squeeze(-1).to(torch.long),  
                "boundary_ghost_stencil_index":torch.from_numpy(self.boundary_ghost_stencil_index).to(torch.long),
                "edge_index_unique":torch.from_numpy(self.edge_index_unique).to(torch.long),
                "edge_index":torch.from_numpy(self.edge_index).to(torch.long),
                "block_cells_node":torch.from_numpy(self.block_cells_node).to(torch.long),
                    "reduce_index":torch.from_numpy(self.reduce_index).to(torch.long),
                    "extend_index":torch.from_numpy(self.extend_index).to(torch.long),
                    "edge_node_xi":torch.from_numpy(self.edge_node_xi).to(torch.long),
                    "edge_node_eta":torch.from_numpy(self.edge_node_eta).to(torch.long),
                    "neighbor_edge_xi":torch.from_numpy(self.neighbor_edge_xi).to(torch.long),
                    "neighbor_edge_eta":torch.from_numpy(self.neighbor_edge_eta).to(torch.long),
                    "neighbor_cell_xi":torch.from_numpy(self.neighbor_cell_xi).to(torch.long),
                    "neighbor_cell_eta":torch.from_numpy(self.neighbor_cell_eta).to(torch.long),
                    "extended_block_metrics":torch.from_numpy(self.extended_block_metrics),
                    "original_block_metrics":torch.from_numpy(self.original_block_metrics),
         
                   }
                    


        h5_dataset = extract_mesh_state(
            mesh,
            path=self.path,
        )

        return h5_dataset






# Define the processing function
def process_file( grid_path,top_path, path, queue):
    
    file_name = os.path.basename(grid_path)
    file_dir = os.path.dirname(grid_path)
    case_name = os.path.basename(file_dir)
    path["file_dir"] = file_dir
    path["case_name"] = case_name
    path["file_name"] = file_name
    

    data = StructuredGrid_Transformer(
            grid_file=grid_path,
            top_file=top_path,
            file_dir=file_dir,
            case_name=case_name,
            path=path,
        )



    h5_dataset = data.extract_mesh()

    # Put the results in the queue
    queue.put((h5_dataset, case_name, file_dir))


# Writer process function
def writer_process(queue, path):

    while True:

        # Get data from queue
        h5_data, case_name, file_dir = queue.get()
        
        # Break if None is received (sentinel value)
        if h5_data is None:
            break
        
        os.makedirs(file_dir, exist_ok=True)
        h5_writer = h5py.File(f"{file_dir}/{case_name}.h5", "w")

        # Write dataset key value
        group = h5_writer.create_group(case_name)
        for key, value in h5_data.items():
            if key in group:
                del group[key]
            group.create_dataset(key, data=value)

        print(f"{case_name} mesh has been writed")

    # 关闭所有的writer
    h5_writer.close()


if __name__ == "__main__":
    # for debugging

    debug_grid_path = None
    debug_top_path = None



    path = {
            "simulator": "???",
            "gird_path": "grid_example/Cavity301",
            "mesh_only": True,
        }

    # stastic total number of data samples
    total_samples = 0
    grid_file_paths = []
    top_file_paths = []
    for subdir, _, files in os.walk(path["gird_path"]):
        for data_name in files:
            if data_name.endswith(".dat"):
                grid_file_paths.append(os.path.join(subdir, data_name))

    for subdir, _, files in os.walk(path["gird_path"]):
        for data_name in files:
            if data_name.endswith(".inp"):
                top_file_paths.append(os.path.join(subdir, data_name))    

    # 统计选中的文件总数
    assert total_samples == 0, "Found no mesh files"
    total_samples = len(grid_file_paths)
    print("total samples: ", total_samples)

    if debug_grid_path is not None:
        multi_process = 1
    elif total_samples < multiprocessing.cpu_count():
        multi_process = total_samples
    else:
        multi_process = multiprocessing.cpu_count()

    # Start to convert data using multiprocessing
    global_data_index = 0
    with multiprocessing.Pool(multi_process) as pool:
        manager = multiprocessing.Manager()
        queue = manager.Queue()

        # Start writer process
        writer_proc = multiprocessing.Process(target=writer_process, args=(queue, path))
        writer_proc.start()

        if debug_grid_path is not None:
            # for debuging
            results = process_file(
             
                        debug_grid_path,
                        debug_top_path,
                        path,
                        queue,
                    ),
        else:
            # Process files in parallel
            results = [
                pool.apply_async(
                    process_file,
                    args=(
             
                        grid_file_path,
                  
                        top_file_path,
                        path,
                        queue,
                    ),
                )
                for (grid_file_index, grid_file_path), (top_file_index, top_file_path) in zip(enumerate(grid_file_paths), enumerate(top_file_paths))
            ]

            # Wait for all processing processes to finish
            for res in results:
                res.get()

        # Send sentinel value to terminate writer process
        queue.put((None, None, None))
        writer_proc.join()

    print("done")
