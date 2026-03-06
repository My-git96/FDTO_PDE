"""Utility functions for reading the datasets."""

import sys
import os
file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(file_dir)

from torch._refs import meshgrid
from torch_scatter import scatter_mean
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from scipy.spatial import KDTree


import torch

from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.loader import DataLoader as torch_geometric_DataLoader
from torch.utils.data import Sampler
import datetime
from Extract_mesh.write_tec import write_u_tecplotzone,write_uvp_tecplotzone
from utils.utilities import find_east, find_west, find_north, find_south, node_1, node_3, node_7, node_9,find_mid
from utils.utilities import export_uvp_to_tecplot, export_u_to_tecplot
from torch_scatter import scatter_mean
from dataset.Load_mesh import H5CFDdataset, CFDdatasetBase

class Data_Pool:
    def __init__(self, params=None,device=None,state_save_dir=None,):
        self.params = params
        self.device = device
        
        try:
            if not (state_save_dir.find("traing_results") != -1):
                os.makedirs(f"{state_save_dir}/traing_results", exist_ok=True)
                self.state_save_dir = f"{state_save_dir}/traing_results"
        except:
            print(
                ">>>>>>>>>>>>>>>>>>>>>>>Warning, no state_save_dir is specified, check if traing states is specified<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            )
        
        # 绘制被重置的这个case当前状态
        self._plot_env=True

        
    def _set_reset_env_flag(self, flag=False, rst_time=1):
        self.reset_env_flag = flag
        self.rst_time = rst_time

    def load_mesh_to_cpu(
        self,
        dataset_dir=None,
    ):
        
        valid_h5file_paths = []
        for subdir, _, files in os.walk(dataset_dir):
            for data_name in files:
                if data_name.endswith(".h5"):
                    valid_h5file_paths.append(os.path.join(subdir, data_name))

        if not valid_h5file_paths:
            raise FileNotFoundError(
                f"No .h5 files found in dataset_dir: '{dataset_dir}'. "
                f"Please check the path is correct."
            )

        mesh_dataset = H5CFDdataset(
            params=self.params, file_list=valid_h5file_paths
        )

        mesh_loader = torch_DataLoader(
            mesh_dataset,
            batch_size=4,
            num_workers=4,
            pin_memory=False,
            collate_fn=lambda x: x,
        )

        print("loading whole dataset to cpu")
        self.meta_pool = []
        self.uvp_node_pool = []

        start_idx = 0
        while True:
            for _, trajs in enumerate(mesh_loader):
                tmp = list(trajs)
                for meta_data, init_uvp_node in tmp:
                    meta_data["global_idx"] = torch.arange(start_idx,start_idx+init_uvp_node.shape[0])
                    self.meta_pool.append(meta_data)
                    self.uvp_node_pool.append(init_uvp_node)

                    start_idx += init_uvp_node.shape[0]
 
                    if len(self.meta_pool)>=self.params.dataset_size:
                        break
                    
            if len(self.meta_pool)>=self.params.dataset_size:
                break
            
        self.uvp_node_pool = torch.cat(self.uvp_node_pool, dim=0)

        self.dataset_size = len(self.meta_pool)
        self.params.dataset_size = self.dataset_size
        

        
        # 控制画图个数的文件夹分组
        self.plot_count = 0
        return self.dataset_size, self.params
    
    @staticmethod
    def datapreprocessing(
        graph_node,graph_Index
    ):
        senders, receivers = graph_node.edge_index
        uvp_node = graph_node.x[:,0:3]
         
        two_way_senders = torch.cat((senders,receivers),dim=0)
        two_way_receivers = torch.cat((receivers,senders),dim=0)
  
        releative_mesh_pos = torch.index_select(graph_node.pos,0,two_way_senders) - torch.index_select(graph_node.pos,0,two_way_receivers)
        edge_length = torch.norm(releative_mesh_pos,dim=1,keepdim=True)
        
        releative_node_attr =( torch.index_select(uvp_node,0,two_way_senders) - torch.index_select(uvp_node,0,two_way_receivers))       
        

        # permute edge direction
        releative_mesh_pos = (
            torch.index_select(graph_node.pos, 0, two_way_senders)
            - torch.index_select(graph_node.pos, 0, two_way_receivers)
        ).to(torch.float32)

        edge_length = torch.norm(releative_mesh_pos, dim=1, keepdim=True)

        
        releative_node_attr = torch.index_select(
            uvp_node, 0, two_way_senders
        ) - torch.index_select(uvp_node, 0, two_way_receivers)



        graph_node.edge_attr = torch.cat(
            (releative_node_attr,releative_mesh_pos, edge_length), dim=1
        )
        pde_theta_node = graph_Index.pde_theta[graph_node.batch]
        graph_node.x = torch.cat((graph_node.x[:,0:3].clone(),graph_node.pos,pde_theta_node),dim=1)

        return graph_node
    
    def reset_env(self, plot=False, graph_node=None, physical_time=None, time_step=None):
        # 弹出第0个网格的mesh数据
        old_mesh = self.meta_pool.pop(0)
        old_global_idx = old_mesh["global_idx"]     
        # 绘图
        if plot:
            pos_all = graph_node.pos.cpu().numpy()
            pos_unique = torch.tensor(old_mesh["mesh_pos_unique"], dtype=torch.float32).to(self.device)
            pos_tree = KDTree(pos_all)
            _, unique_pos_indices = pos_tree.query(pos_unique.cpu().numpy(), k=1)
            global_idx_unique = graph_node.global_idx.cpu()[unique_pos_indices]
            batch_unique = graph_node.batch.cpu()[unique_pos_indices]     
            reduce_index = old_mesh["reduce_index"]
            if not isinstance(reduce_index, torch.Tensor):
                reduce_index = torch.tensor(reduce_index, dtype=torch.long).to(self.device)
            else:
                reduce_index = reduce_index.to(self.device)
            mesh_pos_unique_size = old_mesh["mesh_pos_unique"].shape[0]
            
            uvp_node = self.uvp_node_pool[old_global_idx].to(self.device)  
            uvp_unique = scatter_mean(uvp_node.clone(), reduce_index, dim=0, dim_size=mesh_pos_unique_size)
            uv_error = torch.zeros_like(uvp_unique[:, 0:1])
            p_error = torch.zeros_like(uvp_unique[:, 0:1])
            uvp_err = torch.cat([uvp_unique.cpu(), uv_error.cpu(),p_error.cpu()], dim=1)                    
            export_uvp_to_tecplot(old_mesh, uvp_err, datalocation="node",
                physical_time=physical_time,
                time_step=time_step,
                state_save_dir=self.state_save_dir)           
            self._plot_env = False
        # 移除属于第0个网格的uvp数据
        self.uvp_node_pool = self.uvp_node_pool[old_global_idx.shape[0]:]        
        for iidx in range(len(self.meta_pool)):
            cur_meta_data = self.meta_pool[iidx]
            cur_meta_data["global_idx"] -= old_global_idx.shape[0]
        # 接着生成新的网格数据，即重新选一个边界条件
        new_mesh, init_uvp = CFDdatasetBase.transform_mesh(
            old_mesh, 
            self.params
        )
        new_mesh["global_idx"] = torch.arange(
            self.uvp_node_pool.shape[0], self.uvp_node_pool.shape[0]+init_uvp.shape[0]
        )
        self.uvp_node_pool = torch.cat((self.uvp_node_pool, init_uvp), dim=0)
        self.meta_pool.append(new_mesh)


    
        
        
    def payback(self, uvp_new, graph_node,global_idx, physical_time=None, time_step=None):
        self.uvp_node_pool[global_idx,0:3] = uvp_new.data
        if self.reset_env_flag:
            for _ in range(self.rst_time):
                self.reset_env(plot=self._plot_env, graph_node=graph_node, physical_time=physical_time, time_step=time_step)
            self.reset_env_flag=False    
            self._plot_env = True
        self.plot_count+=1

    def update_env(self, mesh):
        
        mesh["time_steps"] += 1

        if "wave" in mesh["flow_type"]:
            (
                mesh,
                theta_PDE,
                sigma,
                source_pressure_node,
            ) = CFDdatasetBase.set_Wave_case(
                mesh,
                self.params,
                mesh["mean_u"].item(),
                mesh["rho"].item(),
                mesh["mu"].item(),
                mesh["source"].item(),
                mesh["aoa"].item(),
                mesh["dt"].item(),
                mesh["source_frequency"].item(),
                mesh["source_strength"].item(),
                time_index=mesh["time_steps"],
            )
            mesh["theta_PDE"] = theta_PDE
            mesh["sigma"] = sigma
            mesh["wave_uvp_on_node"][0, :, 2:3] += source_pressure_node

            return mesh

        else: 

            mesh = CFDdatasetBase.To_Cartesian(mesh,resultion=(300,100))

        return mesh
    
    def mirror_points_about_circle(self,
        P,
        C,
        R: float
        ) :
        """
        对一组圆内点 P 批量计算：
        -- 最近的圆面切点 B（shape (N,2)）
        -- 关于切点的外镜像点 P_mirror（shape (N,2)）

        参数:
            P (np.ndarray): 一组点坐标，形状 (N, 2)，每行 [x, y]，应满足距离 C 小于 R。
            C (np.ndarray): 圆心坐标，形状 (2,)。
            R (float): 圆的半径。

        返回:
            B (np.ndarray): 切点数组，形状 (N, 2)。
            P_mirror (np.ndarray): 镜像点数组，形状 (N, 2)。
        """
        # 向量差和对应距离
        v = P - C                    # shape (N,2)
        d = torch.linalg.norm(v, dim=1)  # shape (N,)
        
        # 检查异常
        # if torch.any(d == 0):
        #     raise ValueError("存在 P 与圆心重合，切点不唯一。")
        # if torch.any(d >= R):
        #     raise ValueError("存在 P 不在圆内，请确保所有点距离圆心小于半径。")
        
        # 计算批量单位法向量 (从圆心指向 P 的方向)
        normals = v / d[:,None]
        
        # 计算批量切点 B
        B = C + R * normals
        
        # 计算批量镜像点 P_mirror
        P_mirror = 2 * B - P
        
        return B, P_mirror, normals
    
    def mirror_points_about_naca(self, P, t=0.12):
        """
        对一组 NACA 翼型内部点 P 批量计算（解析方法）：
        -- 最近的翼型表面点 B（shape (N,2)）
        -- 关于表面点的外镜像点 P_mirror（shape (N,2)）
        -- 表面点处的单位法向量（shape (N,2)）

        参数:
            P (torch.Tensor): 一组点坐标，形状 (N, 2)，每行 [x, y]，应在 NACA 翼型内部。
            t (float): 厚度比 (默认 0.12 对应 NACA0012)。
            tol (float): 数值容差。

        返回:
            B (torch.Tensor): 表面点数组，形状 (N, 2)。
            P_mirror (torch.Tensor): 镜像点数组，形状 (N, 2)。
            normals (torch.Tensor): 单位法向量数组，形状 (N, 2)。
        
        原理:
            1. 对于每个内部点，直接计算其 x 坐标对应的上下表面 y 值
            2. 判断点更接近上表面还是下表面
            3. 解析计算该表面点的法向量
            4. 计算镜像点
        """

        
        # 固定参数（与 naca_mask 保持一致）
        c = 1.0
        x_qc = 1.0
        y_qc = 1.0
        
        # 提取坐标
        x = P[:, 0]
        y = P[:, 1]
        
        # 转换到标准翼型坐标系
        x_rel = x - x_qc
        y_rel = y - y_qc
        x_std = x_rel / c + 0.25
        y_std = y_rel / c
        
        # 限制 x_std 在 [0, 1] 范围内
        x_std = torch.clamp(x_std, 0.0, 1.0)
        
        # 计算该 x 位置处的厚度 y_t（NACA 4-digit 公式）
        y_t = (t / 0.2) * (
            0.2969 * torch.sqrt(x_std + 1e-10)
            - 0.1260 * x_std
            - 0.3516 * x_std**2
            + 0.2843 * x_std**3
            - 0.1015 * x_std**4
        )
        
        # 上表面和下表面的 y 坐标（标准坐标系）
        y_upper_std = y_t
        y_lower_std = -y_t
        
        # 判断点更接近上表面还是下表面
        dist_to_upper = torch.abs(y_std - y_upper_std)
        dist_to_lower = torch.abs(y_std - y_lower_std)
        is_closer_to_upper = dist_to_upper <= dist_to_lower
        
        # 选择最近的表面点（标准坐标系）
        y_surface_std = torch.where(is_closer_to_upper, y_upper_std, y_lower_std)
        
        # 转换回物理坐标系
        x_surface = (x_std - 0.25) * c + x_qc
        y_surface = y_surface_std * c + y_qc
        B = torch.stack([x_surface, y_surface], dim=1)
        
        # 计算厚度分布的导数 dy_t/dx（标准坐标系）
        dy_t_dx = (t / 0.2) * (
            0.2969 / (2 * torch.sqrt(x_std + 1e-10))
            - 0.1260
            - 2 * 0.3516 * x_std
            + 3 * 0.2843 * x_std**2
            - 4 * 0.1015 * x_std**3
        )
        
        # 计算法向量
        # 上表面：切向量 (1, dy_t/dx)，法向量 (-dy_t/dx, 1) 指向外侧
        # 下表面：切向量 (1, -dy_t/dx)，法向量 (dy_t/dx, -1) 指向外侧
        
        # 初始化法向量
        normal_x = torch.where(is_closer_to_upper, -dy_t_dx, dy_t_dx)
        normal_y = torch.where(is_closer_to_upper, 
                            torch.ones_like(dy_t_dx), 
                            -torch.ones_like(dy_t_dx))
        
        # 归一化法向量
        normals = torch.stack([normal_x, normal_y], dim=1)
        normals = normals / torch.norm(normals, dim=1, keepdim=True)
        
        # 计算镜像点：P_mirror = 2*B - P
        P_mirror = 2 * B - P
        
        return B, P_mirror, normals

    def naca_mask(self, pos, t=0.12, tol=1e-9):
        """
        固定 c=0.1, quarter_chord=(1.0,1.0) 的 NACA0012 掩码（无旋转）。
        输入:
            pos: torch.Tensor, shape [N,2]
            t: 厚度比 (默认 0.12)
            tol: 数值容差
        返回:
            mask: torch.BoolTensor, shape [N], True 表示在翼型内部（含边界）
        """
        assert pos.ndim == 2 and pos.shape[1] == 2, "pos 必须是 (N,2) 的张量"
    
        dtype = pos.dtype if pos.dtype.is_floating_point else torch.float32

        # 固定参数
        c = 1.0
        x_qc = 1.0
        y_qc = 1.0

        x = pos[:, 0].to(dtype)
        y = pos[:, 1].to(dtype)

        # 平移到以 quarter-chord 为原点
        x_rel = x - x_qc
        y_rel = y - y_qc

        # 缩放到标准翼型坐标 (弦长 1, quarter-chord at 0.25)
        x_std = x_rel / c + 0.25
        y_std = y_rel / c

        # 在弦投影范围内的掩码
        in_range = (x_std >= -tol) & (x_std <= 1.0 + tol)
        
        # 限制 x_std 在 [0, 1] 范围内（对所有点统一处理）
        xs = torch.clamp(x_std, 0.0, 1.0)
        
        # NACA 4-digit 厚度分布 (向量化，对所有点计算)
        y_t = (t / 0.2) * (
            0.2969 * torch.sqrt(xs + 1e-10)  # 添加小值避免 sqrt(0)
            - 0.1260 * xs
            - 0.3516 * xs**2
            + 0.2843 * xs**3
            - 0.1015 * xs**4
        )

        # 判断点是否在翼型内部
        inside = (y_std >= -y_t - tol) & (y_std <= y_t + tol)
        
        # 最终掩码：必须同时满足 in_range 和 inside
        mask = in_range & inside

        return mask

    
    def update_IBM_stencils(self,graph_node,row,col,parms):

        pos = graph_node.pos

        cells_node = graph_node.face.reshape(-1,4)

        node_idx = torch.arange(pos.shape[0],device=pos.device)

        if len(parms)==3:

            center_x,center_y,radius = parms

            geo_mask = (pos[:,0] - center_x)**2 + (pos[:,1] - center_y)**2 <= radius**2

        if len(parms)>3:
            geo_mask = self.naca_mask(pos)
            

        mask_grad  = node_idx[geo_mask]

        n_point = find_north(node_idx.reshape(row,col))
        s_point = find_south(node_idx.reshape(row,col))
        w_point = find_west(node_idx.reshape(row,col))
        e_point = find_east(node_idx.reshape(row,col))
        c1_point = node_1(node_idx.reshape(row,col))
        c2_point = node_3(node_idx.reshape(row,col))
        c3_point = node_7(node_idx.reshape(row,col))
        c4_point = node_9(node_idx.reshape(row,col))
        mid_point = find_mid(node_idx.reshape(row,col))

        stencil = torch.stack((mid_point,n_point,e_point,w_point,s_point,c1_point,c2_point,c3_point,c4_point),dim=-1)

        mask_mid = ~torch.isin(mid_point,mask_grad)
        mask_n = torch.isin(n_point,mask_grad)
        mask_e = torch.isin(e_point,mask_grad)
        mask_w = torch.isin(w_point,mask_grad)
        mask_s = torch.isin(s_point,mask_grad)
        mask_c1 = torch.isin(c1_point,mask_grad)
        mask_c2 = torch.isin(c2_point,mask_grad)
        mask_c3 = torch.isin(c3_point,mask_grad)
        mask_c4 = torch.isin(c4_point,mask_grad)    
        
        mask_ghost = (mask_n|mask_e|mask_w|mask_s|mask_c1|mask_c2|mask_c3|mask_c4)

        mask_stencil = torch.squeeze(mask_mid&mask_ghost)    

        selected_stencil  =  stencil[mask_stencil]

        unique_stencil_nodes = torch.unique(selected_stencil[:,1:].reshape(-1))

        query_stencil_index = unique_stencil_nodes[torch.isin(unique_stencil_nodes,mask_grad)]


        ghost_pos = pos[query_stencil_index]

        if len(parms)==3:
      
            BI_pos, mirror_pos, BI_unv = self.mirror_points_about_circle(ghost_pos,torch.stack((center_x,center_y),dim=1), radius)

        if len(parms)>3:
             BI_pos, mirror_pos, BI_unv = self.mirror_points_about_naca(ghost_pos, t=0.12)

      

        cell_center =  (pos[cells_node[:,0]]+ pos[cells_node[:,1]] + pos[cells_node[:,2]]+ pos[cells_node[:,3]])/4

        dists = torch.linalg.norm(mirror_pos[:, None, :] - cell_center[None, :, :], dim=2)

        closest_cell_indices = torch.argmin(dists, dim=1)

        sup_cells_node = cells_node[closest_cell_indices]

        mask_ghost_in_sup_cells_node = torch.isin(sup_cells_node, mask_grad)

        sup_cells_node_pos = pos[sup_cells_node]

        ghost_pos_in_sup_cells_node = sup_cells_node_pos[mask_ghost_in_sup_cells_node]

        mask = (ghost_pos_in_sup_cells_node[:, None, :] == ghost_pos[None, :, :]).all(dim=2)

        indices_in_ghost_pos = torch.argmax(mask.int(), dim=1).long()

        BI_pos_for_replace = BI_pos[indices_in_ghost_pos]

        sup_cells_node_pos[mask_ghost_in_sup_cells_node] = BI_pos_for_replace

        BI_unv_for_replace = BI_unv[indices_in_ghost_pos]

        mask_flow = ~torch.isin(node_idx,query_stencil_index)
        return (query_stencil_index, sup_cells_node, mask_flow, mirror_pos,sup_cells_node_pos, mask_ghost_in_sup_cells_node,BI_pos_for_replace,BI_unv_for_replace)

    def payback_u_for_vis(self, q, global_idx, num_graph, batch, physical_time=None, time_step=None):
        # update u pool
        self.uvp_node_pool[global_idx,0:1] = q.data[:,0:1]

        for i in range(num_graph):

            sample_mask = torch.where(batch==i)[0]
            u = q.data[sample_mask,0:1]
            mesh = self.meta_pool[i]
            export_u_to_tecplot(
                mesh=mesh,
                u=u,
                datalocation="node",
                physical_time=physical_time,
                time_step=time_step,
                state_save_dir=self.state_save_dir,
            )

    def payback_uvp_for_vis(self, q, global_idx, num_graph, batch, physical_time=None, time_step=None,to_export = True):

        self.uvp_node_pool[global_idx,0:3] = q.data[:,0:3]

        for i in range(num_graph):

            sample_mask = torch.where(batch==i)[0]
            #sample_mask = global_idx
            #uvp_err = torch.cat([self.uvp_node_pool[sample_mask, 0:3], err.unsqueeze(1)], dim=1)
            uvp_err = q.data[sample_mask,0:5]
            mesh = self.meta_pool[i]
            export_uvp_to_tecplot(
                mesh=mesh,
                uvp_err=uvp_err,
                datalocation="node",
                physical_time=physical_time,
                time_step=time_step,
                state_save_dir=self.state_save_dir,
                device=self.device,
                plot_count=self.plot_count,
                to_export = to_export
            )
        
class CustomGraphData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        offset_rules = {
            "edge_index": self.num_nodes,
            "face": self.num_nodes,
            "cells_node": self.num_nodes,
            "face_node": self.num_nodes,
            "cells_face": self.num_nodes,
            "neighbour_cell": self.num_nodes,
            "face_node_x": self.num_nodes,
            "support_edge": self.num_nodes,
            "periodic_idx": self.num_nodes,
            "init_loss":0,
            "case_name":0,
            "query": 0,
            "grids": 0,
            "pos": 0,
            "A_node_to_node": 0,
            "A_node_to_node_x": 0,
            "B_node_to_node": 0,
            "B_node_to_node_x": 0,
            "cells_area": 0,
            "node_type": 0,
            "graph_index": 0,
            "theta_PDE": 0,
            "sigma": 0,
            "uvp_dim": 0,
            "dt_graph": 0,
            "x": 0,
            "y": 0,
            "m_ids": 0,
            "m_gs": 0,
            "global_idx": 0,
        }
        return offset_rules.get(key, super().__inc__(key, value, *args, **kwargs))

    def __cat_dim__(self, key, value, *args, **kwargs):
        cat_dim_rules = {
            "x": 0,
            "pos": 0,
            "y": 0,
            "norm_y": 0,
            "query": 0,  # 保持query为列表，不进行拼接
            "grids": 0,  # 保持query为列表，不进行拼接
            "edge_index": 1,  # edge_index保持默认的offset拼接
            "face":0,
            "voxel": 0,
            "init_loss":0,
            "support_edge":1,
            "graph_index": 0,
            "global_idx": 0,
            "periodic_idx": 1,
        }
        return cat_dim_rules.get(key, super().__cat_dim__(key, value, *args, **kwargs))
    
class GraphNodeDataset(InMemoryDataset):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool
    
    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]
        extend_index = minibatch_data["extend_index"].to(torch.long)
        mesh_pos = minibatch_data["mesh_pos"].to(torch.float32)
        mesh_pos_unique = minibatch_data["mesh_pos_unique"].to(torch.float32)
        face_node = minibatch_data["edge_index"].long()
        node_type = minibatch_data["node_type"].long()
        case_name = minibatch_data["case_name"]
        global_idx = minibatch_data["global_idx"].long()
        cells_node = minibatch_data["cells_node"].long()

        target_on_node = minibatch_data["target|uvp"].to(torch.float32)

        original_block_metrics = minibatch_data['original_block_metrics'].to(torch.float32)
            
        graph_node = CustomGraphData(
       
       
            edge_index=face_node,
            extend_index = extend_index,
            face = cells_node,
   
            pos=mesh_pos,
            original_block_metrics = original_block_metrics,
            node_type=node_type,
            y=target_on_node,
            global_idx=global_idx,
            case_name=torch.tensor([ord(char) for char in (case_name)], dtype=torch.long),
            graph_index=torch.tensor([idx],dtype=torch.long),
        )

        return graph_node




class Graph_INDEX_Dataset(InMemoryDataset):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool
    
    @property
    def params(self):
        return self.base_dataset.params

    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]
      
        theta_PDE = minibatch_data["theta_PDE"].to(torch.float32)
        sigma = minibatch_data["sigma"].to(torch.float32)
        uvp_dim = minibatch_data["uvp_dim"].to(torch.float32)
        dt_graph = minibatch_data["dt_graph"].to(torch.float32)
        relaxtion = torch.tensor([minibatch_data["solving_params"]["relaxtion"]]).reshape(-1,1).to(torch.float32)

        graph_Index = CustomGraphData(
            x=torch.tensor([idx],dtype=torch.long),
            pde_theta=theta_PDE,
            sigma=sigma,
            uvp_dim=uvp_dim,
 
            dt_graph=dt_graph,
            relaxtion = relaxtion,
            graph_index=torch.tensor([idx],dtype=torch.long),
        )

        return graph_Index
    

class GraphNode_uniqueDataset(InMemoryDataset):
    
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        
        minibatch_data = self.pool[idx]        
        """Optional node attr"""
      
        mesh_pos = minibatch_data['mesh_pos_unique'].to(torch.float32)
        face_node = minibatch_data['edge_index_unique'].to(torch.long)
        cells_node = minibatch_data['cells_node_unique'].to(torch.long)
        node_type = minibatch_data['node_type_unique']
        reduce_index = minibatch_data["reduce_index"].to(torch.long)
        graph_nodeuns = Data(x = mesh_pos,
                        edge_index = face_node.T,
                        face = cells_node.T,
                        pos=mesh_pos,
                        node_type=node_type,
                        reduce_index = reduce_index,
                        graph_index= torch.as_tensor([idx])) 
        return graph_nodeuns
    

class GraphExtended_Edge_xiDataset(InMemoryDataset):
    
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        
        minibatch_data = self.pool[idx]        
        
        neighbor_edge_xi = minibatch_data['neighbor_edge_xi'].to(torch.long)
        tmp = minibatch_data['edge_node_xi'].to(torch.long)[:,0:1]
        graph_extended_edge_xi = Data(x=tmp,
                          face=neighbor_edge_xi,
                          graph_index= torch.as_tensor([idx]),
                         )
        
        return graph_extended_edge_xi

class GraphExtended_Edge_etaDataset(InMemoryDataset):
    
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        
        minibatch_data = self.pool[idx]        
        
        neighbor_edge_eta = minibatch_data['neighbor_edge_eta'].to(torch.long)
        tmp = minibatch_data['edge_node_eta'].to(torch.long)[:,0:1]
        graph_extended_edge_eta = Data(x=tmp,
                          face=neighbor_edge_eta,
                          graph_index= torch.as_tensor([idx]),
                         )
        
        return graph_extended_edge_eta

class GraphExtended_CellDataset(InMemoryDataset):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]        
        # current_time_steps = torch.as_tensor([minibatch_data['time_steps']]).to(torch.long)

        # cell_attr
        tmp = minibatch_data['block_cells_node'].to(torch.long)[:,0:1]
        neighbor_cell_xi = minibatch_data['neighbor_cell_xi'].to(torch.long)
        neighbor_cell_eta = minibatch_data['neighbor_cell_eta'].to(torch.long)

        graph_extended_block_cell = Data(x = tmp,
                        xi_cell_index=neighbor_cell_xi,
                        eta_cell_index = neighbor_cell_eta,
                        graph_index= torch.as_tensor([idx]),
          )
        
        return graph_extended_block_cell 

class GraphExtended_NodeDataset(InMemoryDataset):
    
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        
        minibatch_data = self.pool[idx]        
        """Optional node attr"""

        edge_node_xi = minibatch_data['edge_node_xi'].to(torch.long)
        edge_node_eta = minibatch_data['edge_node_eta'].to(torch.long)
        block_cells_node = minibatch_data['block_cells_node'].to(torch.long)

        extended_block_pos = minibatch_data['extended_block_pos'].to(torch.float32)
        boundary_ghost_stencil_index = minibatch_data['boundary_ghost_stencil_index'].to(torch.long)

        extended_node_type = minibatch_data['extended_node_type'].to(torch.long)

        extended_block_metrics = minibatch_data['extended_block_metrics'].to(torch.float32)

        
        graph_extended_node = Data(x = extended_block_pos,
                        boundary_ghost_stencil_index = boundary_ghost_stencil_index,
        
                        extended_block_metrics = extended_block_metrics,
                        block_cells_node_index = block_cells_node,
             
                        edge_node_xi_index  = edge_node_xi,
                        edge_node_eta_index = edge_node_eta,

                        node_type = extended_node_type,
 
                        graph_index= torch.as_tensor([idx])) 
        return graph_extended_node

class SharedSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.epoch = 0
        self.specific_indices = None  # 用于存储特定的索引

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.specific_indices is not None:
            return iter(self.specific_indices)
        return iter(torch.randperm(len(self.data_source), generator=g).tolist())

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = int(datetime.datetime.now().timestamp())

    def set_specific_indices(self, indices):
        self.specific_indices = indices

class CustomDataLoader:
    def __init__(
        self,
        graph_node_dataset,
        graph_extended_edge_xi_dataset,
        graph_extended_edge_eta_dataset,
        graph_extended_cell_dataset,
        graph_extended_node_dataset,
        graph_Index_dataset,
        batch_size,
        sampler,
        num_workers=4,
        pin_memory=False,
    ):
        # 保存输入参数到实例变量
        self.graph_node_dataset = graph_node_dataset
        self.graph_extended_edge_xi_dataset = graph_extended_edge_xi_dataset
        self.graph_extended_edge_eta_dataset = graph_extended_edge_eta_dataset
        self.graph_extended_cell_dataset = graph_extended_cell_dataset
        self.graph_extended_node_dataset = graph_extended_node_dataset
        self.graph_Index_dataset = graph_Index_dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # 初始化DataLoaders
        self.loader_A = torch_geometric_DataLoader(
            graph_node_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_B = torch_geometric_DataLoader(
            graph_extended_edge_xi_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_C = torch_geometric_DataLoader(
            graph_extended_edge_eta_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_D = torch_geometric_DataLoader(
            graph_extended_cell_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.loader_E = torch_geometric_DataLoader(
            graph_extended_node_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_F= torch_geometric_DataLoader(
            graph_Index_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


    def __iter__(self):
        return zip(
            self.loader_A, self.loader_B, self.loader_C, self.loader_D, self.loader_E,self.loader_F
        )

    def __len__(self):
        return min(
            len(self.loader_A),
            len(self.loader_B),
            len(self.loader_C),
            len(self.loader_D),
            len(self.loader_E),
            len(self.loader_F),
        )

    def get_specific_data(self, indices):
        # 设置Sampler的特定索引
        self.sampler.set_specific_indices(indices)

        # 重新创建DataLoaders来使用更新的Sampler
        self.loader_A = torch_geometric_DataLoader(
            self.graph_node_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_B = torch_geometric_DataLoader(
            self.graph_extended_edge_xi_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_C = torch_geometric_DataLoader(
            self.graph_extended_edge_eta_dataset,  
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_D = torch_geometric_DataLoader(
            self.graph_extended_cell_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        self.loader_E = torch_geometric_DataLoader(
            self.graph_extended_node_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_F = torch_geometric_DataLoader(
            self.graph_Index_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


        graph_node, graph_extended_edge_xi, graph_extended_edge_eta, graph_extended_cell_dataset, graph_extended_node_dataset, graph_Index = next(
            iter(self)
        )

        minibatch_data = self.graph_node_dataset.pool[indices[0]]

        origin_mesh_path = "".join(
            [chr(int(f)) for f in minibatch_data["origin_mesh_path"][0, :, 0].numpy()]
        )

        flow_type = minibatch_data["flow_type"]
        if ("cavity" in flow_type) or ("possion" in flow_type):
            has_boundary = False
        else:
            has_boundary = True

        return (
            graph_node,
            graph_extended_edge_xi,
            graph_extended_edge_eta,
            graph_extended_cell_dataset,
            graph_extended_node_dataset,
            graph_Index,
            has_boundary,
            origin_mesh_path,
        )

class DatasetFactory:
    def __init__(
        self,
        params=None,
        dataset_dir=None,
        state_save_dir=None,
        device=None,
    ):
        self.base_dataset = Data_Pool(
            params=params,
            device=device,
            state_save_dir=state_save_dir,
        )

        self.dataset_size, self.params = self.base_dataset.load_mesh_to_cpu(
            dataset_dir=dataset_dir,
        )

    def create_datasets(self, batch_size=100, num_workers=4, pin_memory=True):
        graph_node_dataset = GraphNodeDataset(base_dataset=self.base_dataset)
        graph_extended_edge_xi_dataset = GraphExtended_Edge_xiDataset(base_dataset=self.base_dataset)
        graph_extended_edge_eta_dataset = GraphExtended_Edge_etaDataset(base_dataset=self.base_dataset)
        graph_extended_cell_dataset = GraphExtended_CellDataset(base_dataset=self.base_dataset)
        graph_extended_node_dataset = GraphExtended_NodeDataset(base_dataset=self.base_dataset)
        graph_Index_dataset = Graph_INDEX_Dataset(base_dataset=self.base_dataset)

        # 创建SharedSampler并将其传递给CustomDataLoader

        sampler = SharedSampler(graph_node_dataset)

        loader = CustomDataLoader(
            graph_node_dataset,
            graph_extended_edge_xi_dataset,
            graph_extended_edge_eta_dataset,
            graph_extended_cell_dataset,
            graph_extended_node_dataset,
            graph_Index_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return self.base_dataset, loader, sampler
