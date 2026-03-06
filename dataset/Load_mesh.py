"""Utility functions for reading the datasets."""

import sys
import os
file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(file_dir)

from torch.utils.data import Dataset

import json
import random
import torch
import numpy as np
import h5py
import math
from utils import get_param
from utils.utilities import (
 
    calc_node_centered_with_cell_attr,
)
from utils.utilities import NodeType
from Extract_mesh.to_h5 import build_k_hop_edge_index
from torch_geometric.nn import knn
from torch_geometric import utils as pyg_utils
from dataset.Set_BC import velocity_profile
from utils.utilities import generate_boundary_zone

class CFDdatasetBase:
    # Base class for CFDdataset with process_trajectory method
    @staticmethod
    def select_PDE_coef(theta_PDE_list=None):
        (
            mean_U,
            rho,
            mu,
            source,
            aoa,
            dt,
            L,
        ) = random.choice(theta_PDE_list)

        return (
            mean_U, 
            rho, 
            mu, 
            source,
            aoa, 
            dt, 
            L,
        )

    @staticmethod
    def calc_charactisc_length(mesh):
        """prepare data for cal_relonyds_number"""
        # 输入为二维点云坐标，形状为 (N, 2)，其中 N 是点的数量
        # 扩展维度以便计算所有点对之间的欧几里得距离
        mesh_pos = mesh["node|pos"]
        surf = mesh["node|surf_mask"]
        
        if not surf.any():
            return torch.zeros(1) # There`s no surface in the mesh
        
        surf_pos = mesh_pos[surf]
        
        points_expanded1 = surf_pos.unsqueeze(0)  # 形状 (1, N, 2)
        points_expanded2 = surf_pos.unsqueeze(1)  # 形状 (N, 1, 2)
        
        # 计算所有点对之间的距离
        distances = torch.norm(points_expanded1 - points_expanded2, dim=2)  # 形状 (N, N)
        
        # 返回距离的最大值
        max_distance = torch.max(distances)

        return max_distance

    @staticmethod
    def init_env(
        mesh,
        max_u=None,
        dimless=False,
    ):
        # init node uvp
   
        node_pos = mesh["mesh_pos"]
        uv_node, p_node = velocity_profile(
            inlet_node_pos=node_pos,
            max_u=max_u,
            aoa=mesh["aoa"],
            inlet_type=mesh["init_field_type"],
        )
        
        # set uniform initial field value
        uvp_node = torch.cat(
            (
                uv_node, 
                p_node
            ),
            dim=1
        ).to(torch.float32)
        
        # generate BC mask
        node_type = mesh["node_type"].long().squeeze()
        Wall_mask = (node_type == NodeType.WALL)
        Inlet_mask = (node_type == NodeType.INFLOW)|(node_type==NodeType.PRESS_POINT)

        if "Cavity" or "cavity" in mesh["case_name"]:
           extended_nt = mesh["extended_node_type"]

           extended_pos = mesh["extended_block_pos"]

           xmin, ymin = extended_pos[:,0].min(), extended_pos[:,1].min()

           pressure_mask = (extended_pos[:,0]== xmin) & (extended_pos[:,1]==ymin) 

           extended_nt[pressure_mask] = NodeType.PRESS_POINT
    
        
        # generate velocity profile
        inlet_uvp_node, _ = velocity_profile(
            inlet_node_pos=node_pos[Inlet_mask],
            max_u=max_u,
            aoa=mesh["aoa"],
            inlet_type=mesh["inlet_type"],
        )
        inlet_uvp_node = inlet_uvp_node.to(torch.float32)
        
        # apply velocity profile and boundary condition
        uvp_node[Inlet_mask,0:2] = inlet_uvp_node[:,0:2]
        uvp_node[Wall_mask,0:2] = 0
  
        # store target node for dirchlet BC and make dimless if possible
    
        if dimless:
             mesh["target|uvp"] = uvp_node[:,0:2].clone() / max_u
        else:
            mesh["target|uvp"] = uvp_node[:,0:2].clone()


   
        
        return mesh, uvp_node

    @staticmethod
    def set_PDE_theta(mesh, params, max_velocity, rho, mu, source, aoa, dt, dL):
        """
        设置用于 PDE 求解的参数 theta_PDE, 并将其添加到 mesh 字典中。

        参数：
        - mesh: 包含网格信息的字典。
        - params: 参数配置对象。
        - max_velocity: 入口最大速度的标量值。
        - rho: 流体密度。
        - mu: 流体黏度。
        - source: 源项大小。
        - aoa: 攻角(angle of attack)以度为单位。
        - dt: 时间步长。
        - dL: 特征长度。

        返回：
        - mesh: 更新后的网格信息字典，包含了计算所得的 PDE 参数。
        - U_in: 乘上攻角之后的入口速度的二维张量。
        """
        U_in = max_velocity*torch.tensor(
            [math.cos(math.radians(aoa)), math.sin(math.radians(aoa))]
        )
        
        mesh_pos = mesh["mesh_pos"][0]
        
        solving_params = mesh["solving_params"]
        
        unsteady_coefficent = solving_params["unsteady"]

        continuity_eq_coefficent = solving_params["continuity"]

        convection_coefficent = solving_params["convection"]

        grad_p_coefficent = solving_params["grad_p"] / rho if rho != 0 else 0.0

        diffusion_coefficent = (
            (mu) if 0 == convection_coefficent else # convection_coefficent=0 means poisson equation
            (mu / (rho * max_velocity)) if (params.dimless and rho != 0 and max_velocity != 0) else
            (mu / rho) if rho != 0 else mu
        ) # 1/Re

        if max_velocity==0:
            source_term = 0.0
        else:
            source_term = source / max_velocity if params.dimless else source

        dt_cell = dt * max_velocity if params.dimless else dt
        
        theta_PDE = torch.tensor(
            [
                unsteady_coefficent,
                continuity_eq_coefficent,
                convection_coefficent,
                grad_p_coefficent,
                diffusion_coefficent,
                source_term,
                U_in[0].item(),
                U_in[1].item(),
                mesh["Re"]/100,
            ],
            device=mesh_pos.device,
            dtype=torch.float32,
        ).view(1,-1)
        mesh["theta_PDE"] = theta_PDE
        
        mesh["dt_graph"] = torch.tensor(
            [
                dt_cell
            ],
            device=mesh_pos.device,
            dtype=torch.float32,
        ).view(1,-1)
        
        mesh["sigma"] = torch.from_numpy(np.array(mesh["sigma"])).view(1,-1)
        
        if params.dimless:
            mesh["uvp_dim"] = torch.tensor(
                [[[max_velocity, max_velocity, (max_velocity**2)]]],
                device=mesh_pos.device,
                dtype=torch.float32,
            ).view(1,-1)

        else:
            mesh["uvp_dim"] = (
                torch.tensor([1, 1, 1], device=mesh_pos.device, dtype=torch.float32)
                .view(1,-1)
            )

        return mesh

    @staticmethod
    def makedimless(
        mesh, params, case_name=None, theta_PDE_list=None
    ):
        (
            mean_u,
            rho,
            mu,
            source,
            aoa,
            dt,
            L,
        ) = CFDdatasetBase.select_PDE_coef(theta_PDE_list)
        
        # L = torch.maximum(
        #     torch.tensor(L, dtype=torch.float32),
        #     CFDdatasetBase.calc_charactisc_length(mesh)
        # )
        inlet_type=mesh["inlet_type"]
        if inlet_type=="parabolic":
            max_u = mean_u*1.5

        else:
            max_u=mean_u
        mesh["mean_u"] = torch.tensor(mean_u, dtype=torch.float32)
        mesh["rho"] = torch.tensor(rho, dtype=torch.float32)
        mesh["mu"] = torch.tensor(mu, dtype=torch.float32)
        mesh["source"] = torch.tensor(source, dtype=torch.float32)
        mesh["aoa"] = torch.tensor(aoa, dtype=torch.float32)
        mesh["L"] = torch.tensor(L, dtype=torch.float32)
        mesh["Re"] = torch.tensor(rho * mean_u * L, dtype=torch.float32) / \
            mu if mu!=0 else torch.tensor(0, dtype=torch.float32)
            
        # 注意这里乘以了雷诺数倒数，则应注意在BC.JSON中给定的dt值，最好是1
        # mesh["dt"] = torch.tensor(dt, dtype=torch.float32)*(1./mesh["Re"])
        mesh["dt"] = torch.tensor(dt, dtype=torch.float32)
        
        
            
         
        mesh = CFDdatasetBase.set_PDE_theta(
            mesh, params, max_u, rho, mu, source, aoa, dt, L
        )

        return mesh, max_u


    @staticmethod
    def cal_node_centered_element_area(mesh):
        cells_area = mesh["cells_area"][0]
        cells_node = mesh["cells_node"][0].to(torch.long)
        cells_index = mesh["cells_index"][0].to(torch.long)
        
        node_area = calc_node_centered_with_cell_attr(cell_attr=cells_area, 
                                          cells_node=cells_node, 
                                          cells_index=cells_index, 
                                          reduce="mean", 
                                          map=True)
        
        mesh["node_area"] = node_area.unsqueeze(0)
        
        return mesh
    
    @staticmethod
    def normalize_coords(coords):
        """
        将二维坐标张量的 x 和 y 分量归一化到 [-1, 1] 范围内。

        参数:
        coords (torch.Tensor): 维度为 (N, 2) 的张量，表示 N 个节点的二维坐标。

        返回:
        torch.Tensor: 归一化后的坐标张量，形状为 (N, 2)。
        """
        
        de_mean = coords - coords.mean(dim=0,keepdim=True)
        
        # 获取每个维度的最小值和最大值
        min_vals, _ = de_mean.min(dim=0)
        max_vals, _ = de_mean.max(dim=0)
        range = torch.maximum(max_vals.abs(),min_vals.abs())
        
        # 归一化到 [0, 1] 范围内
        normalized = de_mean / range.unsqueeze(0)

        return normalized

    @staticmethod
    def To_Cartesian(mesh, resultion:tuple):

        if "grids" not in mesh.keys():
            
            L,K = resultion
            mesh_pos = mesh["mesh_pos"][0].to(torch.float32)
            
            xmax = torch.max(mesh_pos[:,0])
            xmin = torch.min(mesh_pos[:,0])
            ymin = torch.min(mesh_pos[:,1])
            ymax = torch.max(mesh_pos[:,1])

            grid_y, grid_x= torch.meshgrid(torch.linspace(ymin, ymax, L), 
                                           torch.linspace(xmin, xmax, K),
                                           indexing='ij')
            
            grid_points = torch.stack((grid_x, grid_y), dim=-1)

            mesh["grids"] = grid_points.unsqueeze(0).to(torch.float32)
            
            mesh["query"] = CFDdatasetBase.normalize_coords(mesh_pos.clone()).unsqueeze(0)
            
        return mesh
    
    @staticmethod
    def construct_stencil(
        mesh, 
        k_hop=2,
        BC_interal_neigbors=4,
        order=None,
    ):
        # Check if required keys exist before proceeding
        required_keys = ["mesh_pos_unique", "edge_index_unique", "node_type_unique", "face_node_x"]
        missing_keys = [key for key in required_keys if key not in mesh.keys()]
        
        if missing_keys:
            print(f"Warning: Missing keys in mesh for construct_stencil: {missing_keys}")
            print("Skipping stencil construction - mesh may not have been properly initialized.")
            return mesh
        
        if not "support_edge" in mesh.keys():
            mesh_pos = mesh["mesh_pos_unique"]
            face_node = mesh["edge_index_unique"].T.long().squeeze()
            node_type = mesh["node_type_unique"].long().squeeze()
            face_node_x = mesh["face_node_x"].long().to(torch.long)

            BC_mask = ((node_type==NodeType.WALL)|
                       (node_type==NodeType.INFLOW)|
                       (node_type==NodeType.OUTFLOW)|
                       (node_type==NodeType.PRESS_POINT)
                       ).squeeze()
            node_index = torch.arange(mesh_pos.shape[0])
            
            ''' including other boundary points '''
            # BC_edge_index = knn(x=mesh_pos, y=mesh_pos[BC_mask], k=BC_interal_neigbors)
            # filter_self_loop = (BC_edge_index[1]==node_index[BC_mask][BC_edge_index[0]]).squeeze()
            # BC_edge_index = torch.stack((BC_edge_index[1], node_index[BC_mask][BC_edge_index[0]]), dim=0)
            # BC_edge_index = BC_edge_index[:,~filter_self_loop]
            # BC_edge_index = torch.unique(BC_edge_index.sort(dim=0)[0],dim=1)
            ''' including other boundary points '''
            
            ''' exclude other boundary points '''
            BC_edge_index = knn(x=mesh_pos[~BC_mask], y=mesh_pos[BC_mask], k=BC_interal_neigbors)
            BC_edge_index = torch.stack((node_index[~BC_mask][BC_edge_index[1]], node_index[BC_mask][BC_edge_index[0]]), dim=0)
            BC_edge_index = torch.unique(BC_edge_index.sort(dim=0)[0],dim=1)
            ''' exclude other boundary points '''
            
            edge_index_ext1 = []
            for k in range(1, k_hop+1):
                edge_index_ext1.append(build_k_hop_edge_index(face_node, k=k))
            edge_index_ext1 = torch.cat((edge_index_ext1), dim=1)
        
            support_edge = torch.cat((
                face_node_x, 
                edge_index_ext1,
                BC_edge_index,
            ), dim=1)
            support_edge = torch.unique(support_edge.sort(dim=0)[0],dim=1)

            ''' 检查模板并绘制'度'的分布 '''
            in_degree = pyg_utils.degree(support_edge[1], num_nodes=mesh_pos.shape[0])
            out_degree = pyg_utils.degree(support_edge[0], num_nodes=mesh_pos.shape[0])
            node_degree = in_degree + out_degree
            
            if order=="1st":
                fix_mask = (node_degree <= 4).squeeze()
                find_extra_nb = knn(x=mesh_pos[~fix_mask], y=mesh_pos[fix_mask], k=6)
                ext_edge_index = torch.stack((node_index[~fix_mask][find_extra_nb[1]], node_index[fix_mask][find_extra_nb[0]]), dim=0)
                support_edge = torch.cat((support_edge, ext_edge_index), dim=1)
                
            elif order=="2nd":
                fix_mask = (node_degree <= 4).squeeze()
                find_extra_nb = knn(x=mesh_pos[~fix_mask], y=mesh_pos[fix_mask], k=8)
                ext_edge_index = torch.stack((node_index[~fix_mask][find_extra_nb[1]], node_index[fix_mask][find_extra_nb[0]]), dim=0)
                support_edge = torch.cat((support_edge, ext_edge_index), dim=1)
                
            elif order=="3rd":
                fix_mask = (node_degree <= 6).squeeze()
                find_extra_nb = knn(x=mesh_pos[~fix_mask], y=mesh_pos[fix_mask], k=13)
                ext_edge_index = torch.stack((node_index[~fix_mask][find_extra_nb[1]], node_index[fix_mask][find_extra_nb[0]]), dim=0)
                support_edge = torch.cat((support_edge, ext_edge_index), dim=1)
                
            elif order=="4th":
                fix_mask = (node_degree <= 8).squeeze()
                find_extra_nb = knn(x=mesh_pos[~fix_mask], y=mesh_pos[fix_mask], k=21)
                ext_edge_index = torch.stack((node_index[~fix_mask][find_extra_nb[1]], node_index[fix_mask][find_extra_nb[0]]), dim=0)
                support_edge = torch.cat((support_edge, ext_edge_index), dim=1)
 
            in_degree = pyg_utils.degree(support_edge[1], num_nodes=mesh_pos.shape[0])
            out_degree = pyg_utils.degree(support_edge[0], num_nodes=mesh_pos.shape[0])
            node_degree = in_degree + out_degree
            print("Degree max, mean ,min:", node_degree.max(), node_degree.mean(), node_degree.min())
            
            # write to file
            mesh["node_degree"] = node_degree
            # z_pos = torch.zeros(
            #     (mesh_pos.shape[0], 1), device=mesh_pos.device, dtype=mesh_pos.dtype
            # )
            # data_to_vtk = {
            #     "node|pos": torch.cat((mesh_pos, z_pos), dim=1).cpu().numpy(),
            #     "node|node_degree":node_degree.cpu().numpy(),
            #     "node|in_degree":in_degree.cpu().numpy(),
            #     "node|out_degree":out_degree.cpu().numpy(),
            #     "cells_node": Delaunay(mesh_pos.cpu().numpy()).simplices,
            # }
            # write_to_vtk(
            #     data_to_vtk,
            #     f"Logger/Grad_test/{mesh['case_name']}_stencil_.vtu",
            # )
            ''' 检查模板并绘制度分布 '''
            
            mesh["support_edge"] = support_edge
            
        return mesh
    
    @staticmethod
    def transform_mesh(
        mesh, 
        params=None
    ):
        
        theta_PDE_list = mesh["theta_PDE_list"]
        case_name = mesh["case_name"]
        
        mesh, max_u = CFDdatasetBase.makedimless(
            mesh,
            theta_PDE_list=theta_PDE_list,
            case_name=case_name,
            params=params,
        )
        
        mesh = CFDdatasetBase.construct_stencil(
            mesh, 
            k_hop=mesh["stencil|khops"], 
            BC_interal_neigbors=mesh["stencil|BC_extra_points"],
            order="2nd",
        )


        
        mesh, init_uvp_node = CFDdatasetBase.init_env(
            mesh,        
            max_u=max_u,
            dimless=params.dimless,
        )

        # start to generate boundary zone
        
        if mesh["Obstacle"]==1:
            boundary_zone = generate_boundary_zone(
                dataset=mesh,
                rho=mesh["rho"].item(),
                mu=mesh["mu"].item(),
                dt=mesh["dt"].item(),
            )
            mesh["boundary_zone"] = boundary_zone

        return mesh, init_uvp_node

class H5CFDdataset(Dataset):
    def __init__(self, params, file_list):
        super().__init__()

        self.file_list = file_list
        self.params = params
        
    def __getitem__(self, index):
        path = self.file_list[index]
        file_dir = os.path.dirname(path)
        case_name = os.path.basename(file_dir)
        h5_file = h5py.File(path, "r")

        try:
            BC_file = json.load(open(f"{file_dir}/BC.json", "r"))
        except:
            raise ValueError(f"BC.json is not found in the {path}")
        
        key_list = list(h5_file.keys())
        mesh_handle = h5_file[key_list[0]]
        mesh = {"case_name":case_name} # set mesh name
        
        # convert to tensors
        for key in mesh_handle.keys():
            mesh[key] = torch.from_numpy(mesh_handle[key][()])

        # import all BC.json item into mesh dict
        for key, value in BC_file.items():
            mesh[key] = value
        mesh["stencil|BC_extra_points"] = 4
        mesh["stencil|khops"] = 2

        # generate all valid theta_PDE combinations
        theta_PDE = BC_file["solving_params"]
        theta_PDE_list = (
                    get_param.generate_combinations(
                        U_range=theta_PDE["inlet"],
                        rho_range=theta_PDE["rho"],
                        mu_range=theta_PDE["mu"],
                        source_range=theta_PDE["source"],
                        aoa_range=theta_PDE["aoa"],
                        dt=theta_PDE["dt"],
                        L=theta_PDE["L"],
                    )
                )
        mesh["theta_PDE_list"] = theta_PDE_list
        
   
        mesh_transformed, init_uvp_node = CFDdatasetBase.transform_mesh(
            mesh, 
            self.params
        )

        # return to CPU!
        return mesh_transformed, init_uvp_node

    def __len__(self):
        return len(self.file_list)

