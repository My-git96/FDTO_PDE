"""Utility functions for reading the datasets."""

import sys
import os
file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(file_dir)


from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data



import torch

from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.loader import DataLoader as torch_geometric_DataLoader
from torch.utils.data import Sampler
import datetime

from utils.utilities import export_uvp_to_tecplot, export_u_to_tecplot

from dataset.Load_mesh import H5CFDdataset

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

            sample_mask = global_idx
            mesh = self.meta_pool[i]
            export_uvp_to_tecplot(
                mesh=mesh,
                uvp=q.data[:,0:3],
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
