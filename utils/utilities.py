import numpy as np
from torch_geometric.data import Data
from enum import IntEnum
import torch
from torch_scatter import scatter,scatter_add,scatter_mean
import torch.nn.functional as F
import os
import math
from Extract_mesh.write_tec import write_u_tecplotzone,write_uvp_tecplotzone


class NodeType(IntEnum):
    CUT1TO1 = -1
    NORMAL = 0
    WALL = 2
    SYMMETRY = 3
    FARFIELD = 4
    INFLOW = 5
    OUTFLOW = 6
    PRESS_POINT = 7


def generate_boundary_zone(
    dataset=None, rho=None, mu=None, dt=None
):
 
    mesh_pos = dataset["mesh_pos"].to(torch.float32)
    node_type = dataset["node_type"]
    
    boundary_zone = {"name": "OBSTACLE", "rho": rho, "mu": mu, "dt": dt}
    boundary_zone["zonename"] = "OBSTICALE_BOUNDARY"
    node_topwall = torch.max(mesh_pos[:,1])
    node_bottomwall = torch.min(mesh_pos[:,1])
    node_outlet = torch.max(mesh_pos[:,0])
    node_inlet = torch.min(mesh_pos[:,0])

    MasknodeT = torch.full((mesh_pos.shape[0],1),True).squeeze(1) 
    MasknodeF = torch.logical_not(MasknodeT) 


    
    mask_node_boundary = torch.where(((((node_type==NodeType.WALL)))&(mesh_pos[:,1:2]<node_topwall).squeeze(1) &(mesh_pos[:,1:2]>node_bottomwall).squeeze(1) &(mesh_pos[:,0:1]>node_inlet).squeeze(1) &(mesh_pos[:,0:1]<node_outlet).squeeze(1) ),MasknodeT,MasknodeF)   

    bc_pos = mesh_pos[mask_node_boundary]

    num_bc_nodes = mask_node_boundary.sum()

    col = torch.arange(num_bc_nodes, dtype=torch.long)
    row = col.clone()+1
    row[-1] = 0

    boundary_zone["face_node"] = torch.stack((row,col),dim=-1)

    boundary_zone["mesh_pos"] = bc_pos.unsqueeze(0)

    boundary_zone["mask_node_boundary"] = mask_node_boundary

    return boundary_zone

def moments_order(
    order="1nd",
    mesh_pos_diff_on_edge=None,
    indegree_node_index=None,
    dim_size=None,
):
    '''
    mesh_pos_diff_on_edge:[2*E, 2]
    indegree_node_index:[N]
    '''
    
    if order=="1st":
        od=2
        displacement = mesh_pos_diff_on_edge.unsqueeze(2)
        
    elif order=="2nd":
        od=3
        displacement = torch.cat(
            (
                mesh_pos_diff_on_edge,
                0.5 * (mesh_pos_diff_on_edge**2),
                mesh_pos_diff_on_edge[:, 0:1] * mesh_pos_diff_on_edge[:, 1:2],
            ),
            dim=-1,
        ).unsqueeze(2)
        
    elif order=="3rd":
        od=4
        displacement = torch.cat(
            (
                mesh_pos_diff_on_edge,
                0.5 * (mesh_pos_diff_on_edge**2),
                mesh_pos_diff_on_edge[:, 0:1] * mesh_pos_diff_on_edge[:, 1:2],
                (1 / 6) * (mesh_pos_diff_on_edge**3),
                0.5 * (mesh_pos_diff_on_edge[:, 0:1] ** 2) * mesh_pos_diff_on_edge[:, 1:2],
                0.5 * (mesh_pos_diff_on_edge[:, 1:2] ** 2) * mesh_pos_diff_on_edge[:, 0:1],
            ),
            dim=-1,
        ).unsqueeze(2)
        
    elif order=="4th":
        od=5
        displacement = torch.cat(
            (
                mesh_pos_diff_on_edge,
                0.5 * (mesh_pos_diff_on_edge**2),
                mesh_pos_diff_on_edge[:, 0:1] * mesh_pos_diff_on_edge[:, 1:2],
                (1 / 6) * (mesh_pos_diff_on_edge**3),
                0.5 * (mesh_pos_diff_on_edge[:, 0:1] ** 2) * mesh_pos_diff_on_edge[:, 1:2],
                0.5 * (mesh_pos_diff_on_edge[:, 1:2] ** 2) * mesh_pos_diff_on_edge[:, 0:1],
                (1 / 24) * (mesh_pos_diff_on_edge[:, 0:1] ** 4),
                (1 / 6)
                * (mesh_pos_diff_on_edge[:, 0:1] ** 3)
                * mesh_pos_diff_on_edge[:, 1:2],
                (1 / 4)
                * (mesh_pos_diff_on_edge[:, 0:1] ** 2)
                * (mesh_pos_diff_on_edge[:, 1:2] ** 2),
                (1 / 6)
                * (mesh_pos_diff_on_edge[:, 0:1])
                * (mesh_pos_diff_on_edge[:, 1:2] ** 3),
                (1 / 24) * (mesh_pos_diff_on_edge[:, 1:2] ** 4),
            ),
            dim=-1,
        ).unsqueeze(2)
    else:
        raise NotImplementedError(f"{order} Order not implemented")
    
    displacement_T = displacement.transpose(1, 2)

    weight_node_to_node = (1 / torch.norm(mesh_pos_diff_on_edge, dim=1, keepdim=True)**od).unsqueeze(2)
        
    left_on_edge = torch.matmul(
        displacement * weight_node_to_node,
        displacement_T,
    )

    A_node_to_node = scatter_add(
        left_on_edge, indegree_node_index, dim=0, dim_size=dim_size
    ) # [N, x, x], x is depend on order
    
    B_node_to_node = weight_node_to_node * displacement
    # [2*E, x]
    
    return A_node_to_node, B_node_to_node

def compute_normal_matrix(
    order="1st",
    mesh_pos=None,
    outdegree=None,
    indegree=None,
    dual_edge=True, # 输入的in/outdegree是否是双向的
):
    """
    Computes the normal matrices A and B for node-based weighted least squares (WLSQ)
    gradient reconstruction.

    Parameters:
    - order (str): The order of the reconstruction ('1st', '2nd', '3rd', or '4th').
    - mesh_pos (torch.Tensor): Tensor of shape [N, D] containing the positions of the mesh nodes.
    - outdegree (torch.Tensor): Tensor containing the indices of source nodes (outgoing edges).
    - indegree (torch.Tensor): Tensor containing the indices of target nodes (incoming edges).
    - dual_edge (bool): If True, the provided outdegree and indegree represent bidirectional edges.
                        If False, the function constructs bidirectional edges by concatenating
                        the input edges.

    Returns:
    - A_node_to_node (torch.Tensor): Normal matrix A for each node.
    - B_node_to_node (torch.Tensor): Matrix B for each node.
    """
    
    if dual_edge:
        outdegree_node_index, indegree_node_index = outdegree, indegree
    else:
        outdegree_node_index = torch.cat((outdegree, indegree), dim=0)
        indegree_node_index = torch.cat((indegree, outdegree), dim=0)
        
    mesh_pos_diff_on_edge = mesh_pos[outdegree_node_index] - mesh_pos[indegree_node_index]

    (A_node_to_node, B_node_to_node) = moments_order(
        order=order,
        mesh_pos_diff_on_edge=mesh_pos_diff_on_edge,
        indegree_node_index=indegree_node_index,
        dim_size=mesh_pos.shape[0],
    )

    return (A_node_to_node, B_node_to_node)

def node_based_WLSQ(
    phi_node=None,
    edge_index=None,
    mesh_pos=None,
    dual_edge=True, # 输入的edge_index是否是双向的
    order=None,
    precompute_Moments: list = None,

):
    '''
    B right-hand sides in precompute_Moments must be SINGLE-WAY
    on edge
    '''
    # edge_index = knn_graph(mesh_pos, k=9, loop=False)
    if (order is None) or (order not in ["1st", "2nd", "3rd", "4th"]):
        raise ValueError("order must be specified in [\"1st\", \"2nd\", \"3rd\", \"4th\"]")
    
    if dual_edge:
        outdegree_node_index, indegree_node_index = edge_index[0], edge_index[1]
    else:
        outdegree_node_index = torch.cat((edge_index[0], edge_index[1]), dim=0)
        indegree_node_index = torch.cat((edge_index[1], edge_index[0]), dim=0)

    if precompute_Moments is None:

        """node to node contribution"""
        (A_node_to_node, two_way_B_node_to_node) = compute_normal_matrix(
            order=order,
            mesh_pos=mesh_pos,
            outdegree=outdegree_node_index,
            indegree=indegree_node_index,
            dual_edge=False if dual_edge else True,
        )
        """node to node contribution"""

        phi_diff_on_edge = two_way_B_node_to_node * (
            (phi_node[outdegree_node_index] - phi_node[indegree_node_index]).unsqueeze(
                1
            )
        )

        B_phi_node_to_node = scatter_add(
            phi_diff_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
        )

    else:
        """use precomputed moments"""
        A_node_to_node, Oneway_B_node_to_node = precompute_Moments

        half_dim = Oneway_B_node_to_node.shape[0]
        
        two_way_B_node_to_node = torch.cat(
            (Oneway_B_node_to_node, Oneway_B_node_to_node), dim=0
        )
        
        # 大于1阶的奇数阶项需要取负
        two_way_B_node_to_node[half_dim:,0:2]*= -1
        od = int(order[0])
        
        if od >=3 :
            two_way_B_node_to_node[half_dim:,5:9]*= -1
            
        phi_diff_on_edge = two_way_B_node_to_node * (
            (phi_node[outdegree_node_index] - phi_node[indegree_node_index]).unsqueeze(
                1
            )
        )

        B_phi_node_to_node = scatter_add(
            phi_diff_on_edge,
            indegree_node_index,
            dim=0,
            dim_size=mesh_pos.shape[0],
        )
        
    # 行归一化
    row_norms = torch.norm(A_node_to_node, p=2, dim=2, keepdim=True)
    A_normalized = A_node_to_node / (row_norms + 1e-8)
    B_normalized = B_phi_node_to_node / (row_norms + 1e-8)
    
    # 添加正则化以避免奇异矩阵（当某些节点没有边时）
    lambda_reg = 1e-5  # 正则化参数
    I = torch.eye(A_normalized.shape[-1], device=A_normalized.device).unsqueeze(0)
    A_normalized = A_normalized + lambda_reg * I
    
    # # 列归一化
    # col_norms = torch.norm(A_normalized, p=2, dim=1, keepdim=True)
    # A_normalized = A_normalized / (col_norms + 1e-8)
    # B_normalized = B_normalized * col_norms
    
    """ first method"""
    # nabla_phi_node_lst = torch.linalg.lstsq(
    #     A_normalized, B_normalized
    # ).solution.transpose(1, 2)

    """ second method"""
    # nabla_phi_node_lst = torch.matmul(A_inv_node_to_node_x,B_phi_node_to_node_x)

    """ third method"""
    nabla_phi_node_lst = torch.linalg.solve(
        A_normalized, B_normalized
    ).transpose(1, 2)

    """ fourth method"""
    # nabla_phi_node_lst = torch.matmul(R_inv_Q_t,B_phi_node_to_node_x)

    return nabla_phi_node_lst

def calc_cell_centered_with_node_attr(
    node_attr, cells_node, cells_index, reduce="mean", map=True
):
    if cells_node.shape != cells_index.shape:
        raise ValueError("wrong cells_node/cells_index dim")

    if len(cells_node.shape) > 1:
        cells_node = cells_node.view(-1)

    if len(cells_index.shape) > 1:
        cells_index = cells_index.view(-1)

    if map:
        mapped_node_attr = node_attr[cells_node]
    else:
        mapped_node_attr = node_attr

    cell_attr = scatter(src=mapped_node_attr, index=cells_index, dim=0, reduce=reduce)

    return cell_attr


def calc_node_centered_with_cell_attr(
    cell_attr, cells_node, cells_index, reduce="mean", map=True
):
    if cells_node.shape != cells_index.shape:
        raise ValueError(f"wrong cells_node/cells_index dim ")

    if len(cells_node.shape) > 1:
        cells_node = cells_node.view(-1)

    if len(cells_index.shape) > 1:
        cells_index = cells_index.view(-1)

    if map:
        maped_cell_attr = cell_attr[cells_index]
    else:
        maped_cell_attr = cell_attr

    cell_attr = scatter(src=maped_cell_attr, index=cells_node, dim=0, reduce=reduce)

    return cell_attr


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def decompose_and_trans_node_attr_to_cell_attr_graph(
    graph, has_changed_node_attr_to_cell_attr
):
    # graph: torch_geometric.data.data.Data
    # TODO: make it more robust
    x, edge_index, edge_attr, face, global_attr, mask_cell_interior = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    for key in graph.keys():
        if key == "x":
            x = graph.x  # avoid exception
        elif key == "edge_index":
            edge_index = graph.edge_index
        elif key == "edge_attr":
            edge_attr = graph.edge_attr
        elif key == "global_attr":
            global_attr = graph.global_attr
        elif key == "face":
            face = graph.face
        elif key == "mask_cell_interior":
            mask_cell_interior = graph.mask_cell_interior
        else:
            pass

    return (x, edge_index, edge_attr, face, global_attr, mask_cell_interior)


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def copy_geometric_data(graph, has_changed_node_attr_to_cell_attr):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    (
        node_attr,
        edge_index,
        edge_attr,
        face,
        global_attr,
        mask_cell_interior,
    ) = decompose_and_trans_node_attr_to_cell_attr_graph(
        graph, has_changed_node_attr_to_cell_attr
    )

    ret = Data(
        x=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
        face=face,
        mask_cell_interior=mask_cell_interior,
    )

    ret.global_attr = global_attr

    return ret


def shuffle_np(array):
    array_t = array.copy()
    np.random.shuffle(array_t)
    return array_t





def extract_cylinder_boundary(
    dataset,
    aoa,
    dataset_all,
    graph_node: Data,

    rho=None,
    mu=None,
    dt=None,
):
    write_zone = {"zonename": "OBSTACLE", "rho": rho, "mu": mu, "dt": dt}
    node_type = graph_node.node_type
    mesh_pos = graph_node.pos
    node_topwall = torch.max(mesh_pos[:,1])
    node_bottomwall = torch.min(mesh_pos[:,1])
    node_outlet = torch.max(mesh_pos[:,0])
    node_inlet = torch.min(mesh_pos[:,0])

    MasknodeT = torch.full((mesh_pos.shape[0],1),True).cuda()
    MasknodeF = torch.logical_not(MasknodeT).cuda()
    mask_node_boundary = torch.where(((((node_type==NodeType.WALL)))&(mesh_pos[:,1:2]<node_topwall)&(mesh_pos[:,1:2]>node_bottomwall)&(mesh_pos[:,0:1]>node_inlet)&(mesh_pos[:,0:1]<node_outlet)),MasknodeT,MasknodeF).squeeze(1)
    
    write_zone["mesh_pos"] = (
        graph_node.pos[mask_node_boundary, 0:2].to("cpu").unsqueeze(0).numpy()
    )

    write_zone["velocity"] = (
        dataset["predicted_node_uvp"][:, mask_node_boundary, 0:2].to("cpu").numpy()
    )

    write_zone["pressure"] = (
        dataset["predicted_node_uvp"][:, mask_node_boundary, 2:3].to("cpu").numpy()
    )


 



    write_zone["mask_node_boundary"] = mask_node_boundary



    num_bc_nodes = mask_node_boundary.sum()

    col = torch.arange(num_bc_nodes, dtype=torch.long)
    row = col.clone()+1
    row[-1] = 0
    for t in range(write_zone["velocity"].shape[0]):
        alpha =  aoa[t]   
    
        wall_p = dataset["predicted_node_uvp"][t,mask_node_boundary,2:3]

        boundary_mesh_pos = mesh_pos[mask_node_boundary,:]


        grad_uv = node_based_WLSQ(
            phi_node=dataset["predicted_node_uvp"][t,:,0:2].cuda(),
            edge_index=dataset_all["support_edge"].cuda(),
            mesh_pos=dataset_all["mesh_pos"][0].cuda().float(),
            dual_edge=False,
            order="2nd",
            precompute_Moments=None,
        )
        grad_wall = grad_uv[mask_node_boundary,:,0:2]
        
        grad_wall_face = (grad_wall[row]+grad_wall[col])/2

        wall_face_length = (boundary_mesh_pos[row]- boundary_mesh_pos[col]).norm(dim=1,keepdim=True)

        wall_tan = -(boundary_mesh_pos[row]- boundary_mesh_pos[col])/wall_face_length

        wall_nor = torch.cat([-wall_tan[:,1:2], wall_tan[:,0:1]], dim=1)

        u_xy = grad_wall_face[:,0,:]
        v_xy = grad_wall_face[:,1,:]

        u_r = (u_xy * wall_nor).sum(1, keepdim=True)

        v_r = (v_xy * wall_nor).sum(1, keepdim=True)

        uth_r = u_r * wall_tan[:,0:1] + v_r *wall_tan[:,1:2]  

        wall_face_p = (wall_p[row]+wall_p[col])/2

        Cp = wall_face_p/(0.5*rho*dataset_all["inf_u"]**2) 

        Cf = -uth_r*mu/(0.5*rho*dataset_all["inf_u"]**2)

        Force_p = Cp*wall_face_length*wall_nor

        Force_f = Cf*wall_face_length*wall_tan

      

        Cl_p = (torch.cos(alpha)*Force_p[:,1:2] - torch.sin(alpha)*Force_p[:,0:1]).sum()
        Cd_p = (torch.sin(alpha)*Force_p[:,1:2] + torch.cos(alpha)*Force_p[:,0:1]).sum()

        Cl_f = (torch.cos(alpha)*Force_f[:,1:2] - torch.sin(alpha)*Force_f[:,0:1]).sum()
        Cd_f = (torch.sin(alpha)*Force_f[:,1:2] + torch.cos(alpha)*Force_f[:,0:1]).sum()

        L = 0.05

        Cl = (Cl_p + Cl_f)/L
        Cd = (Cd_p + Cd_f)/L


        dataset["CL_list"].append(Cl.to("cpu").numpy()) 
        dataset["CD_list"].append(Cd.to("cpu").numpy())    
    

    write_zone["face"] = torch.stack((row,col),dim=-1).unsqueeze(0).cpu().numpy()
    write_zone["mesh_pos"]  = dataset["mesh_pos"][:, mask_node_boundary, 0:2].to("cpu").numpy()
    
    write_zone["zonename"] = "OBSTICALE_BOUNDARY"

    overset_nodetype = np.full((write_zone["mesh_pos"].shape[1], 1), 1)
    write_zone["overset_nodetype"]  = overset_nodetype
    write_zone["data_packing_type"]  = "node"
    return write_zone



def extract_cylinder_boundary_only_training(
    dataset=None, rho=None, mu=None, dt=None
):
    node_type = dataset["node_type"][0]
    face_node = dataset["face"][0].long()
    if face_node.shape[0]>face_node.shape[1]:
        face_node = face_node.T
    mesh_pos = dataset["mesh_pos"][0]
    boundary_zone = {"name":"OBSTACLE",
                    "rho":rho,
                    "mu":mu,
                    "dt":dt} 



    node_topwall = torch.max(mesh_pos[:,1])
    node_bottomwall = torch.min(mesh_pos[:,1])
    node_outlet = torch.max(mesh_pos[:,0])
    node_inlet = torch.min(mesh_pos[:,0])

    MasknodeT = torch.full((mesh_pos.shape[0],1),True)
    MasknodeF = torch.logical_not(MasknodeT)


    
    mask_node_boundary = torch.where(((((node_type==NodeType.WALL)))&(mesh_pos[:,1:2]<node_topwall)&(mesh_pos[:,1:2]>node_bottomwall)&(mesh_pos[:,0:1]>node_inlet)&(mesh_pos[:,0:1]<node_outlet)),MasknodeT,MasknodeF).squeeze(1)
    




    boundary_zone["mask_node_boundary"] = mask_node_boundary


    num_bc_nodes = mask_node_boundary.sum()



    


    col = torch.arange(num_bc_nodes, dtype=torch.long)
    row = col.clone()+1
    row[-1] = 0
    
    boundary_mesh_pos = mesh_pos[mask_node_boundary,:]

    wall_p = dataset["pressure_on_node"][0,mask_node_boundary,:]


    grad_uv = node_based_WLSQ(
        phi_node=dataset["velocity_on_node"][0],
        edge_index=dataset["support_edge"],
        mesh_pos=dataset["mesh_pos"][0],
        dual_edge=False,
        order="2nd",
        precompute_Moments=None,
    )
    grad_wall = grad_uv[mask_node_boundary,:,0:2]
    
    grad_wall_face = (grad_wall[row]+grad_wall[col])/2

    wall_face_length = (boundary_mesh_pos[row]- boundary_mesh_pos[col]).norm(dim=1,keepdim=True)

    wall_tan = -(boundary_mesh_pos[row]- boundary_mesh_pos[col])/wall_face_length

    wall_nor = torch.cat([-wall_tan[:,1:2], wall_tan[:,0:1]], dim=1)

    u_xy = grad_wall_face[:,0,:]
    v_xy = grad_wall_face[:,1,:]

    u_r = (u_xy * wall_nor).sum(1, keepdim=True)

    v_r = (v_xy * wall_nor).sum(1, keepdim=True)

    uth_r = u_r * wall_tan[:,0:1] + v_r *wall_tan[:,1:2]  

    wall_face_p = (wall_p[row]+wall_p[col])/2

    Cp = wall_face_p/(0.5*rho*dataset["inf_u"]**2) 

    Cf = -uth_r*mu/(0.5*rho*dataset["inf_u"]**2)

    Force_p = Cp*wall_face_length*wall_nor

    Force_f = Cf*wall_face_length*wall_tan

    alpha = dataset["alpha"]

    Cl_p = (torch.cos(alpha)*Force_p[:,1:2] - torch.sin(alpha)*Force_p[:,0:1]).sum()
    Cd_p = (torch.sin(alpha)*Force_p[:,1:2] + torch.cos(alpha)*Force_p[:,0:1]).sum()

    Cl_f = (torch.cos(alpha)*Force_f[:,1:2] - torch.sin(alpha)*Force_f[:,0:1]).sum()
    Cd_f = (torch.sin(alpha)*Force_f[:,1:2] + torch.cos(alpha)*Force_f[:,0:1]).sum()

    L = 0.05

    Cl = (Cl_p + Cl_f)/L
    Cd = (Cd_p + Cd_f)/L


    dataset["Cl"] = Cl
    dataset["Cd"] = Cd

    boundary_zone["face"] = torch.stack((row,col),dim=-1).unsqueeze(0).numpy()
    boundary_zone["mesh_pos"] = boundary_mesh_pos.unsqueeze(0).numpy()
    boundary_zone["zonename"] = "OBSTICALE_BOUNDARY"

    overset_nodetype = np.full((boundary_zone["mesh_pos"].shape[1], 1), 1)

    return boundary_zone

def export_uvp_to_tecplot(mesh, uvp_err, datalocation="node", file_name=None, physical_time=None, time_step=None,
                state_save_dir=None, device=None, plot_count=None,to_export = True):
        """
        导出uvp(node)数据到Tecplot DAT文件
        """
        case_name = mesh["case_name"]
        dt = mesh["dt"].squeeze().item()
        source = mesh["source"].squeeze().item()
        aoa = mesh["aoa"].squeeze().item()
        to_numpy = lambda x: x.cpu().numpy() if x.is_cuda else x.numpy()
        write_dataset = []
        interior_zone_numpy = {}
        boundary_zone_numpy = {}

        for k, v in mesh.items():
            if isinstance(v, torch.Tensor):
                interior_zone_numpy[k] = to_numpy(v)
            else:
                interior_zone_numpy[k] = v   
        uv_uns = uvp_err[:,0:2].unsqueeze(0).float()
        p_uns = uvp_err[:,2:3].unsqueeze(0).float()
        if uvp_err.shape[1] > 3:
            uv_err = uvp_err[:,3:4].unsqueeze(0).float()
            interior_zone_numpy["uv_error"] = uv_err.numpy()
        if uvp_err.shape[1] > 4:
            p_err = uvp_err[:,4:5].unsqueeze(0).float()
            interior_zone_numpy["p_error"] = p_err.numpy()
        interior_zone_numpy["velocity"] = uv_uns.numpy()
        interior_zone_numpy["pressure"] = p_uns.numpy()
        
        interior_zone_numpy["mesh_pos"] = mesh["mesh_pos_unique"].unsqueeze(0).float().numpy()
        interior_zone_numpy['cells'] = mesh['cells_node_unique']
        interior_zone_numpy['cells_index'] = mesh['cells_index']
        interior_zone_numpy['face_node'] = mesh["edge_index_unique"]  
        interior_zone_numpy['zonename'] = 'Fluid'
        interior_zone_numpy["data_packing_type"] = ["node"]
        write_dataset.append(interior_zone_numpy)              
        if "boundary_zone" in mesh:
    
            boundary_zone = mesh["boundary_zone"]
            for k, v in boundary_zone.items():
                if not isinstance(v, torch.Tensor):
                    continue
                else:
                    boundary_zone_numpy[k] = to_numpy(v)
            
            # mask_node_boundary是针对原始mesh的全节点的布尔mask（长度为mesh的总节点数）
            # 但uvp_err只包含当前batch/sample的节点（可能长度不同）
            mask_node_boundary = boundary_zone["mask_node_boundary"]
            
            # 如果mask长度与uvp_err不匹配，需要做映射
            if isinstance(mask_node_boundary, torch.Tensor):
                # mask_node_boundary是布尔mask，需要转为索引
                boundary_indices = torch.where(mask_node_boundary)[0]
                # 只保留在当前uvp_err范围内的边界节点索引
                valid_boundary_indices = boundary_indices[boundary_indices < uvp_err.shape[0]]
                
                if len(valid_boundary_indices) > 0:
                    boundary_zone_numpy["velocity"] = uv_uns[:, valid_boundary_indices, :].numpy()
                    boundary_zone_numpy["pressure"] = p_uns[:, valid_boundary_indices, :].numpy()
                    if uvp_err.shape[1] > 3:
                        uv_err = uvp_err[:,3:4].unsqueeze(0).float()
                        boundary_zone_numpy["uv_error"] = uv_err[:, valid_boundary_indices, :].numpy()
                    if uvp_err.shape[1] > 4:
                        p_err = uvp_err[:,4:5].unsqueeze(0).float()
                        boundary_zone_numpy["p_error"] = p_err[:, valid_boundary_indices, :].numpy()
                else:
                    print("[WARNING] No valid boundary indices found in uvp_err")
            else:
                # mask_node_boundary已经是索引数组，直接使用
                if len(mask_node_boundary) > 0 and mask_node_boundary.max() < uvp_err.shape[0]:
                    boundary_zone_numpy["velocity"] = uv_uns[:, mask_node_boundary, :].numpy()
                    boundary_zone_numpy["pressure"] = p_uns[:, mask_node_boundary, :].numpy()
                    if uvp_err.shape[1] > 3:
                        uv_err = uvp_err[:,3:4].unsqueeze(0).float()
                        boundary_zone_numpy["uv_error"] = uv_err[:, mask_node_boundary, :].numpy()
                    if uvp_err.shape[1] > 4:
                        p_err = uvp_err[:,4:5].unsqueeze(0).float()
                        boundary_zone_numpy["p_error"] = p_err[:, mask_node_boundary, :].numpy()
                        
            boundary_zone_numpy["data_packing_type"] = ["node"]
            boundary_zone_numpy['zonename'] = 'BOUNDARY'
            write_dataset.append(boundary_zone_numpy)

        try:
            Re=mesh["Re"].squeeze().item()
        except:
            Re=0
            Warning("No Re number in the mesh set to 0")
        
        # If Re is 0, try to compute it from rho, mean_u, and L
        if Re == 0:
            try:
                mean_u = mesh.get("mean_u", 1.0)
                if isinstance(mean_u, torch.Tensor):
                    mean_u = mean_u.item()
                L = mesh.get("L", 1.0)
                if isinstance(L, torch.Tensor):
                    L = L.item()
                mu_val = mesh.get("mu", 1.0)
                if isinstance(mu_val, torch.Tensor):
                    mu_val = mu_val.item()
                rho_val = mesh.get("rho", 1.0)
                if isinstance(rho_val, torch.Tensor):
                    rho_val = rho_val.item()
                
                if mu_val != 0:
                    Re = (rho_val * mean_u * L) / mu_val
            except:
                Re = 0

        if file_name is None:
            # 使用物理时间命名，类似OpenFOAM的时间目录风格
            if physical_time is not None:
                # 创建以物理时间命名的目录 (类似OpenFOAM的 0/, 0.01/, 0.02/ 等)
                time_dir = f"{state_save_dir}/t_{physical_time:.4f}s"
                os.makedirs(time_dir, exist_ok=True)
                saving_path = f"{time_dir}/t_{physical_time:.4f}s_{case_name}_Re={Re:.2f}_aoa={aoa:.2f}.dat"
            elif time_step is not None:
                # 使用时间步索引
                saving_dir = f"{state_save_dir}/step_{time_step:04d}"
                os.makedirs(saving_dir, exist_ok=True)
                saving_path = f"{saving_dir}/{case_name}_Re={Re:.2f}_dt={dt:.3f}_source={source:.2f}_aoa={aoa:.2f}.dat"
            else:
                # Default case: use state_save_dir if available, otherwise use current directory
                if state_save_dir is not None:
                    os.makedirs(state_save_dir, exist_ok=True)
                    saving_path = f"{state_save_dir}/{case_name}_Re={Re:.2f}_aoa={aoa:.2f}.dat"
                else:
                    saving_path = f"{case_name}_Re={Re:.2f}_aoa={aoa:.2f}.dat"
        else:
            saving_path = file_name

        if to_export:
            write_uvp_tecplotzone(
                filename=saving_path,
                datasets=write_dataset,
                time_step_length=1,
            )
        
        # 计算并保存升阻力系数
        if ("NACA" in case_name) or ("RAE" in case_name):
            compute_and_save_lift_drag_coefficients(
                mesh=mesh,
                uv_uns=uv_uns,
                p_uns=p_uns,
                case_name=case_name,
                Re=Re,
                aoa=aoa,
                time_step=time_step,
                physical_time=physical_time,
                dt=dt,
                state_save_dir=state_save_dir,
                device=device,
                plot_count=plot_count,
                saving_path=saving_path
            )
            #计算并保存表面压力
            export_surface_pressure(
                mesh=mesh,
                p_uns=p_uns,
                case_name=case_name,
                Re=Re,
                aoa=aoa,
                time_step=time_step,
                physical_time=physical_time,
                dt=dt,
                state_save_dir=state_save_dir,
                device=device,
                plot_count=plot_count,
                saving_path=saving_path)
        elif ("Cylinder" in case_name):
            compute_and_save_lift_drag_coefficients(
                mesh=mesh,
                uv_uns=uv_uns,
                p_uns=p_uns,
                case_name=case_name,
                Re=Re,
                aoa=aoa,
                time_step=time_step,
                physical_time=physical_time,
                dt=dt,
                state_save_dir=state_save_dir,
                device=device,
                plot_count=plot_count,
                saving_path=saving_path
            )
def export_u_to_tecplot(mesh, u, datalocation="node", file_name=None, physical_time=None, time_step=None,state_save_dir=None):
        """
        导出u和error(node)数据到Tecplot DAT文件
        """
        case_name = mesh["case_name"]
        dt = mesh["dt"].squeeze().item()
        source = mesh["source"].squeeze().item()
        aoa = mesh["aoa"].squeeze().item()
        to_numpy = lambda x: x.cpu().numpy() if x.is_cuda else x.numpy()
        write_dataset = []
        interior_zone_numpy = {}
        boundary_zone_numpy = {}

        for k, v in mesh.items():
            if isinstance(v, torch.Tensor):
                interior_zone_numpy[k] = to_numpy(v)
            else:
                interior_zone_numpy[k] = v   

        u_uns = u[:,0].unsqueeze(0).float()
        interior_zone_numpy["velocity"] = u_uns.numpy()
        interior_zone_numpy["mesh_pos"] = mesh["mesh_pos_unique"].unsqueeze(0).float().numpy()
        interior_zone_numpy['cells'] = mesh['cells_node_unique']
        interior_zone_numpy['cells_index'] = mesh['cells_index']
        interior_zone_numpy['face_node'] = mesh["edge_index_unique"]  
        interior_zone_numpy['zonename'] = 'Fluid'
        interior_zone_numpy["data_packing_type"] = ["node"]
        write_dataset.append(interior_zone_numpy)              
        if "boundary_zone" in mesh:
            boundary_zone = mesh["boundary_zone"]
            for k, v in boundary_zone.items():
                if not isinstance(v, torch.Tensor):
                    continue
                else:
                    boundary_zone_numpy[k] = to_numpy(v)
            mask_node_boundary = boundary_zone["mask_node_boundary"]
            boundary_zone_numpy["velocity"] = u_uns[:,mask_node_boundary,:].numpy()
            boundary_zone_numpy["data_packing_type"] = ["node"]
            boundary_zone_numpy['zonename'] = 'BOUNDARY'
            write_dataset.append(boundary_zone_numpy)

        try:
            Re=mesh["Re"].squeeze().item()
        except:
            Re=0
            Warning("No Re number in the mesh set to 0")
        
        # If Re is 0, try to compute it from rho, mean_u, and L
        if Re == 0:
            try:
                mean_u = mesh.get("mean_u", 1.0)
                if isinstance(mean_u, torch.Tensor):
                    mean_u = mean_u.item()
                L = mesh.get("L", 1.0)
                if isinstance(L, torch.Tensor):
                    L = L.item()
                mu_val = mesh.get("mu", 1.0)
                if isinstance(mu_val, torch.Tensor):
                    mu_val = mu_val.item()
                rho_val = mesh.get("rho", 1.0)
                if isinstance(rho_val, torch.Tensor):
                    rho_val = rho_val.item()
                
                if mu_val != 0:
                    Re = (rho_val * mean_u * L) / mu_val
            except:
                Re = 0


        if file_name is None:
            # 使用物理时间命名，类似OpenFOAM的时间目录风格
            if physical_time is not None:
                # 创建以物理时间命名的目录 (类似OpenFOAM的 0/, 0.01/, 0.02/ 等)
                time_dir = f"{state_save_dir}/t_{physical_time:.4f}s"
                os.makedirs(time_dir, exist_ok=True)
                saving_path = f"{time_dir}/t_{physical_time:.4f}s_{case_name}_Re={Re:.2f}_aoa={aoa:.2f}.dat"
            elif time_step is not None:
                # 使用时间步索引
                saving_dir = f"{state_save_dir}/step_{time_step:04d}"
                os.makedirs(saving_dir, exist_ok=True)
                saving_path = f"{saving_dir}/{case_name}_Re={Re:.2f}_dt={dt:.3f}_source={source:.2f}_aoa={aoa:.2f}.dat"
            else:
                # Default case: use state_save_dir if available, otherwise use current directory
                if state_save_dir is not None:
                    os.makedirs(state_save_dir, exist_ok=True)
                    saving_path = f"{state_save_dir}/{case_name}_Re={Re:.2f}_aoa={aoa:.2f}.dat"
                else:
                    saving_path = f"{case_name}_Re={Re:.2f}_aoa={aoa:.2f}.dat"
        else:
            saving_path = file_name

        write_u_tecplotzone(
            filename=saving_path,
            datasets=write_dataset,
            time_step_length=1,
        )
        
def export_Uref_to_tecplot(mesh_pos,u_ref,physical_time,folder_name):
    time_dir = f"{folder_name}/t_{physical_time:.4f}s"
    os.makedirs(time_dir, exist_ok=True)
    filename = f"{time_dir}/mixflow_t_{physical_time:.4f}.dat"
    tecplot_dir = os.path.dirname(filename)
    if not os.path.exists(tecplot_dir):
        os.makedirs(tecplot_dir)  
    u_ref = u_ref.cpu().detach().numpy()
    I = int(math.sqrt(mesh_pos.shape[0]))
    J = I
    xx = mesh_pos.to('cpu')[:, 0].reshape(I, J)   # 所有 x
    yy = mesh_pos.to('cpu')[:, 1].reshape(I, J)   # 所有 y
    
    with open(filename, 'w') as f:
        # 写入文件头 (修正：这是一个标量场，只有u)
        f.write('VARIABLES = "X", "Y", "u"\n')
        f.write(f'ZONE I={I}, J={J}, F=POINT\n')

        # 写入数据
        for j in range(J):
            for i in range(I):
                u_ref = u_ref.reshape(xx.shape)
                f.write(f'{xx[i, j]:.8e} {yy[i, j]:.8e} {u_ref[i, j]:.8e}\n')
    
def calculate_airfoil_lift_drag(
        mesh,
        uvp_unique,
        inf_u,
        time_step=0,
        device="cuda"
    ):
    """
    计算翼型升力和阻力
    
    Args:
        mesh: 网格字典，包含BC.json配置信息（mesh["solving_params"]）
        time_step: 时间步索引，用于从aoa数组中选取对应的攻角
        device: 计算设备
    
    Returns:
        Cl: 升力系数
        Cd: 阻力系数
    """
    # 从BC.json配置中提取参数
    solving_params = mesh["solving_params"]
    aoa_list = solving_params["aoa"]  # aoa是一个数组
    L = solving_params["L"]  # 特征长度
    
    # 获取当前时间步的攻角（如果aoa是数组，取对应的索引；否则取第一个值）
    if isinstance(aoa_list, (list, tuple, np.ndarray)) and len(aoa_list) > int(time_step):
        alpha = aoa_list[int(time_step)]
    else:
        alpha = aoa_list[0] if isinstance(aoa_list, (list, tuple, np.ndarray)) else aoa_list
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device) * (torch.pi / 180.0)  # 转换为弧度
    
    # 提取物理参数
    rho = solving_params["rho"][0] if isinstance(solving_params["rho"], (list, tuple)) else solving_params["rho"]
    mu = solving_params["mu"][0] if isinstance(solving_params["mu"], (list, tuple)) else solving_params["mu"]
    rho = torch.tensor(rho, dtype=torch.float32, device=device)
    mu = torch.tensor(mu, dtype=torch.float32, device=device)
    L = torch.tensor(L, dtype=torch.float32, device=device)
    
    if not torch.is_tensor(inf_u):
        inf_u = torch.tensor(inf_u, dtype=torch.float32, device=device)
    else:
        inf_u = inf_u.to(device)

    # 确保核心张量在同一设备上
    uvp_unique = uvp_unique.to(device)
    mesh_pos_unique = mesh["mesh_pos_unique"].to(device)
    
    # 识别边界节点（使用去重后的网格）
    node_type = mesh.get("node_type_unique", None)
    if node_type is None:
        raise ValueError("mesh 缺少 node_type_unique")
    node_type = node_type.to(device)
    mask_node_boundary = (node_type == NodeType.WALL)
    num_bc_nodes = int(mask_node_boundary.sum().item())
    if num_bc_nodes < 2:
        print("Warning: boundary nodes < 2, skip lift/drag; check node_type_unique and BC labels.")
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    # 获取边界节点数据
    wall_p = uvp_unique[mask_node_boundary, 2:3]  # 压力
    boundary_mesh_pos = mesh_pos_unique[mask_node_boundary, :]
    
    # 计算速度梯度（使用去重后的数据）
    # 需要确保edge_index也是基于去重后的网格
    support_edge = mesh["support_edge"].to(device)
    mesh_pos_for_grad = mesh["mesh_pos_unique"].to(device).float()
    if mesh_pos_for_grad.shape[0] != uvp_unique.shape[0]:
        print("Warning: mesh_pos_for_grad size mismatch uvp_unique; gradients may be zero.")
    
    # 如果support_edge是基于原始网格的，需要映射到去重后的网格
    # 这里假设已经提供了正确的edge_index，或者需要预处理
    grad_uv = node_based_WLSQ(
        phi_node=uvp_unique[:, 0:2],  # 使用去重后的速度
        edge_index=support_edge,
        mesh_pos=mesh_pos_for_grad,
        dual_edge=False,
        order="2nd",
        precompute_Moments=None,
    )
    
    grad_wall = grad_uv[mask_node_boundary, :, 0:2]
    
    # 依据几何中心按角度排序，保证边界点方向一致（顺/逆时针均可）
    num_bc_nodes = boundary_mesh_pos.shape[0]
    original_order = torch.arange(num_bc_nodes, device=device)
    center = boundary_mesh_pos.mean(dim=0, keepdim=True)
    rel = boundary_mesh_pos - center
    angles = torch.atan2(rel[:, 1], rel[:, 0])

    order_ccw = torch.argsort(angles)
    order_cw = torch.argsort(angles, descending=True)

    def _is_cyclic_match(order_a, order_b):
        """检查两个序列是否仅相差循环起点。"""
        for shift in range(order_a.numel()):
            if torch.equal(order_a, order_b.roll(shifts=-shift, dims=0)):
                return True
        return False

    if _is_cyclic_match(original_order, order_ccw) or _is_cyclic_match(original_order, order_cw):
        ordered_idx = original_order
    else:
        signed_area = (
            boundary_mesh_pos[:, 0] * boundary_mesh_pos.roll(-1, 0)[:, 1]
            - boundary_mesh_pos.roll(-1, 0)[:, 0] * boundary_mesh_pos[:, 1]
        ).sum() * 0.5
        ordered_idx = order_ccw if signed_area >= 0 else order_cw

    left_node = ordered_idx
    right_node = ordered_idx.roll(-1, 0)

    # 计算面中心的值
    grad_wall_face = (grad_wall[left_node] + grad_wall[right_node]) / 2
    wall_face_vec = boundary_mesh_pos[right_node] - boundary_mesh_pos[left_node]
    wall_face_length = wall_face_vec.norm(dim=1, keepdim=True)
    if torch.any(wall_face_length < 1e-12):
        print("Warning: zero-length boundary edges detected; check boundary ordering or duplicates.")
        wall_face_length = wall_face_length + 1e-12
    wall_tan = -(boundary_mesh_pos[right_node] - boundary_mesh_pos[left_node]) / wall_face_length
    wall_nor = torch.cat([-wall_tan[:, 1:2], wall_tan[:, 0:1]], dim=1)
    
    # 计算速度梯度分量
    u_xy = grad_wall_face[:, 0, :]
    v_xy = grad_wall_face[:, 1, :]
    
    u_r = (u_xy * wall_nor).sum(1, keepdim=True)
    v_r = (v_xy * wall_nor).sum(1, keepdim=True)
    uth_r = u_r * wall_tan[:, 0:1] + v_r * wall_tan[:, 1:2]
    
    # 计算面中心压力
    wall_face_p = (wall_p[left_node] + wall_p[right_node]) / 2
    
    # 计算压力系数和摩擦系数
    denom = 0.5 * rho * inf_u ** 2
    if torch.abs(denom) < 1e-12:
        print("Warning: inf_u near zero; lift/drag will be zero.")
        denom = denom + 1e-12

    Cp = wall_face_p / denom
    Cf = -uth_r * mu / denom
    
    # 计算力和力矩
    Force_p = Cp * wall_face_length * wall_nor
    Force_f = Cf * wall_face_length * wall_tan
    
    # 计算升力分量（压力+摩擦）
    Cl_p = (torch.cos(alpha) * Force_p[:, 1:2] - torch.sin(alpha) * Force_p[:, 0:1]).sum()
    Cd_p = (torch.sin(alpha) * Force_p[:, 1:2] + torch.cos(alpha) * Force_p[:, 0:1]).sum()
    
    Cl_f = (torch.cos(alpha) * Force_f[:, 1:2] - torch.sin(alpha) * Force_f[:, 0:1]).sum()
    Cd_f = (torch.sin(alpha) * Force_f[:, 1:2] + torch.cos(alpha) * Force_f[:, 0:1]).sum()
    
    # 归一化到特征长度
    Cl = (Cl_p + Cl_f) / L
    Cd = (Cd_p + Cd_f) / L
    
    return Cl, Cd


def compute_and_save_lift_drag_coefficients(
    mesh,
    uv_uns,
    p_uns,
    case_name,
    Re,
    aoa,
    time_step,
    physical_time,
    dt,
    state_save_dir,
    device,
    plot_count,
    saving_path
):
    """
    计算升阻力系数并保存到文件
    
    Args:
        mesh: 网格数据字典，需要包含solving_params
        uv_uns: 去重后的速度张量 [1, N_unique, 2] (u, v)
        p_uns: 去重后的压力张量 [1, N_unique, 1] (p)
        case_name: 案例名称
        Re: 雷诺数
        aoa: 攻角
        time_step: 时间步索引
        physical_time: 物理时间
        dt: 时间步长
        state_save_dir: 保存目录
        device: 计算设备
        plot_count: 绘图计数（用于确定time_step）
        saving_path: dat文件保存路径
    
    Returns:
        lift_drag_result: dict - 包含升阻力系数、相对误差和其他参数的字典，如果计算失败则返回None
    """
    import json
    import os
    
    # 检查mesh是否包含solving_params
    if "solving_params" not in mesh:
        return None
    
    try:
        # 准备uvp_unique数据
        # uv_uns是[1, N, 2]，p_uns是[1, N, 1]，需要去掉第0维并合并
        uv_unique = uv_uns.squeeze(0).float()  # [N, 2] (u, v)
        p_unique = p_uns.squeeze(0).float()  # [N, 1] (p)
        uvp_unique = torch.cat([uv_unique, p_unique], dim=1)  # [N, 3] (u, v, p)
        

        
        # 获取入口速度
        solving_params = mesh["solving_params"]
        inlet_list = solving_params.get("inlet", [1.0])
        if isinstance(inlet_list, (list, tuple)):
            # 如果inlet是数组，根据time_step选择，否则取第一个
            if time_step is not None and int(time_step) < len(inlet_list):
                inf_u = inlet_list[int(time_step)]
            else:
                inf_u = inlet_list[0]
        else:
            inf_u = inlet_list

        
        # 确定time_step索引（用于选择aoa），确保是整数
        calc_time_step = int(time_step) if time_step is not None else 0
        
        # 调用计算函数
        Cl, Cd = calculate_airfoil_lift_drag(
            mesh=mesh,
            uvp_unique=uvp_unique,
            inf_u=inf_u,
            time_step=calc_time_step,
            device=device
        )
        
        Cl_value = Cl.item()
        Cd_value = Cd.item()

        # # 读取参考升阻力系数（来自 Comsol_ref/{case_name}/Ref_coefficient.json）
        # ref_Cl = None
        # ref_Cd = None
        # rel_err_Cl = None
        # rel_err_Cd = None
        # try:
        #     repo_root = os.path.dirname(os.path.dirname(__file__))
        #     # 使用 case_name 作为子目录名称，以支持不同算例（如 NACA0012）
        #     ref_dir = os.path.join(repo_root, "Comsol_ref", str(case_name))
        #     ref_json_path = os.path.join(ref_dir, "Ref_coefficient.json")
        #     with open(ref_json_path, "r") as rf:
        #         ref_data = json.load(rf)
        #     coeff = ref_data.get("coefficient", {})
        #     ref_Cl = float(coeff.get("Cl"))
        #     ref_Cd = float(coeff.get("Cd"))

        #     # 使用 torch.norm 计算相对误差
        #     ref_Cl_t = torch.tensor(ref_Cl, dtype=Cl.dtype, device=Cl.device)
        #     ref_Cd_t = torch.tensor(ref_Cd, dtype=Cd.dtype, device=Cd.device)

        #     # 避免除零
        #     rel_err_Cl_t = torch.norm(Cl - ref_Cl_t) / (torch.norm(ref_Cl_t) + 1e-12)
        #     rel_err_Cd_t = torch.norm(Cd - ref_Cd_t) / (torch.norm(ref_Cd_t) + 1e-12)

        #     rel_err_Cl = rel_err_Cl_t.item()
        #     rel_err_Cd = rel_err_Cd_t.item()
        # except Exception as e:
        #     print(f"Warning: 无法读取或解析参考升阻力系数: {e}")
        
        # 构建结果字典
        lift_drag_result = {
            "case_name": case_name,
            "Cl": Cl_value,
            "Cd": Cd_value,
            "Re": Re,
            "aoa": aoa,
            "time_step": calc_time_step if time_step is not None else plot_count,
            "physical_time": physical_time if physical_time is not None else (dt * calc_time_step if time_step is not None else dt * plot_count),
            "dat_file": saving_path,
        }
        # 如果存在参考值与相对误差，则一并保存
        # if ref_Cl is not None and ref_Cd is not None and rel_err_Cl is not None and rel_err_Cd is not None:
        #     lift_drag_result["ref_Cl"] = ref_Cl
        #     lift_drag_result["ref_Cd"] = ref_Cd
        #     lift_drag_result["rel_err_Cl"] = rel_err_Cl
        #     lift_drag_result["rel_err_Cd"] = rel_err_Cd
        
        # 从solving_params获取其他参数
        if "rho" in solving_params:
            rho_list = solving_params["rho"]
            rho = rho_list[calc_time_step] if isinstance(rho_list, (list, tuple)) and calc_time_step < len(rho_list) else (rho_list[0] if isinstance(rho_list, (list, tuple)) else rho_list)
            lift_drag_result["rho"] = rho
        if "mu" in solving_params:
            mu_list = solving_params["mu"]
            mu = mu_list[calc_time_step] if isinstance(mu_list, (list, tuple)) and calc_time_step < len(mu_list) else (mu_list[0] if isinstance(mu_list, (list, tuple)) else mu_list)
            lift_drag_result["mu"] = mu
        if "L" in solving_params:
            lift_drag_result["L"] = solving_params["L"]
        
        # 保存升阻力结果到.dat文件（同时保存与参考值的相对误差）
        lift_drag_file = os.path.join(state_save_dir, "lift_drag_coefficients.dat")
        
        # 检查文件是否存在，以确定是否需要写入文件头
        file_exists = os.path.exists(lift_drag_file)
        
        # 准备公共信息（文件头）- 只在文件不存在时写入
        if not file_exists:
            header_info = {
                "case_name": case_name,
                "Re": Re,
                "aoa": aoa,
            }
            # 从solving_params获取公共参数
            if "rho" in solving_params:
                rho_list = solving_params["rho"]
                rho_val = rho_list[calc_time_step] if isinstance(rho_list, (list, tuple)) and calc_time_step < len(rho_list) else (rho_list[0] if isinstance(rho_list, (list, tuple)) else rho_list)
                header_info["rho"] = rho_val
            if "mu" in solving_params:
                mu_list = solving_params["mu"]
                mu_val = mu_list[calc_time_step] if isinstance(mu_list, (list, tuple)) and calc_time_step < len(mu_list) else (mu_list[0] if isinstance(mu_list, (list, tuple)) else mu_list)
                header_info["mu"] = mu_val
            if "L" in solving_params:
                header_info["L"] = solving_params["L"]
            # dat_file路径（保存第一个时间步的路径作为参考）
            if saving_path:
                header_info["dat_file"] = saving_path
            
            # 写入文件头
            with open(lift_drag_file, 'w') as f:
                # 写入公共信息作为注释
                f.write("# Lift-Drag Coefficients Data File\n")
                f.write("# Common Parameters (only saved once)\n")
                for key, value in header_info.items():
                    if isinstance(value, str):
                        f.write(f"# {key} = {value}\n")
                    elif isinstance(value, (int, float)):
                        f.write(f"# {key} = {value:.10e}\n")
                    else:
                        f.write(f"# {key} = {value}\n")
                f.write("#\n")
                f.write("# Data columns: time_step, physical_time, Cl, Cd\n")
                f.write('VARIABLES = "time_step"\n"physical_time"\n"Cl"\n"Cd"\n"')
                f.write(f'DATASETAUXDATA Common.time_stepVar="1"\n')
                f.write(f'DATASETAUXDATA Common.physical_timeVar="2"\n')
                f.write(f'DATASETAUXDATA Common.ClVar="3"\n')
                f.write(f'DATASETAUXDATA Common.CdVar="4"\n')
                # f.write(f'DATASETAUXDATA Common.rel_err_ClVar="5"\n')
                # f.write(f'DATASETAUXDATA Common.rel_err_CdVar="6"\n')

        
        # 追加当前时间步的数据
        time_step_val = calc_time_step if time_step is not None else plot_count
        physical_time_val = physical_time if physical_time is not None else (dt * calc_time_step if time_step is not None else dt * plot_count)
        with open(lift_drag_file, 'a') as f:
            f.write(
                f"{time_step_val}, {physical_time_val:.10e}, "
                f"{Cl_value:.10e}, {Cd_value:.10e}\n"
            )
        
        print(f"升阻力系数已计算并保存: Cl={Cl_value:.6f}, Cd={Cd_value:.6f}")
        return lift_drag_result
        
    except Exception as e:
        print(f"Warning: 计算升阻力系数时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_surface_pressure(mesh,
                         p_uns,
                         case_name,
                         Re,
                         aoa,
                         time_step,
                         physical_time,
                         dt,
                         state_save_dir,
                         device,
                         plot_count,
                         saving_path):
    """
    将去重后的翼型表面坐标和压力导出为 Tecplot DAT 文件（每个时间步一个文件）。

    仅导出 node_type_unique == NodeType.WALL 的节点：
        X, Y, Sur_P

    文件命名与 export_to_tecplot 保持一致风格：
        - 若提供 physical_time:  state_save_dir/t_{time}s/t_{time}s_*_SurfaceP.dat
        - 若提供 time_step:     state_save_dir/step_xxxx/*_SurfaceP.dat
        - 否则使用 plot_count 分段目录。
    """
    # 1. 获取并检查去重后的节点类型
    node_type_unique = mesh.get("node_type_unique", None)
    if node_type_unique is None:
        print("Warning: mesh 缺少 node_type_unique，无法导出表面压力。")
        return

    # 统一到与 p_uns 相同的 device，避免索引时报 device 不一致
    target_device = p_uns.device
    node_type_unique = node_type_unique.to(target_device)
    mask_node_boundary = (node_type_unique == NodeType.WALL)

    num_bc_nodes = int(mask_node_boundary.sum().item())
    if num_bc_nodes == 0:
        print("Warning: 未找到 node_type_unique == NodeType.WALL 的节点，跳过表面压力导出。")
        return

    # 2. 获取去重后的坐标和压力（也放到与 p_uns 相同的 device 上）
    mesh_pos_unique = mesh["mesh_pos_unique"].to(target_device)

    # p_uns 可能是 [1, N, 1] 或 [N, 1]
    if p_uns.dim() == 3:
        # [1, N, 1] -> [N]
        wall_p = p_uns[0, mask_node_boundary, 0]
    elif p_uns.dim() == 2:
        # [N, 1] 或 [N] -> [N]
        wall_p = p_uns[mask_node_boundary].view(-1)
    else:
        raise ValueError(f"Unsupported p_uns shape: {p_uns.shape}, expected [1,N,1] or [N,1].")

    boundary_mesh_pos = mesh_pos_unique[mask_node_boundary, :]

    # 3. 构造保存路径（风格参考 export_to_tecplot，但生成独立的表面压力文件）
    try:
        Re_val = mesh["Re"].squeeze().item()
    except Exception:
        Re_val = 0.0

    try:
        aoa_val = mesh["aoa"].squeeze().item()
    except Exception:
        aoa_val = aoa

    case_name_val = mesh.get("case_name", case_name)

    if physical_time is not None:
        time_dir = f"{state_save_dir}/t_{physical_time:.4f}s"
        os.makedirs(time_dir, exist_ok=True)
        filename = (
            f"{time_dir}/t_{physical_time:.4f}s_"
            f"{case_name_val}_Re={Re_val:.2f}_aoa={aoa_val:.2f}_SurfaceP.dat"
        )
    elif time_step is not None:
        saving_dir = f"{state_save_dir}/step_{int(time_step):04d}"
        os.makedirs(saving_dir, exist_ok=True)
        filename = (
            f"{saving_dir}/{case_name_val}_Re={Re_val:.2f}_dt={dt:.3f}_aoa={aoa_val:.2f}_SurfaceP.dat"
        )
    else:
        save_dir_num = plot_count // 50
        saving_dir = f"{state_save_dir}/{save_dir_num*50}-{(save_dir_num+1)*50}"
        os.makedirs(saving_dir, exist_ok=True)
        filename = (
            f"{saving_dir}/NO.{plot_count}_{case_name_val}_Re={Re_val:.2f}_dt={dt:.3f}_aoa={aoa_val:.2f}_SurfaceP.dat"
        )

    # 4. 写 Tecplot DAT 文件（单 ZONE，按时间步一个文件）
    X = boundary_mesh_pos[:, 0].detach().cpu().numpy().astype(np.float32)
    Y = boundary_mesh_pos[:, 1].detach().cpu().numpy().astype(np.float32)
    P = wall_p.detach().cpu().numpy().astype(np.float32)

    # 物理时间 / 解的时间
    if physical_time is not None:
        solution_time = float(physical_time)
    elif time_step is not None:
        solution_time = float(time_step) * float(dt)
    else:
        solution_time = float(plot_count) * float(dt)

    with open(filename, "w") as f:
        # 文件头
        f.write('TITLE = "FOGN surface pressure"\n')
        f.write('VARIABLES = "X"\n"Y"\n"Sur_P"\n')
        f.write('DATASETAUXDATA Common.Sur_PVar="3"\n')

        # 使用 ORDERED 1D 区域，点按顺序连接成折线
        f.write(
            f'ZONE T="Surface_Pressure", '
            f"STRANDID=1, SOLUTIONTIME={solution_time}\n"
        )
        f.write(f"I={num_bc_nodes}, DATAPACKING=POINT\n")

        # 数据：每行一个节点 (X, Y, Sur_P)
        for xi, yi, pi in zip(X, Y, P):
            f.write(f"{xi:.8e} {yi:.8e} {pi:.8e}\n")

    print(f"表面压力已导出到: {filename}")

id1_kenerl = torch.Tensor([[1, 0],[0, 0]]).unsqueeze(0).unsqueeze(0)
def get_id1(id):
    return F.conv2d(torch.from_numpy(id).unsqueeze(0).unsqueeze(0).float(), id1_kenerl, padding=(0, 0)).long().reshape(-1).numpy()

id2_kenerl = torch.Tensor([[0, 1],[0, 0]]).unsqueeze(0).unsqueeze(0)
def get_id2(id):
    return F.conv2d(torch.from_numpy(id).unsqueeze(0).unsqueeze(0).float(), id2_kenerl, padding=(0, 0)).long().reshape(-1).numpy()

id3_kenerl = torch.Tensor([[0, 0],[0, 1]]).unsqueeze(0).unsqueeze(0)
def get_id3(id):
    return F.conv2d(torch.from_numpy(id).unsqueeze(0).unsqueeze(0).float(), id3_kenerl, padding=(0, 0)).long().reshape(-1).numpy()

id4_kenerl = torch.Tensor([[0, 0],[1, 0]]).unsqueeze(0).unsqueeze(0)
def get_id4(id):
    return F.conv2d(torch.from_numpy(id).unsqueeze(0).unsqueeze(0).float(), id4_kenerl, padding=(0, 0)).long().reshape(-1).numpy()

n_kernel =torch.Tensor([[0, 1,0],[0, 0,0],[0, 0,0]]).unsqueeze(0).unsqueeze(0)
def find_north(v):
    if isinstance(v, np.ndarray):
        return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), n_kernel, padding=(0, 0)).long().reshape(-1).numpy()
    elif isinstance(v, torch.Tensor):
        return F.conv2d(v.unsqueeze(0).unsqueeze(0).float(), n_kernel.cuda(), padding=(0, 0)).long().reshape(-1)

s_kernel = torch.Tensor([[0, 0,0],[0, 0,0],[0, 1,0]]).unsqueeze(0).unsqueeze(0)
def find_south(v):
    if isinstance(v, torch.Tensor):
        return F.conv2d(v.unsqueeze(0).unsqueeze(0).float(), s_kernel.cuda(), padding=(0, 0)).long().reshape(-1)
    elif isinstance(v, np.ndarray):
        return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), s_kernel, padding=(0, 0)).long().reshape(-1).numpy()

w_kernel = torch.Tensor([[0, 0,0],[1, 0,0],[0, 0,0]]).unsqueeze(0).unsqueeze(0)
def find_west(v):
    if isinstance(v, torch.Tensor):
        return F.conv2d(v.unsqueeze(0).unsqueeze(0).float(), w_kernel.cuda(), padding=(0, 0)).long().reshape(-1)
    
    elif isinstance(v, np.ndarray):
        return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), w_kernel, padding=(0, 0)).long().reshape(-1).numpy()

e_kernel = torch.Tensor([[0, 0,0],[0, 0,1],[0, 0,0]]).unsqueeze(0).unsqueeze(0)

def find_east(v):
    if isinstance(v, torch.Tensor):
        return F.conv2d(v.unsqueeze(0).unsqueeze(0).float(), e_kernel.cuda(), padding=(0, 0)).long().reshape(-1)
    elif isinstance(v, np.ndarray): 
        return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), e_kernel, padding=(0, 0)).long().reshape(-1).numpy()

mid_kernel = torch.Tensor([[0, 0,0],[0, 1,0],[0, 0,0]]).unsqueeze(0).unsqueeze(0)
def find_mid(v):
    if isinstance(v, torch.Tensor):
        return F.conv2d(v.unsqueeze(0).unsqueeze(0).float(), mid_kernel.cuda(), padding=(0, 0)).long().reshape(-1)
    elif isinstance(v, np.ndarray):
        return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), mid_kernel, padding=(0, 0)).long().reshape(-1).numpy()

left_kernel = torch.Tensor([1, 0]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
def get_left(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), left_kernel, padding=(0, 0)).long().reshape(-1).numpy()

right_kernel = torch.Tensor([0, 1]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
def get_right(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), right_kernel, padding=(0, 0)).long().reshape(-1).numpy()

up_kernel = torch.Tensor([1, 0]).unsqueeze(0).unsqueeze(1).unsqueeze(3)
def get_up(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), up_kernel, padding=(0, 0)).long().reshape(-1).numpy()

down_kernel = torch.Tensor([0, 1]).unsqueeze(0).unsqueeze(1).unsqueeze(3)
def get_down(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), down_kernel, padding=(0, 0)).long().reshape(-1).numpy()


left_kernel_node = torch.Tensor([1, 0,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
def get_left_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), left_kernel_node, padding=(0, 0)).long().reshape(-1).numpy()

right_kernel_node = torch.Tensor([0,0, 1]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
def get_right_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), right_kernel_node, padding=(0, 0)).long().reshape(-1).numpy()

up_kernel_node = torch.Tensor([1, 0,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3)
def get_up_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), up_kernel_node, padding=(0, 0)).long().reshape(-1).numpy()

down_kernel_node= torch.Tensor([0,0, 1]).unsqueeze(0).unsqueeze(1).unsqueeze(3)
def get_down_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), down_kernel_node, padding=(0, 0)).long().reshape(-1).numpy()

kernel_1 = torch.Tensor([[1, 0,0],[0, 0,0],[0, 0,0]]).unsqueeze(0).unsqueeze(0)
def node_1(v):
    if isinstance(v, torch.Tensor):
        return F.conv2d(v.unsqueeze(0).unsqueeze(0).float(), kernel_1.cuda(), padding=(0, 0)).long().reshape(-1)
    elif isinstance(v, np.ndarray):
        return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kernel_1, padding=(0, 0)).long().reshape(-1).numpy()

kernel_3 = torch.Tensor([[0, 0,1],[0, 0,0],[0, 0,0]]).unsqueeze(0).unsqueeze(0)
def node_3(v):
    if isinstance(v, torch.Tensor):
        return F.conv2d(v.unsqueeze(0).unsqueeze(0).float(), kernel_3.cuda(), padding=(0, 0)).long().reshape(-1)
    elif isinstance(v, np.ndarray):
        return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kernel_3, padding=(0, 0)).long().reshape(-1).numpy()

kernel_7 = torch.Tensor([[0, 0,0],[0, 0,0],[1, 0,0]]).unsqueeze(0).unsqueeze(0)
def node_7(v):
    if isinstance(v, torch.Tensor):
        return F.conv2d(v.unsqueeze(0).unsqueeze(0).float(), kernel_7.cuda(), padding=(0, 0)).long().reshape(-1)
    elif isinstance(v, np.ndarray):
        return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kernel_7, padding=(0, 0)).long().reshape(-1).numpy()

kernel_9 = torch.Tensor([[0, 0,0],[0, 0,0],[0, 0,1]]).unsqueeze(0).unsqueeze(0)
def node_9(v):
    if isinstance(v, torch.Tensor):
        return F.conv2d(v.unsqueeze(0).unsqueeze(0).float(), kernel_9.cuda(), padding=(0, 0)).long().reshape(-1)
    elif isinstance(v, np.ndarray):
        return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kernel_9, padding=(0, 0)).long().reshape(-1).numpy()

kenerl_ww = torch.Tensor([1, 0,0,0,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
def ww_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kenerl_ww, padding=(0, 0)).long().reshape(-1).numpy()

kenerl_w = torch.Tensor([0, 1,0,0,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
def w_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kenerl_w, padding=(0, 0)).long().reshape(-1).numpy()

kenerl_ee = torch.Tensor([0, 0,0,0,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
def ee_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kenerl_ee, padding=(0, 0)).long().reshape(-1).numpy()

kenerl_e = torch.Tensor([0, 0,0,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
def e_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kenerl_e, padding=(0, 0)).long().reshape(-1).numpy()

kenerl_nn = torch.Tensor([1, 0,0,0,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3)
def nn_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kenerl_nn, padding=(0, 0)).long().reshape(-1).numpy()

kenerl_n = torch.Tensor([0, 1,0,0,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3)
def n_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kenerl_n, padding=(0, 0)).long().reshape(-1).numpy()

kenerl_ss = torch.Tensor([0, 0,0,0,1]).unsqueeze(0).unsqueeze(1).unsqueeze(3)
def ss_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kenerl_ss, padding=(0, 0)).long().reshape(-1).numpy()

kenerl_s = torch.Tensor([0, 0,0,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3)
def s_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), kenerl_s, padding=(0, 0)).long().reshape(-1).numpy()


find_kenerl_ww = torch.Tensor([[0, 0,0,0,0],[0, 0,0,0,0],[1, 0,0,0,0],[0, 0,0,0,0],[0, 0,0,0,0]]).unsqueeze(0).unsqueeze(0)
def find_ww_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), find_kenerl_ww, padding=(0, 0)).long().reshape(-1).numpy()

find_kenerl_ee = torch.Tensor([[0, 0,0,0,0],[0, 0,0,0,0],[0, 0,0,0,1],[0, 0,0,0,0],[0, 0,0,0,0]]).unsqueeze(0).unsqueeze(0)
def find_ee_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), find_kenerl_ee, padding=(0, 0)).long().reshape(-1).numpy()

find_kenerl_nn = torch.Tensor([[0, 0,1,0,0],[0, 0,0,0,0],[0, 0,0,0,0],[0, 0,0,0,0],[0, 0,0,0,0]]).unsqueeze(0).unsqueeze(0)
def find_nn_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), find_kenerl_nn, padding=(0, 0)).long().reshape(-1).numpy()

find_kenerl_ss = torch.Tensor([[0, 0,0,0,0],[0, 0,0,0,0],[0, 0,0,0,0],[0, 0,0,0,0],[0, 0,1,0,0]]).unsqueeze(0).unsqueeze(0)
def find_ss_node(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), find_kenerl_ss, padding=(0, 0)).long().reshape(-1).numpy()


find_kenerl_mid_quick = torch.Tensor([[0, 0,0,0,0],[0, 0,0,0,0],[0, 0,1,0,0],[0, 0,0,0,0],[0, 0,0,0,0]]).unsqueeze(0).unsqueeze(0)
def find_mid_quick(v):
    return F.conv2d(torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float(), find_kenerl_mid_quick, padding=(0, 0)).long().reshape(-1).numpy()