import sys
import os

# cur_path = os.path.split(__file__)[0]
# sys.path.append(cur_path)
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import torch
import torch.nn as nn
from utils.utilities import NodeType
import numpy as np


import get_param
import time
from get_param import get_hyperparam
from utils.Logger import Logger
from utils.utilities import export_Uref_to_tecplot
from torch_geometric.nn import global_add_pool,global_mean_pool

from dataset import Graph_loader

import random
import datetime
import subprocess
from scipy.spatial import KDTree

def u(t, x, y, vt_max=0.385):
    # 计算 r
    r = torch.sqrt(x**2 + y**2)
    # 计算 v_t = sech^2(r) * tanh(r), 其中 sech(r) = 1 / cosh(r)
    v_t = (1.0 / torch.cosh(r))**2 * torch.tanh(r)
    # 计算 omega = (1/r) * (v_t / vt_max)
    omega = torch.zeros_like(r)
    mask = r > 1e-12
    if mask.any():
        omega[mask] = (1.0 / r[mask]) * (v_t[mask] / vt_max)
    # 按附件公式计算 u(t,x,y) = -tanh( (y/2)cos(omega t) - (x/2)sin(omega t) )
    term = (y / 2.0) * torch.cos(omega * t) - (x / 2.0) * torch.sin(omega * t)
    return -torch.tanh(term)

torch.set_float32_matmul_precision('high')
# configurate parameters
params = get_param.params()
seed = int(datetime.datetime.now().timestamp())
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.set_per_process_memory_fraction(0.99, params.on_gpu)
torch.set_num_threads(os.cpu_count() // 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vt_max=0.385

# initialize Logger and load model / optimizer if according parameters were given
logger = Logger(
    get_hyperparam(params),
    use_dat=True,
    params=params,
    copy_code=True,
)


# initialize Training Dataset
start = time.time()
datasets_factory = Graph_loader.DatasetFactory(
    params=params,
    dataset_dir=params.dataset_dir,
    state_save_dir=logger.saving_path,
    device=device,
)

# refresh dataset size
params.dataset_size = datasets_factory.dataset_size

# create dataset objetc
datasets, loader, sampler = datasets_factory.create_datasets(
    batch_size=params.batch_size, num_workers=0, pin_memory=False
)

end = time.time()
print("Training traj has been loaded time consuming:{0}".format(end - start))

# initialize fluid model

class FD_discretizer(nn.Module):
    def __init__(self ) :
        super(FD_discretizer,self).__init__()

    def hard_enforce_BC(self, u_tensor, graph_node, graph_extended, t_current=0.0):
        pos = graph_node.pos
        xi = pos[:, 0]
        yi = pos[:, 1]
        
        # 在当前时间t_current处应用边界条件
        t_bc = torch.tensor(t_current, device=device, dtype=torch.float32)
        u_bc = u(t_bc, xi, yi).reshape(-1, 1)
        
        dummy_node_u = u_tensor.clone()
        
        # 只在边界节点应用约束
        mask_bc = ((graph_node.node_type == NodeType.INFLOW) | 
                   (graph_node.node_type == NodeType.WALL))
        dummy_node_u[mask_bc] = u_bc[mask_bc]

        # 扩展
        extend_index = graph_node.extend_index
        extended_u = u_tensor[extend_index].clone()
        extened_dummy_node_u = dummy_node_u[extend_index].clone()
        node_type_extended = graph_extended.node_type
        mask_bc_extended = ((node_type_extended == NodeType.INFLOW) | 
                           (node_type_extended == NodeType.WALL))
        extended_u[mask_bc_extended] = extened_dummy_node_u[mask_bc_extended]
        
        u_to_vis = dummy_node_u.clone().detach()
        return extended_u, u_to_vis
    
    def forward(self, original_u=None, u_old=None, graph_node=None, graph_edge_xi=None, 
                 graph_edge_eta=None,  graph_extended=None, 
                 graph_Index=None):
        if u_old is not None:
            extended_u_old, _ = self.hard_enforce_BC(u_old, graph_node, graph_extended)
        else:
            extended_u_old = None

        extended_u, u_to_vis = self.hard_enforce_BC(original_u, graph_node, graph_extended)
        #############################################

        original_block_metrics = graph_node.original_block_metrics
        J_o = original_block_metrics[:,4]
        l_node,r_node  =  graph_extended.edge_node_xi_index
        
        d_node,u_node  =  graph_extended.edge_node_eta_index
        l_edge,r_edge = graph_edge_xi.face
        d_edge,u_edge = graph_edge_eta.face

        # pde coefficent
        #dt_node = 0.1
        dt_node = 0.015625
        pde_theta_node = graph_Index.pde_theta[graph_node.batch]
        unsteady_coefficent = 1.0 # pde_theta_node[:, 0:1]
        convection_coefficent = 1.0 # pde_theta_node[:, 2:3]

        relaxtion = graph_Index.relaxtion[graph_node.batch]
  
        ####get metrics on padded blocks###
        metrics_extended = graph_extended.extended_block_metrics  # shape: (N, 5)
        dxi_dx = metrics_extended[:,0]
        dxi_dy = metrics_extended[:,1]
        deta_dx = metrics_extended[:,2]
        deta_dy = metrics_extended[:,3]
        J = metrics_extended[:,4]  

        pos_extended = graph_extended.x
        xi_node = pos_extended[:, 0]
        yi_node = pos_extended[:, 1]
        
        r = torch.sqrt(xi_node**2 + yi_node**2)
        vt_max = 0.385
        v_t = (1.0 / torch.cosh(r))**2 * torch.tanh(r)
        
        omega = torch.zeros_like(r)
        mask_r = r > 1e-12
        omega[mask_r] = (1.0 / r[mask_r]) * (v_t[mask_r] / vt_max)
        
        a = -omega * yi_node
        b = omega * xi_node

        # 向量化计算对流通量（使用物理速度系数 a, b）
        if extended_u_old is not None:
            convect_old = self.convect_flux(extended_u_old, a, b, dxi_dx, dxi_dy, deta_dx, deta_dy, J, l_node, r_node, d_node, u_node, l_edge, r_edge, d_edge, u_edge)
            
        else:
            convect_old = None
            
        convect_new = self.convect_flux(extended_u, a, b, dxi_dx, dxi_dy, deta_dx, deta_dy, J, l_node, r_node, d_node, u_node, l_edge, r_edge, d_edge, u_edge)

        if extended_u_old is not None:
            convect = convect_old * relaxtion + convect_new * (1 - relaxtion)
        else:
            convect = convect_new

        # 向量化非定常项计算
        if extended_u_old is not None:
            unsteady = ((original_u- u_old) / dt_node)/ J_o.unsqueeze(-1)   # shape: (N, 1)

        if extended_u_old is not None:
            loss_pde = (unsteady_coefficent * unsteady + convection_coefficent * convect)  # shape: (N, 1)
        else:
            loss_pde = convection_coefficent * convect  # shape: (N, 1)

        
        return loss_pde, u_to_vis
 

    def convect_flux(self, extended_u, a, b, dxi_dx, dxi_dy, deta_dx, deta_dy, J, l_node, r_node, d_node, u_node, l_edge, r_edge, d_edge, u_edge):

        u_hat = extended_u[:,0]

        # 计算逆变速度 U, V
        U_hat = a * dxi_dx + b * dxi_dy
        V_hat = a * deta_dx + b * deta_dy

        U_face = 0.5*((U_hat/J)[l_node]+(U_hat/J)[r_node])
        V_face = 0.5*((V_hat/J)[d_node]+(V_hat/J)[u_node])

        extended_u_face_xi = 0.5*(u_hat[l_node]+u_hat[r_node])
        extended_u_face_eta = 0.5*(u_hat[d_node]+u_hat[u_node])

        face_e_flux_hat = (extended_u_face_xi[r_edge]*U_face[r_edge]).unsqueeze(-1)
        face_w_flux_hat = (extended_u_face_xi[l_edge]*U_face[l_edge]).unsqueeze(-1)

        face_n_flux_hat =  (extended_u_face_eta[u_edge]*V_face[u_edge]).unsqueeze(-1)

        face_s_flux_hat = (extended_u_face_eta[d_edge]*V_face[d_edge]).unsqueeze(-1)

        dE1u = (face_e_flux_hat[:,0]-face_w_flux_hat[:,0]).unsqueeze(-1)

        dE2u =(face_n_flux_hat[:,0]-face_s_flux_hat[:,0]).unsqueeze(-1)


        return dE1u+dE2u


FD_discretize = FD_discretizer()

n_epochs = params.n_epochs

graph_node,graph_extended_edge_xi,graph_extended_edge_eta,graph_extended_cell,graph_extended_node,graph_Index = [data.to(device) for data in next(iter(loader))]

pos = graph_node.pos
xi = pos[:,0]
yi = pos[:,1]

t_initial = torch.tensor(0.0)
u_exact = u(t_initial,xi,yi)

primes = torch.nn.Parameter(u_exact.clone().to(device).unsqueeze(-1).float())
u_old = primes.clone()

current_time = 0.0
dt_step = 0.015625  # 时间步长（与forward中的dt_node保持一致）

for epoch in range(n_epochs+1):

    primes.data = u_old.clone()
    
    start = time.time()
    optimizer = torch.optim.LBFGS([primes], lr=1.0, max_iter=100, history_size=20, line_search_fn='strong_wolfe')

    closure_results = {}
    
    # 定义closure函数（必需！）
    def closure():
        optimizer.zero_grad()
        
        loss_pde, _ = FD_discretize(
            original_u=primes, u_old=u_old,
            graph_node=graph_node, graph_edge_xi=graph_extended_edge_xi,
            graph_edge_eta=graph_extended_edge_eta,
            graph_extended=graph_extended_node, graph_Index=graph_Index
        )
        
        loss_batch = global_mean_pool(loss_pde**2, batch=graph_node.batch).view(-1)

        loss = torch.mean(torch.log(loss_batch))

        #loss = torch.mean(loss_batch)
        
        loss.backward()
        closure_results['loss'] = loss.cpu().item()
        closure_results['pde_loss'] = (loss_batch).mean().cpu().item()
        
        return loss
    
    # 单次调用，L-BFGS内部会多次迭代
    optimizer.step(closure)

    # epoch结束，用收敛后的结果更新下一时间步
    with torch.no_grad():
        # 重新计算最终的uvp_to_vis（使用优化后的primes）
        _, u_final = FD_discretize(original_u=primes, u_old=u_old,
                                           graph_node=graph_node, graph_edge_xi=graph_extended_edge_xi,
                                           graph_edge_eta=graph_extended_edge_eta, 
                                           graph_extended=graph_extended_node, graph_Index=graph_Index)
        u_to_vis = u_final
        
        # 在当前时间步更新参考解
        t_current = torch.tensor(current_time)
        u_exact_current = u(t_current, xi, yi)
        
    u_old = u_to_vis[:,0:1].clone()

        # perform optimization step
    learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]

    # 输出此epoch的处理时间
    print(f"Epoch {epoch} (t={current_time:.4f}s) completed in {time.time() - start:.2f} seconds")

    # Print the headers and values
    if epoch%1 == 0:

        headers = ["Epoch", "Loss", "Epoch Time"]
        values = [
            epoch,
            f"{closure_results['loss'] :.4e}",
            f"{time.time() - start:.2f}s",
        ]

        # Determine the maximum width for each column
        column_widths = [max(len(str(x)), len(headers[i])) + 2 for i, x in enumerate(values)]

        # Create a format string for each row
        row_format = "".join(["{:<" + str(width) + "}" for width in column_widths])

        print(row_format.format(*headers))
        print(row_format.format(*values))
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        global_idx = graph_node.global_idx.cpu()
        batch = graph_node.batch.cpu()

        datasets.payback_u_for_vis(u_to_vis.detach().cpu(), global_idx, params.dataset_size, batch,
                                 physical_time=current_time, time_step=epoch)
    current_time += dt_step