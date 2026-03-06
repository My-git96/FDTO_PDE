import sys
import os

# cur_path = os.path.split(__file__)[0]
# sys.path.append(cur_path)
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import torch
from utils.utilities import NodeType
import numpy as np

import torch.nn as nn
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
import torch.nn.functional as F
from scipy.spatial import KDTree

# configurate parameters
params = get_param.params()


torch.set_float32_matmul_precision('high')


# configurate parameters
params = get_param.params()
seed = int(datetime.datetime.now().timestamp())
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.set_per_process_memory_fraction(0.99, params.on_gpu)
torch.set_num_threads(os.cpu_count() // 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__( self ) :
        super(FD_discretizer,self).__init__()



    def hard_enforce_BC(self,u,graph_node,graph_extended):



        dummy_node_u = u[:,0:1].clone()


        mask_bc = ((graph_node.node_type==NodeType.INFLOW)|(graph_node.node_type==NodeType.WALL)) 


        dummy_node_u[mask_bc] = 0. 



        extend_index = graph_node.extend_index

        dummy_extended_u = dummy_node_u[extend_index]

        extended_u = u[extend_index].clone()



        mask_bc_extended = (graph_extended.node_type==NodeType.INFLOW)|(graph_extended.node_type==NodeType.WALL)

        extended_u[mask_bc_extended,0:1] = dummy_extended_u[mask_bc_extended,0:1]





        u_to_vis = u.clone().detach()

        return extended_u,u_to_vis

    def forward(self, original_u=None,u_old = None, graph_node=None,graph_edge_xi=None,graph_edge_eta=None,graph_block_cell=None,graph_extended=None,graph_Index=None,params=None):

        if u_old is not None:

            extended_u_old,_ = self.hard_enforce_BC(u_old,graph_node,graph_extended)
        else:
            extended_u_old = None

        extended_u,u_to_vis = self.hard_enforce_BC(original_u,graph_node,graph_extended)


        #############################################

        block_cells_node = graph_extended.block_cells_node_index
   
        original_block_metrics = graph_node.original_block_metrics
        J_o = original_block_metrics[:,4]
        
        l_node,r_node  =  graph_extended.edge_node_xi_index
        
        d_node,u_node  =  graph_extended.edge_node_eta_index


        l_edge,r_edge = graph_edge_xi.face
        d_edge,u_edge = graph_edge_eta.face

        l_cell,r_cell = graph_block_cell.xi_cell_index
        d_cell,u_cell = graph_block_cell.eta_cell_index


        # pde coefficent
        dt_node = graph_Index.dt_graph[graph_node.batch]
        pde_theta_node = graph_Index.pde_theta[graph_node.batch]

        unsteady_coefficent = pde_theta_node[:, 0:1]

        relaxtion = graph_Index.relaxtion[graph_node.batch] 
 
        ####get metrics on padded blocks###
        metrics_extended = graph_extended.extended_block_metrics  # shape: (N, 5)
        dxi_dx = metrics_extended[:,0]
        dxi_dy = metrics_extended[:,1]
        deta_dx = metrics_extended[:,2]
        deta_dy = metrics_extended[:,3]
        J = metrics_extended[:,4]  

        if extended_u_old is not None:
            diffuse_flux_old = self.diffuse_flux(extended_u_old,dxi_dx,dxi_dy,deta_dx,deta_dy,J,l_node,r_node,d_node,u_node,block_cells_node,l_cell,r_cell,d_cell,u_cell,l_edge,r_edge,d_edge,u_edge)
        diffuse_flux_new = self.diffuse_flux(extended_u,dxi_dx,dxi_dy,deta_dx,deta_dy,J,l_node,r_node,d_node,u_node,block_cells_node,l_cell,r_cell,d_cell,u_cell,l_edge,r_edge,d_edge,u_edge)

        # 向量化松弛计算 - 一次性计算所有通量
        if extended_u_old is not None:
            
            diffuse_flux = diffuse_flux_old * relaxtion + diffuse_flux_new * (1 - relaxtion)

        else:
      
            diffuse_flux = diffuse_flux_new
     
        dt_node  = 0.01
        # 向量化非定常项计算
        if extended_u_old is not None:
            unsteady = ((original_u- u_old) / dt_node) / J_o.unsqueeze(-1)  # shape: (N, 1)
        
        alpha = 0.05
        
        # 向量化动量方程损失计算,守恒型差分，约等于有限体积
        if extended_u_old is not None:
            loss_pde = (unsteady_coefficent * unsteady - alpha*(
                       1.0 * diffuse_flux))  # shape: (N, 1)
        else:
            loss_pde = alpha*(                 
                        1.0 * diffuse_flux)  # shape: (N, 1)
    
        return loss_pde,u_to_vis
 

    def diffuse_flux(self,extended_u_hat,dxi_dx,dxi_dy,deta_dx,deta_dy,J,l_node,r_node,d_node,u_node,block_cells_node,l_cell,r_cell,d_cell,u_cell,l_edge,r_edge,d_edge,u_edge):
        u_hat = extended_u_hat[:,0]
   
        #Calculating Diffusive terms, here no numerical dissipating needed
        #First Calculating at half points to maintain 3 point stencil        
        J_half_x = (0.5*(J[l_node]+J[r_node])).unsqueeze(-1)
        J_half_y = (0.5*(J[d_node]+J[u_node])).unsqueeze(-1)
        
        
        #g11和g22的半节点版本#######################################
        g11_half = ((0.5*(dxi_dx[l_node]+dxi_dx[r_node])).pow(2)+(0.5*(dxi_dy[l_node]+dxi_dy[r_node])).pow(2)).unsqueeze(-1)

        g22_half = ((0.5*(deta_dx[d_node]+deta_dx[u_node])).pow(2)+(0.5*(deta_dy[d_node]+deta_dy[u_node])).pow(2)).unsqueeze(-1)

        g12_half = ((0.5*(dxi_dx[l_node]+dxi_dx[r_node]))*(0.5*(deta_dx[l_node]+deta_dx[r_node]))+(0.5*(dxi_dy[l_node]+dxi_dy[r_node]))*(0.5*(deta_dy[l_node]+deta_dy[r_node]))).unsqueeze(-1)

        g21_half = ((0.5*(dxi_dx[d_node]+dxi_dx[u_node]))*(0.5*(deta_dx[d_node]+deta_dx[u_node]))+(0.5*(dxi_dy[d_node]+dxi_dy[u_node]))*(0.5*(deta_dy[d_node]+deta_dy[u_node]))).unsqueeze(-1)

        du_dxi_on_xi = (u_hat[r_node]-u_hat[l_node]).unsqueeze(-1)

        u_xi = 0.5*(u_hat[r_node]+u_hat[l_node]).unsqueeze(-1)

        u_eta = 0.5*(u_hat[u_node]+u_hat[d_node]).unsqueeze(-1)

        cells_u = ((u_hat[block_cells_node[0]]+u_hat[block_cells_node[1]]+u_hat[block_cells_node[2]]+u_hat[block_cells_node[3]])/4).unsqueeze(-1)
       

        du_deta_on_xi = cells_u[u_cell]-cells_u[d_cell]


        du_deta_on_eta = (u_hat[u_node]-u_hat[d_node]).unsqueeze(-1)


        du_dxi_on_eta = cells_u[r_cell]-cells_u[l_cell]
 

        Ev_1_u = u_xi*((g11_half*du_dxi_on_xi/J_half_x)+(g12_half*du_deta_on_xi/J_half_x))



        Ev_2_u = u_eta*((g22_half*du_deta_on_eta/J_half_y)+(g21_half*du_dxi_on_eta/J_half_y))


   
        dEv_1_u = (Ev_1_u[r_edge]-Ev_1_u[l_edge])
  
        
        dEv_2_u = (Ev_2_u[u_edge]-Ev_2_u[d_edge])
  

        return dEv_1_u+dEv_2_u


FD_discretize = FD_discretizer()

n_epochs = params.n_epochs

graph_node,graph_extended_edge_xi,graph_extended_edge_eta,graph_extended_cell,graph_extended_node,graph_Index = [data.to(device) for data in next(iter(loader))]

pos = graph_node.pos
xi = pos[:,0]
yi = pos[:,1]

u_i = 0.25 * torch.exp(-((xi - 0.3)**2 + (yi - 0.2)**2) / 0.1) + \
         0.4 * torch.exp(-((xi + 0.5)**2 + (yi + 0.1)**2) * 15) + \
         0.3 * torch.exp(-(xi**2 + (yi + 0.5)**2) * 20)

primes = torch.nn.Parameter(u_i.clone().to(device).unsqueeze(-1).float())
u_old = primes.clone()

t=0.0
for epoch in range(n_epochs+1):
    start = time.time()
    optimizer =  torch.optim.LBFGS([primes], lr=1.0, max_iter=100,history_size=20, line_search_fn='strong_wolfe')
    closure_results = {}
    
    # 定义closure函数（必需！）
    def closure():
        optimizer.zero_grad() 
        
        loss_pde, _ = FD_discretize(
            original_u=primes, u_old=u_old,
            graph_node=graph_node, graph_edge_xi=graph_extended_edge_xi,
            graph_edge_eta=graph_extended_edge_eta, graph_block_cell=graph_extended_cell,
            graph_extended=graph_extended_node, graph_Index=graph_Index,params=params
        )
        
        loss_batch = global_mean_pool(loss_pde**2, batch=graph_node.batch).view(-1)

        loss = torch.mean(torch.log(loss_batch))
        
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
                                           graph_edge_eta=graph_extended_edge_eta, graph_block_cell=graph_extended_cell,
                                           graph_extended=graph_extended_node, graph_Index=graph_Index,
                                            params=params)
        u_to_vis = u_final
        
    u_old = u_to_vis[:,0:1].clone()
    primes = torch.nn.Parameter(u_to_vis.clone())


        # perform optimization step
    learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]

    # --- Evaluation and Tecplot output at each time step ---
    dt = 0.01 # Time step size, assuming 100 steps from t=0 to t=1.0
    
    # 输出此epoch的处理时间
    print(f"Epoch {epoch} completed in {time.time() - start:.2f} seconds")
    
    # Print the headers and values
    if epoch%1 == 0:

        headers = ["Epoch", " Loss", "Learning Rate", "Epoch Time","PDE Loss"]
        values = [
            epoch,
            f"{closure_results['loss'] :.4e}",
            f"{learning_rate:.2e}",
            f"{time.time() - start:.2f}s",
            f"{closure_results['pde_loss'] :.4e}"
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
                                 physical_time=t, time_step=epoch)

    t += dt
