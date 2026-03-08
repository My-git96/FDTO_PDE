import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import torch
from torch.optim import Adam
import numpy as np
from models.Numericalmodel import FD_discretizer

import get_param
import time
from get_param import get_hyperparam
from utils.Logger import Logger

import datetime as dt

from torch_geometric.nn import global_add_pool,global_mean_pool

from dataset import Graph_loader

import random
import datetime
from scipy.spatial import KDTree
import subprocess

def init_field(init_field_path, graph_node, device):
    try:
        # 读取Tecplot格式的初始条件文件
        with open(init_field_path, 'r') as f:
            lines = f.readlines()
        
        # 解析ZONE信息获取节点数和单元数
        nodes = None
        elements = None
        data_start_idx = None
        
        for idx, line in enumerate(lines):
            if 'Nodes=' in line:
                # 格式: Nodes=40401, Elements=40000, ZONETYPE=FEQuadrilateral
                parts = line.split(',')
                for part in parts:
                    if 'Nodes=' in part:
                        nodes = int(part.split('Nodes=')[1].strip())
                    if 'Elements=' in part:
                        elements = int(part.split('Elements=')[1].strip())
            
            if 'DATAPACKING=BLOCK' in line:
                # BLOCK格式：数据从下一行开始
                data_start_idx = idx + 1
                # 跳过DT行
                if data_start_idx < len(lines) and 'DT=' in lines[data_start_idx]:
                    data_start_idx += 1
                break
        
        if data_start_idx is None or nodes is None:
            raise ValueError("Could not find data start or Nodes count in tecplot file")
        
        # 读取数据块（BLOCK格式）
        # X, Y, U, V, P各一块，共nodes*5个值
        data_values = []
        value_count = 0
        expected_values = nodes * 5
        
        for line in lines[data_start_idx:]:
            line = line.strip()
            if line and not line.startswith('ZONE'):
                try:
                    values = [float(x) for x in line.split()]
                    data_values.extend(values)
                    value_count += len(values)
                    
                    # 只读取节点数据，不读取单元数据
                    if value_count >= expected_values:
                        break
                except ValueError:
                    # 跳过无法解析的行
                    continue
        
        # 检查数据长度
        if len(data_values) < expected_values:
            raise ValueError(f"Data too short: {len(data_values)} < {expected_values}")
        
        # 取前expected_values个值（只要节点数据）
        data_values = data_values[:expected_values]
        
        # BLOCK格式：分解数据
        X = np.array(data_values[0*nodes : 1*nodes], dtype=np.float32)
        Y = np.array(data_values[1*nodes : 2*nodes], dtype=np.float32)
        U = np.array(data_values[2*nodes : 3*nodes], dtype=np.float32)
        V = np.array(data_values[3*nodes : 4*nodes], dtype=np.float32)
        P = np.array(data_values[4*nodes : 5*nodes], dtype=np.float32)

        # 创建KDTree进行坐标匹配，以防顺序不同
        init_coords = np.column_stack([X, Y])
        graph_coords = graph_node.pos.cpu().numpy()
        
        kd_tree = KDTree(init_coords)
        distances, indices = kd_tree.query(graph_coords, k=1)
        
        # 检查匹配质量
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        if max_dist > 1e-6:
            print(f"Warning: Maximum coordinate distance in matching: {max_dist:.2e}")
            print(f"         Mean coordinate distance: {mean_dist:.2e}")
        
        # 按匹配的顺序重新排列数据
        U_matched = U[indices]
        V_matched = V[indices]
        P_matched = P[indices]
        
        # 初始化primes
        primes_init = np.zeros((graph_node.pos.shape[0], 3), dtype=np.float32)
        primes_init[:, 0] = U_matched
        primes_init[:, 1] = V_matched
        primes_init[:, 2] = P_matched
        
        primes = torch.nn.Parameter(
            torch.from_numpy(primes_init).to(device=device, dtype=torch.float32)
        )
    except FileNotFoundError:
        print(f"Warning: Initial field file not found at {init_field_path}")
        print("Using zero initialization instead")
        primes = torch.nn.Parameter(torch.zeros(graph_node.pos.shape[0], 3, device=device, dtype=torch.float32))
    except Exception as e:
        print(f"Error loading initial field: {e}")
        print("Using zero initialization instead")
        import traceback
        traceback.print_exc()
        primes = torch.nn.Parameter(torch.zeros(graph_node.pos.shape[0], 3, device=device, dtype=torch.float32))
    
    return primes

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
# 从初始条件文件读取流场初始化
init_field_path = "TestCase/Cavity/initial_field/Cavity201_Re3200_Re=3200.dat"
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

FD_discretize = FD_discretizer()

n_epochs = params.n_epochs

graph_node,graph_extended_edge_xi,graph_extended_edge_eta,graph_extended_cell,graph_extended_node,graph_Index = [data.to(device) for data in next(iter(loader))]

primes = init_field(init_field_path,graph_node,device)
lr_scheduler = None

uv_old = primes[:, 0:2].clone().detach()  # 使用初始流场作为前一时间步

for epoch in range(n_epochs):

    optimizer =  torch.optim.LBFGS([primes], lr=1e-5, max_iter=100,history_size=20, line_search_fn='strong_wolfe')
    
    start = time.time()

    closure_results = {}
    
    # 定义closure函数（必需！）
    def closure():
        optimizer.zero_grad()
        
        loss_cont, loss_mom_x, loss_mom_y, uvp_to_vis = FD_discretize(
            original_uv=primes, uv_old=uv_old,
            graph_node=graph_node, graph_edge_xi=graph_extended_edge_xi,
            graph_edge_eta=graph_extended_edge_eta, graph_block_cell=graph_extended_cell,
            graph_extended=graph_extended_node, graph_Index=graph_Index,params=params
        )
        
        loss_cont = global_mean_pool(loss_cont**2, batch=graph_node.batch).view(-1)
        loss_mom = global_mean_pool(loss_mom_x**2, batch=graph_node.batch).view(-1) + \
                   global_mean_pool(loss_mom_y**2, batch=graph_node.batch).view(-1)
        
        loss_batch = 6e4*loss_cont + 5e4*loss_mom
        loss = torch.mean(torch.log(loss_batch+ 1e-30))
        
        loss.backward()
        closure_results['loss'] = loss.cpu().item()
        closure_results['pde_loss'] = (loss_batch).mean().cpu().item()
        return loss
    
    # 单次调用，L-BFGS内部会多次迭代
    optimizer.step(closure)

    # epoch结束，用收敛后的结果更新下一时间步
    with torch.no_grad():
        # 重新计算最终的uvp_to_vis（使用优化后的primes）
        _, _, _, uvp_final = FD_discretize(original_uv=primes, uv_old=uv_old,
                                           graph_node=graph_node, graph_edge_xi=graph_extended_edge_xi,
                                           graph_edge_eta=graph_extended_edge_eta, graph_block_cell=graph_extended_cell,
                                           graph_extended=graph_extended_node, graph_Index=graph_Index,
                                            params=params)
        uvp_to_vis = uvp_final
    
    uv_old = uvp_to_vis[:,0:2].clone()
    primes = uvp_to_vis.clone().requires_grad_(True)


        # perform optimization step
    learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]

    # 输出此epoch的处理时间
    print(f"Epoch {epoch} completed in {time.time() - start:.2f} seconds")

    # Print the headers and values
    if epoch%1 == 0:
        epoch_time = time.time() - start
        headers = ["Epoch", " Loss", "Learning Rate", "Epoch Time","PDE Loss"]
        #headers = ["Epoch", " Loss", "Learning Rate", "Epoch Time","PDE Loss"]
        values = [
            epoch,
            f"{closure_results['loss'] :.4e}",          
            f"{learning_rate:.2e}",
            f"{time.time() - start:.2f}s",
            f"{closure_results['pde_loss'] :.4e}",
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
        datasets.payback_uvp_for_vis(uvp_to_vis.detach().cpu(), global_idx, params.dataset_size, batch, physical_time=epoch, time_step=epoch)
