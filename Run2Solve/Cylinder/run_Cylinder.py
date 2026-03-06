
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import torch
from torch.optim import Adam
import numpy as np

import get_param
import time
from get_param import get_hyperparam
from utils.Logger import Logger
from SOAPopt import SOAP
from utils.scheduler import ProgressiveRestartCosineAnnealingLR
from utils.utilities import NodeType

from models.Numericalmodel import FD_discretizer

from torch_geometric.nn import global_add_pool,global_mean_pool

from dataset import Graph_loader

import random
import datetime
from scipy.spatial import KDTree
import subprocess

from torch_scatter import scatter_mean
import json

def create_optimizer(params_to_optimize, optimizer_type, lr):
    """创建优化器"""
    if optimizer_type == "ADAM":
        optimizer = Adam(params_to_optimize, lr=lr)
    elif optimizer_type == "SOAP":
        optimizer = SOAP(
            params_to_optimize,
            lr=lr,
            betas=(0.95, 0.95),
            weight_decay=0,
            precondition_frequency=10,
            max_precond_dim=10000,
            precondition_1d=False,
            normalize_grads=False,
            eps=1e-8,
            correct_bias=True,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def solve_time_step(
    primes,
    uv_old,
    FD_discretize,
    graph_data,
    params,
    optimizer_type,
    lr,
    max_iter,
    tolerance,
    logger,
    time_step,
    physical_time,
    loss_init,
    print_interval=100,
):
    graph_node, graph_extended_edge_xi, graph_extended_edge_eta, \
        graph_extended_cell, graph_extended_node, graph_Index = graph_data

    # 创建优化器和学习率调度器
    optimizer = create_optimizer([primes], optimizer_type, lr)
    scheduler = ProgressiveRestartCosineAnnealingLR(optimizer, 
                                                    window_size = 2000,
                                                    total_windows = 5,
                                                    initial_max_lr = lr,
                                                    decay_factor = 0.5,
                                                    min_restart_lr = 1e-4,
                                                    eta_min = 5e-5)     # 提高最小学习率

    converged = False
    for n_iter in range(max_iter):
        optimizer.zero_grad()

        # 前向计算PDE残差
        loss_cont, loss_momentum_x, loss_momentum_y, _ = FD_discretize(
            original_uv=primes,
            uv_old=uv_old,
            graph_node=graph_node,
            graph_edge_xi=graph_extended_edge_xi,
            graph_edge_eta=graph_extended_edge_eta,
            graph_block_cell=graph_extended_cell,
            graph_extended=graph_extended_node,
            graph_Index=graph_Index,
            params=params
        )
        loss_cont = global_mean_pool(loss_cont**2,batch = graph_node.batch).view(-1)
        loss_mom = global_mean_pool(loss_momentum_x**2,batch = graph_node.batch).view(-1)+global_mean_pool(loss_momentum_y**2,batch = graph_node.batch).view(-1)
        loss_batch = (     
                6e4 * loss_cont
                + 5e4 * loss_mom
            )
        loss = torch.mean(torch.log(loss_batch))

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        if logger is not None:
            logger.log_residuals(
                time_step=time_step,
                physical_time=physical_time,
                inner_iter=n_iter,
                loss_cont=loss_cont.mean().item(),
                loss_mom=loss_mom.mean().item(),
                learning_rate=current_lr,
            )

        if n_iter % print_interval == 0 or n_iter == max_iter - 1:
            logger.info(f"  Time = {physical_time:.4f}s | Iter {n_iter:4d} | "
                       f"Cont: {loss_cont.mean().item():.4e} | loss_mom: {loss_mom.mean().item():.4e} |"
                       f"lr: {current_lr:.2e}")

    # 获取最终解
    with torch.no_grad():
        _, _, _, uvp_final = FD_discretize(
            original_uv=primes,
            uv_old=uv_old,
            graph_node=graph_node,
            graph_edge_xi=graph_extended_edge_xi,
            graph_edge_eta=graph_extended_edge_eta,
            graph_block_cell=graph_extended_cell,
            graph_extended=graph_extended_node,
            graph_Index=graph_Index,
            params=params
        )

    residuals = {'cont': loss_cont.mean().item(), 'mom': loss_mom.mean().item()}
    return uvp_final, converged, residuals, loss_init, n_iter + 1


def main():
    # ============== 参数配置 ==============
    params = get_param.params()

    torch.set_float32_matmul_precision('high')

    # 随机种子
    seed = 42  # 固定种子以便复现
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_per_process_memory_fraction(0.99, params.on_gpu)
    torch.set_num_threads(os.cpu_count() // 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_path = "Comsol_ref/Cylinder/Cylinder-Re40.dat"

    # ============== 初始化Logger ==============
    logger = Logger(
        get_hyperparam(params),
        use_dat=True,
        params=params,
        copy_code=True,
    )

    logger.info(f"Using device: {device}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Dataset directory: {params.dataset_dir}")
    logger.info(f"Logger saving path: {logger.saving_path}")

    # ============== 加载数据集 ==============
    start = time.time()
    datasets_factory = Graph_loader.DatasetFactory(
        params=params,
        dataset_dir=params.dataset_dir,
        state_save_dir=logger.saving_path,
        device=device,
    )

    params.dataset_size = datasets_factory.dataset_size

    datasets, loader, _ = datasets_factory.create_datasets(
        batch_size=params.batch_size, num_workers=0, pin_memory=False
    )

    logger.info(f"Dataset loaded in {time.time() - start:.2f}s")

    # ============== 初始化模型 ==============
    FD_discretize = torch.compile(FD_discretizer(),dynamic=True).to(device)
    # 获取图数据
    graph_data = [data.to(device) for data in next(iter(loader))]
    graph_Index = graph_data[-1]
    graph_node = graph_data[0]
    dt = graph_Index.dt_graph[0, 0].item() if hasattr(graph_Index, 'dt_graph') else 0.01

    mesh = datasets.meta_pool[0] 
    n_time_steps = params.n_epochs

    # max_iter: 每个时间步内最大迭代次数（类似maxIter）
    max_iter = params.max_inner_steps

    # tolerance: 收敛阈值（时间步内相对残差，类似relTol）
    tolerance_cont = params.convergence_tol_cont
    tolerance_mom = params.convergence_tol_mom
    # print_interval: 打印间隔
    print_interval = max(1, max_iter // 10)  # 每个时间步打印约10次

    # write_interval: 结果输出间隔（每N个时间步输出一次，来自命令行参数）
    write_interval = params.write_interval

    # ============== 初始化场变量 ==============
    init_uv = graph_node.y.clone().to(device)
    init_p = torch.zeros_like(init_uv[:, 0:1])
    init_uvp = torch.cat((init_uv, init_p), dim=-1)

    # 当前时间步的待优化变量
    primes = torch.nn.Parameter(init_uvp.clone())

    # 上一时间步的速度场（初始时刻等于初始条件）
    uv_old = init_uv.clone()

    # ============== 优化器配置 ==============
    OPT = "SOAP"  # "ADAM" or "SOAP"

    # ============== 打印求解器配置（类似OpenFOAM启动信息） ==============
    logger.info("=" * 70)
    logger.info("FDGO Solver Configuration (OpenFOAM-style)")
    logger.info("=" * 70)
    logger.info("controlDict:")
    logger.info(f"  deltaT        : {dt}")
    logger.info(f"  endTime       : {dt * n_time_steps:.4f} ({n_time_steps} time steps)")
    logger.info(f"  writeInterval : {write_interval} time steps")
    logger.info("")
    logger.info("fvSolution:")
    logger.info(f"  optimizer     : {OPT}")
    logger.info(f"  learning rate : {params.lr}")
    logger.info(f"  maxIter       : {max_iter}")
    logger.info("=" * 70)

    # ============== 时间推进主循环 ==============
    loss_init = {'cont': None, 'mom': None}
    physical_time = 0.0
    total_inner_iters = 0

    for time_step in range(n_time_steps):
        
        step_start = time.time()
        
        physical_time=dt*time_step
        # ============== 记录求解前的 uv 值 ==============
        uv_t0 = primes[:, 0:2].detach().clone()
        p_t0 = primes[:, 2:3].detach().clone()
        # ============== 求解当前时间步 ==============
        uvp_converged, converged, residuals, loss_init, n_iter = solve_time_step(
            primes=primes,
            uv_old=uv_old,
            FD_discretize=FD_discretize,
            graph_data=graph_data,
            params=params,
            optimizer_type=OPT,
            lr=params.lr,
            max_iter=max_iter,
            tolerance=[tolerance_cont,tolerance_mom],
            logger=logger,
            time_step=time_step,
            physical_time=physical_time,
            loss_init=loss_init,
            print_interval=print_interval,
        )

        total_inner_iters += n_iter
        step_time = time.time() - step_start
        physical_time = step_time
        logger.info("")
        logger.info(f"Time = {physical_time:.4f}s (step {time_step + 1}/{n_time_steps})")
        logger.info("-" * 50)
        
        # ============== 检查收敛并推进时间步 ==============
        if not converged:
            logger.warning(f"  Time step {time_step + 1} did not converge after {n_iter} iterations!")
            logger.warning(f"  Consider increasing maxIter or adjusting learning rate.")

        # 时间推进：更新到下一时间步
        uv_old = uvp_converged[:, 0:2].clone()
        primes = torch.nn.Parameter(uvp_converged.clone())
        #计算流场误差      
        reduce_index = mesh["reduce_index"]
        if not isinstance(reduce_index, torch.Tensor):
            reduce_index = torch.tensor(reduce_index, dtype=torch.long).to(device)
        else:
            reduce_index = reduce_index.to(device)
        mesh_pos_unique_size = mesh["mesh_pos_unique"].shape[0]
        uvp_old_unique = scatter_mean(uvp_converged.clone(), reduce_index, dim=0, dim_size=mesh_pos_unique_size)

        # ============== 时间步总结 ==============
        status = "Converged" if converged else "NOT CONVERGED"
        logger.info(f"Time = {physical_time:.4f}s | Iters: {n_iter:4d} | "
                   f"Cont: {residuals['cont']:.4e} | Mom: {residuals['mom']:.4e} | "
                   f"Wall time: {step_time:.2f}s | {status}")
        # ============== 定期输出指定时刻结果 ==============
        if (time_step) % 1 == 0 or time_step == n_time_steps - 1:
            pos_all = graph_node.pos.cpu().numpy()
            pos_unique = torch.tensor(mesh["mesh_pos_unique"], dtype=torch.float32).to(device)
            pos_tree = KDTree(pos_all)
            _, unique_pos_indices = pos_tree.query(pos_unique.cpu().numpy(), k=1)
            global_idx = graph_node.global_idx.cpu()[unique_pos_indices]
            batch_unique = graph_node.batch.cpu()[unique_pos_indices]

            uvp_dim = graph_Index.uvp_dim[batch_unique]  # shape: [N, 3]
            u_dim = uvp_old_unique[:, 0:1] * uvp_dim[:, 0:1]
            v_dim = uvp_old_unique[:, 1:2] * uvp_dim[:, 1:2]
            p_dim = uvp_old_unique[:, 2:3] * uvp_dim[:, 2:3]
            dim_predicted_uvp_node = torch.cat([u_dim, v_dim, p_dim], dim=1)
            datasets.payback_uvp_for_vis(
                dim_predicted_uvp_node.detach().cpu(),
                global_idx,
                params.dataset_size,
                batch_unique,
                physical_time=time_step,
                time_step=time_step,
                to_export = True
            )
            logger.info(f"  -> Results written at Time = {physical_time:.4f}s")
    # ============== 模拟完成 ==============
    logger.finalize_residuals()

    logger.info("")
    logger.info("=" * 70)
    logger.info("Simulation completed!")
    logger.info(f"  End Time      : {physical_time:.4f}s")
    logger.info(f"  Time Steps    : {n_time_steps}")
    logger.info(f"  Total Iters   : {total_inner_iters}")
    logger.info(f"  Avg Iters/Step: {total_inner_iters / n_time_steps:.1f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
