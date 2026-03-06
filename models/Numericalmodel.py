
import torch.nn as nn
from utils.utilities import NodeType
import torch
from scipy.spatial import KDTree


class FD_discretizer(nn.Module):
    def __init__(self ) :
        super(FD_discretizer,self).__init__()

    @torch.compiler.disable
    def hard_enforce_BC(self, uvp, graph_node, graph_extended):
        # 先clone一份,避免破坏计算图
        dummy_node_uvp = uvp.clone()
        mask_bc = ((graph_node.node_type == NodeType.INFLOW) | 
                (graph_node.node_type == NodeType.WALL))
        out_mask = graph_node.node_type == NodeType.OUTFLOW

        # 在clone的副本上做就地修改（不影响原始uvp的梯度）
        dummy_node_uvp[mask_bc, 0:2] = graph_node.y[mask_bc, 0:2]
        dummy_node_uvp[out_mask, 2:3] = 0.

        extend_index = graph_node.extend_index
        dummy_extended_uvp = dummy_node_uvp[extend_index]
        
        # 同样先clone再修改
        extended_uvp = uvp[extend_index].clone()
        # 硬施加
        # mask_bc_extended = ((graph_extended.node_type == NodeType.INFLOW) | 
        #                    (graph_extended.node_type == NodeType.WALL))
        # extended_uvp[mask_bc_extended, 0:2] = dummy_extended_uvp[mask_bc_extended, 0:2]
        
        # out_mask_extended = graph_extended.node_type == NodeType.OUTFLOW
        # extended_uvp[out_mask_extended, 2:3] = 0.
        
        # ghost节点处理
        boundary_ghost_stencil_index = graph_extended.boundary_ghost_stencil_index
        ghost_bc_nodes = boundary_ghost_stencil_index[:, 0]
        ghost_bc_node_type = graph_extended.node_type[ghost_bc_nodes]

        ghost_uv_neumann_mask = ghost_bc_node_type == NodeType.OUTFLOW
        ghost_p_neumann_mask = ((ghost_bc_node_type == NodeType.WALL) | 
                                 (ghost_bc_node_type == NodeType.INFLOW))

        ghost_uv_neumann_stencil = boundary_ghost_stencil_index[ghost_uv_neumann_mask]
        ghost_p_neumann_stencil = boundary_ghost_stencil_index[ghost_p_neumann_mask]
        #线性外推
        extended_uvp[ghost_p_neumann_stencil[:, 0], 0:2] = 2* dummy_extended_uvp[ghost_p_neumann_stencil[:, 1], 0:2] - \
                                                           extended_uvp[ghost_p_neumann_stencil[:, 2], 0:2]
        extended_uvp[ghost_uv_neumann_stencil[:, 0], 2:3] = 2* dummy_extended_uvp[ghost_uv_neumann_stencil[:, 1], 2:3] - \
                                                           extended_uvp[ghost_uv_neumann_stencil[:, 2], 2:3]
        extended_uvp[ghost_uv_neumann_stencil[:, 0], 0:2] = extended_uvp[ghost_uv_neumann_stencil[:, 2], 0:2]
        
        
        extended_uvp[ghost_p_neumann_stencil[:, 0], 2:3] = extended_uvp[ghost_p_neumann_stencil[:, 2], 2:3]
        
        # 硬施加
        # extended_uvp[ghost_uv_neumann_stencil[:, 0], 0:2] = (
        #     4 * extended_uvp[ghost_uv_neumann_stencil[:, 1], 0:2] - 
        #     extended_uvp[ghost_uv_neumann_stencil[:, 2], 0:2]
        # ) / 3
        
        # extended_uvp[ghost_p_neumann_stencil[:, 0], 2:3] = (
        #     4 * extended_uvp[ghost_p_neumann_stencil[:, 1], 2:3] - 
        #     extended_uvp[ghost_p_neumann_stencil[:, 2], 2:3]
        # ) / 3
        return extended_uvp
    
    def forward(self, original_uv=None,uv_old = None, graph_node=None,graph_edge_xi=None,graph_edge_eta=None,graph_block_cell=None,graph_extended=None,graph_Index=None,params=None,smooth=True):
   
        if uv_old is not None:

            extended_uvp_old = self.hard_enforce_BC(uv_old,graph_node,graph_extended)
        else:
            extended_uvp_old = None
       
        extended_uvp = self.hard_enforce_BC(original_uv,graph_node,graph_extended)

        #########for cavity pressure point##########

        pressure_point = graph_extended.node_type==NodeType.PRESS_POINT

        extended_uvp[pressure_point,2]=0.

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
        continuity_eq_coefficent = pde_theta_node[:, 1:2]
        convection_coefficent = pde_theta_node[:, 2:3]
        grad_p_coefficent = pde_theta_node[:, 3:4]
        diffusion_coefficent = pde_theta_node[:, 4:5]

        relaxtion = graph_Index.relaxtion[graph_node.batch]
  
        ####get metrics on padded blocks###
        metrics_extended = graph_extended.extended_block_metrics  # shape: (N, 5)
        dxi_dx = metrics_extended[:,0]
        dxi_dy = metrics_extended[:,1]
        deta_dx = metrics_extended[:,2]
        deta_dy = metrics_extended[:,3]
        J = metrics_extended[:,4]  



        # 向量化变换速度计算
        uv_extended = extended_uvp[:,:2]  # shape: (N, 2)
        p_extended = extended_uvp[:,2]
        
        # 向量化计算变换速度 U = u*dxi_dx + v*dxi_dy, V = u*deta_dx + v*deta_dy
        transformation_matrix = torch.stack([
            torch.stack([dxi_dx, dxi_dy], dim=1),   # xi方向变换
            torch.stack([deta_dx, deta_dy], dim=1)  # eta方向变换
        ], dim=1)  # shape: (N, 2, 2)
        
        UV = torch.bmm(transformation_matrix, uv_extended.unsqueeze(-1)).squeeze(-1)  # shape: (N, 2)
        U, V = UV[:,0], UV[:,1]

        # 向量化面心插值
        U_face = 0.5*((U/J)[l_node]+(U/J)[r_node])
        V_face = 0.5*((V/J)[d_node]+(V/J)[u_node])
          
        loss_cont = ((((U_face[r_edge]-U_face[l_edge]+V_face[u_edge]-V_face[d_edge])).unsqueeze(-1)*continuity_eq_coefficent))
        # 向量化计算对流和扩散通量
        if extended_uvp_old is not None: 
            convect_flux_old = self.convect_flux(extended_uvp_old[:,0:2],dxi_dx,dxi_dy,deta_dx,deta_dy,J,l_node,r_node,d_node,u_node,l_edge,r_edge,d_edge,u_edge)
            convect_flux_old_tensor = torch.cat(convect_flux_old, dim=1)  # shape: (N, 4)

        convect_flux_new = self.convect_flux(extended_uvp[:,0:2],dxi_dx,dxi_dy,deta_dx,deta_dy,J,l_node,r_node,d_node,u_node,l_edge,r_edge,d_edge,u_edge)
        diffuse_flux_new = self.diffuse_flux(extended_uvp[:,0:2],dxi_dx,dxi_dy,deta_dx,deta_dy,J,l_node,r_node,d_node,u_node,block_cells_node,l_cell,r_cell,d_cell,u_cell,l_edge,r_edge,d_edge,u_edge)

        # 真正的向量化松弛计算
        # 将通量组织成张量进行批量计算
        convect_flux_new_tensor = torch.cat(convect_flux_new, dim=1)  # shape: (N, 4)
        diffuse_flux_tensor = torch.cat(diffuse_flux_new, dim=1)  # shape: (N, 4)
        
        # 向量化松弛计算 - 一次性计算所有通量
        if extended_uvp_old is not None:
            convect_flux_tensor = convect_flux_old_tensor * relaxtion + convect_flux_new_tensor * (1 - relaxtion)
        else:
            convect_flux_tensor = convect_flux_new_tensor
     
        # 解包回单独变量
        dE1u, dE1v, dE2u, dE2v = convect_flux_tensor.unbind(dim=1)
        dEv_1_u, dEv_1_v, dEv_2_u, dEv_2_v = diffuse_flux_tensor.unbind(dim=1)


        ####################################################################################################
        # 向量化压力梯度计算
        # 计算压力在计算坐标系中的导数
        #Calculating Gradient of Pressure
        p_xi_x = p_extended*dxi_dx/J
        p_xi_y = p_extended*dxi_dy/J
        p_eta_x = p_extended*deta_dx/J
        p_eta_y = p_extended*deta_dy/J

        p_xi_x_face = 0.5*(p_xi_x[l_node]+p_xi_x[r_node])
        p_xi_y_face = 0.5*(p_xi_y[l_node]+p_xi_y[r_node])
        p_eta_x_face = 0.5*(p_eta_x[d_node]+p_eta_x[u_node])
        p_eta_y_face = 0.5*(p_eta_y[d_node]+p_eta_y[u_node])

        grad_P_x = ((p_xi_x_face[r_edge]-(p_xi_x_face)[l_edge])+((p_eta_x_face)[u_edge]-(p_eta_x_face)[d_edge])).unsqueeze(-1)
        grad_P_y = (((p_eta_y_face)[u_edge]-(p_eta_y_face)[d_edge])+((p_xi_y_face)[r_edge]-(p_xi_y_face)[l_edge])).unsqueeze(-1)
                                        
        # 向量化RHS计算
        # 合并对流和扩散项
        convect = torch.stack([dE1u + dE2u, dE1v + dE2v], dim=1)  # shape: (N, 2)
        diffusion = torch.stack([dEv_1_u + dEv_2_u, dEv_1_v + dEv_2_v], dim=1)  # shape: (N, 2)
        
        # 向量化非定常项计算
        if extended_uvp_old is not None:
            unsteady = ((original_uv[:,:2] - uv_old[:,:2]) / dt_node) / J_o.unsqueeze(-1)  # shape: (N, 2)
        
        # 向量化压力梯度
        grad_P = torch.cat([grad_P_x, grad_P_y], dim=1)  # shape: (N, 2)
        
        # 向量化动量方程损失计算,守恒型差分，约等于有限体积
        if extended_uvp_old is not None:
            momentum = (unsteady_coefficent * unsteady + 
                       convection_coefficent * convect + 
                       grad_p_coefficent * grad_P - 
                       diffusion_coefficent * diffusion)  # shape: (N, 2)

        else:
            momentum = (
                        convection_coefficent * convect + 
                        grad_p_coefficent * grad_P - 
                        diffusion_coefficent * diffusion)  # shape: (N, 2)
        loss_mom_x = momentum[:, 0:1]
        loss_mom_y = momentum[:, 1:2]
        if smooth:
            cell_uvp = ((extended_uvp[block_cells_node[0]]+extended_uvp[block_cells_node[1]]+extended_uvp[block_cells_node[2]]+extended_uvp[block_cells_node[3]])/4)
            extended_uvp_face_eta = 0.5*(cell_uvp[l_cell]+cell_uvp[r_cell])
            uvp_to_vis = (0.5*(extended_uvp_face_eta[u_edge]+extended_uvp_face_eta[d_edge])).clone().detach()
            # cell_p = ((extended_uvp[block_cells_node[0],2:3]+extended_uvp[block_cells_node[1],2:3]+extended_uvp[block_cells_node[2],2:3]+extended_uvp[block_cells_node[3],2:3])/4)
            # extended_p_face_eta = 0.5*(cell_p[l_cell]+cell_p[r_cell])
            # uvp_to_vis_p = (0.5*(extended_p_face_eta[u_edge]+extended_p_face_eta[d_edge])).clone().detach()
            # uvp_to_vis = torch.cat([original_uv[:,:2].clone().detach(),uvp_to_vis_p],dim=1)
        else:
            uvp_to_vis = original_uv.clone().detach()
        return loss_cont,loss_mom_x,loss_mom_y,uvp_to_vis
 
        
    def convect_flux(self,extended_uv_hat,dxi_dx,dxi_dy,deta_dx,deta_dy,J,l_node,r_node,d_node,u_node,l_edge,r_edge,d_edge,u_edge):
        
        u_hat = extended_uv_hat[:,0]
        v_hat = extended_uv_hat[:,1]
        
        ####################correct convect#####################
        U_hat= u_hat*dxi_dx+v_hat*dxi_dy 
        V_hat = u_hat*deta_dx+v_hat*deta_dy 

        U_face = 0.5*((U_hat/J)[l_node]+(U_hat/J)[r_node])
        V_face = 0.5*((V_hat/J)[d_node]+(V_hat/J)[u_node])

        extended_uv_face_xi = 0.5*(extended_uv_hat[l_node]+extended_uv_hat[r_node])
        extended_uv_face_eta = 0.5*(extended_uv_hat[d_node]+extended_uv_hat[u_node])

        #######################central scheme##############################
        face_e_flux_hat = extended_uv_face_xi[r_edge,0:2]*U_face[r_edge].unsqueeze(-1)
    
       
        face_w_flux_hat = extended_uv_face_xi[l_edge,0:2]*U_face[l_edge].unsqueeze(-1)

       
        face_n_flux_hat =  extended_uv_face_eta[u_edge,0:2]*V_face[u_edge].unsqueeze(-1)

        face_s_flux_hat = extended_uv_face_eta[d_edge,0:2]*V_face[d_edge].unsqueeze(-1)
        
        #######################central scheme##############################
        dE1u = (face_e_flux_hat[:,0]-face_w_flux_hat[:,0]).unsqueeze(-1)

        dE1v = (face_e_flux_hat[:,1]-face_w_flux_hat[:,1]).unsqueeze(-1)

        dE2u =(face_n_flux_hat[:,0]-face_s_flux_hat[:,0]).unsqueeze(-1)

        dE2v =(face_n_flux_hat[:,1]-face_s_flux_hat[:,1]).unsqueeze(-1)

        return dE1u,dE1v,dE2u,dE2v


    def convect_flux_noncons(self,extended_uv_hat,original_uv_hat,dxi_dx,dxi_dy,deta_dx,deta_dy,J,l_node,r_node,d_node,u_node,l_edge,r_edge,d_edge,u_edge):
        
        node_flux_x_xi = extended_uv_hat*(dxi_dx/J).unsqueeze(-1)

        node_flux_x_eta = extended_uv_hat*(deta_dx/J).unsqueeze(-1)

        node_flux_y_xi = extended_uv_hat*(dxi_dy/J).unsqueeze(-1)

        node_flux_y_eta = extended_uv_hat*(deta_dy/J).unsqueeze(-1)

        face_flux_x_xi = 0.5*(node_flux_x_xi[l_node]+node_flux_x_xi[r_node])

        face_flux_x_eta = 0.5*(node_flux_x_eta[d_node]+node_flux_x_eta[u_node])

        face_flux_y_xi = 0.5*(node_flux_y_xi[l_node]+node_flux_y_xi[r_node])

        face_flux_y_eta = 0.5*(node_flux_y_eta[d_node]+node_flux_y_eta[u_node])
        
        #######################central scheme##############################

        dflux_x_xi = face_flux_x_xi[r_edge]-face_flux_x_xi[l_edge]

        dflux_x_eta = face_flux_x_eta[u_edge]-face_flux_x_eta[d_edge]

        dflux_y_xi = face_flux_y_xi[r_edge]-face_flux_y_xi[l_edge]

        dflux_y_eta = face_flux_y_eta[u_edge]-face_flux_y_eta[d_edge]

        dE1u = original_uv_hat[:,0:1]*(dflux_x_xi[:,0:1]+dflux_x_eta[:,0:1])

        dE2u = original_uv_hat[:,1:2]*(dflux_y_xi[:,0:1]+dflux_y_eta[:,0:1])

        dE1v = original_uv_hat[:,0:1]*(dflux_x_xi[:,1:2]+dflux_x_eta[:,1:2])

        dE2v = original_uv_hat[:,1:2]*(dflux_y_xi[:,1:2]+dflux_y_eta[:,1:2])

        return dE1u,dE1v,dE2u,dE2v


    def diffuse_flux(self,extended_uv_hat,dxi_dx,dxi_dy,deta_dx,deta_dy,J,l_node,r_node,d_node,u_node,block_cells_node,l_cell,r_cell,d_cell,u_cell,l_edge,r_edge,d_edge,u_edge):
        u_hat = extended_uv_hat[:,0]
        v_hat = extended_uv_hat[:,1]
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
        dv_dxi_on_xi = (v_hat[r_node]-v_hat[l_node]).unsqueeze(-1)

        cells_u = ((u_hat[block_cells_node[0]]+u_hat[block_cells_node[1]]+u_hat[block_cells_node[2]]+u_hat[block_cells_node[3]])/4).unsqueeze(-1)
        cells_v = ((v_hat[block_cells_node[0]]+v_hat[block_cells_node[1]]+v_hat[block_cells_node[2]]+v_hat[block_cells_node[3]])/4).unsqueeze(-1)

        du_deta_on_xi = cells_u[u_cell]-cells_u[d_cell]
        dv_deta_on_xi = cells_v[u_cell]-cells_v[d_cell]

        du_deta_on_eta = (u_hat[u_node]-u_hat[d_node]).unsqueeze(-1)
        dv_deta_on_eta = (v_hat[u_node]-v_hat[d_node]).unsqueeze(-1)

        du_dxi_on_eta = cells_u[r_cell]-cells_u[l_cell]
        dv_dxi_on_eta = cells_v[r_cell]-cells_v[l_cell]

        Ev_1_u = (g11_half*du_dxi_on_xi/J_half_x)+(g12_half*du_deta_on_xi/J_half_x)
        Ev_1_v = (g11_half*dv_dxi_on_xi/J_half_x)+(g12_half*dv_deta_on_xi/J_half_x)


        Ev_2_u = (g22_half*du_deta_on_eta/J_half_y)+(g21_half*du_dxi_on_eta/J_half_y)
        Ev_2_v = (g22_half*dv_deta_on_eta/J_half_y)+(g21_half*dv_dxi_on_eta/J_half_y)

   
        dEv_1_u = (Ev_1_u[r_edge]-Ev_1_u[l_edge])
        dEv_1_v = (Ev_1_v[r_edge]-Ev_1_v[l_edge])
        
        dEv_2_u = (Ev_2_u[u_edge]-Ev_2_u[d_edge])
        dEv_2_v = (Ev_2_v[u_edge]-Ev_2_v[d_edge])

        return dEv_1_u,dEv_1_v,dEv_2_u,dEv_2_v






