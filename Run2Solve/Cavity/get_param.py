import argparse
import json
import itertools
import numpy as np

def str2bool(v):
    """
    'boolean type variable' for add_argument
    """
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')

def params(load=None):

    if load is not None:
        parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')
        params = vars(parser.parse_args())
        with open(load+'/commandline_args.json', 'rt') as f:
            params.update(json.load(f))
        for k, v in params.items():
            parser.add_argument('--' + k, default=v)
        args = parser.parse_args()
        return  args
    else:
        """
        return parameters for training / testing / plotting of models
        :return: parameter-Namespace
        """
        # #endregion
        parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')
        # Training parameters
        parser.add_argument('--problem', default="Forward", type=str, help='network to train', choices=["Forward","Inverse"])
        parser.add_argument('--n_epochs', default=1000, type=int, help='number of time steps to simulate (like endTime/dt)')
        parser.add_argument('--batch_size', default=1, type=int, help='batch size (default: 1)')
        parser.add_argument('--dataset_size', default=1, type=int, help='size of dataset (default: 1)')
        parser.add_argument('--average_sequence_length', default=300, type=int, help='average_sequence_length')
        parser.add_argument('--all_on_gpu', default=False, type=str2bool, help='whether put all dataset on GPU')
        parser.add_argument('--lr', default=1.0, type=float, help='learning rate of optimizer (default: 0.0005)')
        parser.add_argument('--use_lr_sche', default=False, type=str2bool, help='whether use_lr_sche')

        parser.add_argument('--log', default=True, type=str2bool, help='log models / metrics during training (turn off for debugging)')
        parser.add_argument('--rollout', default=False, type=str2bool, help='rolling out or not (turn off for debugging)')
        parser.add_argument('--on_gpu', default=0, type=int, help='set training on which gpu')
        parser.add_argument('--dimless', default=True, type=str2bool, help='dimless')
        parser.add_argument('--max_inner_steps', default=100, type=int, help='max iterations per time step (like fvSolution maxIter)')
        parser.add_argument('--convergence_tol_cont', default=1e-13, type=float, help='convergence tolerance for step-relative residual (like relTol)')
        parser.add_argument('--convergence_tol_mom', default=1e-13, type=float, help='convergence tolerance for momentum (like relTol)')
        parser.add_argument('--print_interval', default=50, type=int, help='print residuals every N inner iterations')
        parser.add_argument('--write_interval', default=1, type=int, help='write results every N time steps (like writeInterval in controlDict)')




       
        
        #dataset params
        
        parser.add_argument('--dataset_dir', default="./grid_example/Cavity201", type=str, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')
        # #endregion
        params = parser.parse_args()


        return params
            


def get_hyperparam(params):

    result = f"FDGO-{params.problem}"
    return result


def generate_list(min_val, step, max_val):
    if min_val == step == max_val:
        return [max_val]
    else:
        # 使用linspace可以确保开始和结束的值都包括在内
        # 并根据步长计算必要的点数
        num_points = int((max_val - min_val) / step) + 1
        return list(np.linspace(min_val, max_val, num_points))
    
def generate_combinations(
    U_range=None, rho_range=None, mu_range=None, source_range=None, aoa_range=None, dt=None, L=None
):

    U_list = generate_list(*U_range)
    rho_list = generate_list(*rho_range)
    mu_list = generate_list(*mu_range)
    source_list = generate_list(*source_range)
    aoa_list = generate_list(*aoa_range)
    
    combinations = list(itertools.product(U_list, rho_list, mu_list, source_list, aoa_list))

    valid_combinations = []
    valid_Re_values = []
    for U, rho, mu, source, aoa_list in combinations:
        if rho==0.:
            rho=1.

    

        Re = (U*rho*L) / mu
       
        valid_combinations.append([U, rho, mu, source, aoa_list, dt, L])
        valid_Re_values.append(Re)

    return valid_combinations
    
if __name__=='__main__':
    
    params_t,git_info = params()
    
    prefix="pf"
    
    if prefix=="cw":
        source_frequency_range = getattr(params_t, f"{prefix}_source_frequency")
        source_strength_range = getattr(params_t, f"{prefix}_source_strength")
        rho_range = getattr(params_t, f"{prefix}_rho")
        dt = getattr(params_t, f"{prefix}_dt")
        
        result = generate_combinations(source_frequency_range=source_frequency_range, source_strength_range=source_strength_range, rho_range=rho_range,dt=dt, eqtype="wave")
    else:
        U_range = getattr(params_t, f"{prefix}_inflow_range")
        rho_range = getattr(params_t, f"{prefix}_rho")
        mu_range = getattr(params_t, f"{prefix}_mu")
        source_range = getattr(params_t, f"{prefix}_source")
        aoa_range = getattr(params_t, f"{prefix}_aoa")
        Re_max = getattr(params_t, f"{prefix}_Re_max")
        Re_min = getattr(params_t, f"{prefix}_Re_min")
        dt = getattr(params_t, f"{prefix}_dt")
        L = getattr(params_t, f"{prefix}_L")

        result = generate_combinations(U_range, rho_range, mu_range, Re_max, Re_min, source_range, aoa_range, dt ,L=L, eqtype="fluid")