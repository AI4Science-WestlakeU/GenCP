from matplotlib import pyplot as plt
import torch
import numpy as np
import yaml
from datetime import datetime
import os
import sys
from termcolor import colored
# from sklearn import gaussian_process as gp  # Commented out to avoid import issues

from typing import Optional, Tuple

COLOR_LIST = [
    "b",
    "r",
    "g",
    "y",
    "c",
    "m",
    "skyblue",
    "indigo",
    "goldenrod",
    "salmon",
    "pink",
    "silver",
    "darkgreen",
    "lightcoral",
    "navy",
    "orchid",
    "steelblue",
    "saddlebrown",
    "orange",
    "olive",
    "tan",
    "firebrick",
    "maroon",
    "darkslategray",
    "crimson",
    "dodgerblue",
    "aquamarine",
    "b",
    "r",
    "g",
    "y",
    "c",
    "m",
    "skyblue",
    "indigo",
    "goldenrod",
    "salmon",
    "pink",
    "silver",
    "darkgreen",
    "lightcoral",
    "navy",
    "orchid",
    "steelblue",
    "saddlebrown",
    "orange",
    "olive",
    "tan",
    "firebrick",
    "maroon",
    "darkslategray",
    "crimson",
    "dodgerblue",
    "aquamarine",
]


# basic utils
class Printer(object):
    ## Example, to print code running time between two p.print() calls
    # p.print(f"test_start", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """

        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(
        self,
        item,
        tabs=0,
        is_datetime=None,
        banner_size=0,
        end=None,
        avg_window=-1,
        precision="second",
        is_silent=False,
    ):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2],
                avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window + 1, len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, "yellow"))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))


def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time

    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string


# config utils
def add_args_from_config(parser):
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    existing_args = {action.dest for action in parser._actions}

    # Add arguments from the config file to the parser
    for key, value in config.items():
        # If the argument is not already added, add it to the parser
        if key not in existing_args:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    return parser


def save_config_from_args(args, config_dir):
    config_dict = {k: v for k, v in vars(args).items() if k != "config"}
    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # config_dir = args.exp_path + "/results/" + time_now
    # os.makedirs(config_dir, exist_ok=True)  # Ensure the directory exists
    config_file_path = os.path.join(config_dir, "config.yaml")
    with open(config_file_path, "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False)

    return


# result analysis
def caculate_confidence_interval(data):
    """'
    input example: abs(pred_design-pred_simu)
    """
    list_dim = range(data.dim())
    if data.dim() > 1:
        MAE_batch_size = torch.mean(data, dim=tuple(list_dim[1:]))
    else:
        MAE_batch_size = data
    mean = torch.mean(MAE_batch_size)

    std_dev = torch.std(MAE_batch_size)
    min_value = min(MAE_batch_size)
    confidence_level = 0.95
    # pdb.set_trace()
    n = len(data)
    # kk = stats.t.ppf((1 + confidence_level) / 2, n - 1) * (std_dev / (n ** 0.5))
    margin_of_error = std_dev * 1.96 / torch.sqrt(torch.tensor(n, dtype=float))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    print("mean:", mean.item())
    print("std:", std_dev.item())
    print(f"margin_of_error:", margin_of_error)

    return mean, std_dev, margin_of_error, min_value


# training utils
def caculate_num_parameters(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # pdb.set_trace()
    return


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


def create_res(path: str, folder_name: str):
    """create result folder

    Args:
        path (str): Overall results folder
        folder_name (str): subfolder in results folder
    """
    if not os.path.exists(path):
        os.makedirs(path)
    res_path = os.path.join(path, folder_name)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        print(f"{res_path} is created。")
    else:
        print(f"{res_path} had been created.")
    return res_path


def get_parameter_net(net):
    total_num = sum(p.numel() for p in net.parameters())
    train_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"the parameter of the net is {train_num:d}")
    return total_num, train_num


def find_max_min(array):
    """
    This function accepts a numpy array or a torch tensor and returns the maximum and minimum values.

    Parameters:
    array (np.ndarray or torch.Tensor): The array or tensor to be checked.

    Returns:
    tuple: A tuple containing the maximum and minimum values.
    """
    # Check if the input is a numpy array
    if isinstance(array, np.ndarray):
        max_val = np.max(array)
        min_val = np.min(array)
    # Check if the input is a torch tensor
    elif isinstance(array, torch.Tensor):
        max_val = torch.max(array).item()  # .item() is used to convert the tensor to a scalar
        min_val = torch.min(array).item()
    else:
        # Raise an error if the input is not a numpy array or a torch tensor
        raise TypeError("Input must be a numpy array or a torch tensor")

    return max_val, min_val


class GRF(object):
    """generate 1D gaussian random field

    Args:
        object (_type_): _description_
    """

    def __init__(self, T=1, kernel="RBF", mean=0.0, length_scale=1.0, sigma=1, N=1000, interp="cubic"):
        """_summary_

        Args:
            T (int, optional): _description_. Defaults to 1.
            kernel (str, optional): _description_. Defaults to "RBF".
            mean (float, optional): mean. Defaults to 0.0.
            length_scale (float, optional): smooth factor. Defaults to 1.0.
            sigma (int, optional): standard deviation. Defaults to 1.
            N (int, optional): _description_. Defaults to 1000.
            interp (str, optional): _description_. Defaults to "cubic".
        """
        self.N = N
        self.mean = mean
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale) * sigma**2
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n):
        """Generate `n` random feature vectors."""
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T + self.mean


def plot_compare_2d(true_d, pred_d, savep=None, paraname="", Unit_="", language="eng"):
    # compare
    # shape: w*h
    if torch.is_tensor(true_d):
        if true_d.is_cuda:
            true_d = true_d.cpu()
            pred_d = pred_d.cpu()
        true_d = true_d.numpy()
        pred_d = pred_d.numpy()
    title_list = ["True ", "Pred ", "Error "]

    re_err = np.abs((true_d - pred_d) / (true_d + 1e-6))
    err = true_d - pred_d
    plt.figure(figsize=(8, 6), dpi=100)

    Unit1 = ""
    if len(Unit_) > 0:
        Unit1 = "(" + Unit_ + ")"

    plt.subplot(131)
    plt.imshow(
        true_d,
        cmap="jet",
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        interpolation="bilinear",
    )
    plt.title(title_list[0] + paraname + Unit1, fontsize=15)
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(
        pred_d,
        cmap="jet",
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        interpolation="bilinear",
    )

    plt.title(title_list[1] + paraname + Unit1, fontsize=15)
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(
        err,
        cmap="jet",
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        interpolation="bilinear",
    )

    plt.title(title_list[2] + paraname + Unit1, fontsize=15)
    plt.colorbar()

    plt.tight_layout()
    if savep is not None:
        plt.savefig(savep)
    plt.show()


def plot_scatter_compare(
    coord_x,
    coord_y,
    true_d,
    pred_d,
    pointsize=1,
    figsize=(18, 4),
    savep=None,
    fontsize=15,
    cmap="viridis",
    paraname="",
    Unit_="",
    language="eng",
    e_min=None,
    e_max=None,
):
    # compare
    # shape: w*h
    if torch.is_tensor(true_d):
        if true_d.is_cuda:
            true_d = true_d.cpu()
            pred_d = pred_d.cpu()
        true_d = true_d.numpy()
        pred_d = pred_d.numpy()
    title_list = ["Ground truth ", "Predition ", "Error "]

    vmin = min(np.min(true_d), np.min(pred_d))
    vmax = max(np.max(true_d), np.max(pred_d))

    Unit1 = ""
    if len(Unit_) > 0:
        Unit1 = "(" + Unit_ + ")"
    plt.figure(figsize=figsize)
    plt.subplot(131)
    sc1 = plt.scatter(coord_x, coord_y, c=true_d, cmap=cmap, edgecolors=None, s=pointsize, vmin=vmin, vmax=vmax)
    plt.title(title_list[0] + paraname + Unit1, fontsize=fontsize)
    plt.colorbar(sc1)

    plt.subplot(132)
    sc2 = plt.scatter(coord_x, coord_y, c=pred_d, cmap=cmap, edgecolors=None, s=pointsize, vmin=vmin, vmax=vmax)
    plt.title(title_list[1] + paraname + Unit1, fontsize=fontsize)
    plt.colorbar(sc2)

    plt.subplot(133)
    if e_max is None or e_min is None:
        e_max, e_min = find_max_min(pred_d - true_d)
    sc3 = plt.scatter(
        coord_x, coord_y, c=(pred_d - true_d), cmap=cmap, edgecolors=None, s=pointsize, vmin=e_min, vmax=e_max
    )
    plt.title(title_list[2] + paraname + Unit1, fontsize=fontsize)
    plt.colorbar(sc3)

    plt.tight_layout()
    if savep is not None:
        plt.savefig(savep)
    plt.show()


def plot_contourf(Z, savep, xlabel="x", ylabel="y", title=None):
    plt.imshow(
        Z,
        cmap="jet",
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        interpolation="bilinear",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(savep)
    plt.close()


def random_split_line(n, bound, min_gap):
    # n: points
    d_step = 1 / n
    while True:
        divisions = np.linspace(0, 1, n)
        divisions = np.random.uniform(0, d_step, n - 1) + divisions[:-1]
        divisions = (divisions - divisions[0]) * (bound[1] - bound[0]) + bound[0]
        divisions = np.concatenate((divisions, np.array(bound[1]).reshape(-1)))
        d_divisions = divisions[1:] - divisions[:-1]
        if np.min(d_divisions) >= min_gap:
            break
    return divisions


def minmax_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def L2_norm(array):
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach().numpy()
    b = array.shape[0]
    array = array.reshape(b, -1)
    norm = np.sum(array**2, axis=1) ** 0.5
    return norm


def relative_error(data_t, data_p):
    return np.sum(L2_norm(data_t - data_p) / L2_norm(data_t)) / data_t.shape[0]


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x



class FluidFieldVisualizer:
    """Fluid field visualizer"""
    
    def __init__(self, args, grid_size = [64, 64], save_dir: str = "visualization_results", create_timestamp_folder: bool = True):
        """
        Initialize visualizer
        
        Args:
            grid_size: Grid size (default 64x64)
            save_dir: Save directory
            create_timestamp_folder: Whether to create timestamp folder
        """
        self.grid_size = grid_size
        self.args = args
        
        if create_timestamp_folder:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = os.path.join(save_dir, timestamp)
        else:
            self.save_dir = save_dir
            
        os.makedirs(self.save_dir, exist_ok=True)
    
    def visualize_field(self, 
                               u_pred: np.ndarray, 
                               u_true: np.ndarray = None,
                               title: str = "Velocity Field",
                               save_name: str = "velocity_field.png",
                               cmap: str = 'RdBu_r',
                               colorbar: bool = True,
                               vmin: Optional[float] = None,
                               vmax: Optional[float] = None,
                                use_interpolation: str = 'bilinear',
                                interpolation_factor: int = 4,
                                smooth_sigma: float = 0.8) -> None:
        
        if vmin is None:
            vmin = min(u_pred.min(), u_true.min())
        if vmax is None:
            vmax = max(u_pred.max(), u_true.max())
        
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        extent = [0, self.grid_size[1]-1, 0, self.grid_size[0]-1]
        im1 = axes[0].imshow(
                u_pred,
                cmap=cmap, 
                alpha=0.8, 
                vmin=vmin, 
                vmax=vmax,
                extent=extent,
                origin='lower',
                interpolation=use_interpolation
            )
        # axes[0].contour(X, Y, u_pred, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        if colorbar:
            cbar1 = plt.colorbar(im1, ax=axes[0])
            # cbar1.set_label('Velocity', fontsize=12)
        axes[0].set_title('Predicted', fontsize=12, fontweight='bold')
        axes[0].set_aspect('equal')
        
        im2 = axes[1].imshow(
                u_true,
                cmap=cmap, 
                alpha=0.8, 
                vmin=vmin, 
                vmax=vmax,
                extent=extent,
                origin='lower',
                interpolation=use_interpolation
            )
        # axes[1].contour(X, Y, u_true, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        if colorbar:
            cbar2 = plt.colorbar(im2, ax=axes[1])
            # cbar2.set_label('Velocity', fontsize=12)
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].set_aspect('equal')
        
        error = np.abs(u_pred - u_true)
        im3 = axes[2].imshow(
                error,
                cmap='Reds', 
                alpha=0.8, 
                vmin=error.min(), 
                vmax=error.max(),
                extent=extent,
                origin='lower',
                interpolation=use_interpolation
            )
        # axes[2].contour(X, Y, error, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        if colorbar:
            cbar3 = plt.colorbar(im3, ax=axes[2])
            # cbar3.set_label('Error', fontsize=12)
        axes[2].set_title('Error', fontsize=12, fontweight='bold')
        axes[2].set_aspect('equal')
        # axes[2].grid(False, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Velocity contour image saved to: {save_path}")

    def visualize_time_series_gif(self, 
                                 u_pred: np.ndarray, 
                                 u_true: np.ndarray = None,
                                 title: str = "Time Series Animation",
                                 save_name: str = "time_series.gif",
                                 fps: int = 2,
                                 show_colorbar: bool = False,
                                 vmin: Optional[float] = None,
                                 vmax: Optional[float] = None) -> None:
        """
        Generate time series GIF animation
        
        Args:
            u_pred: Predicted velocity field data (time_steps, grid_size, grid_size)
            u_true: Ground truth velocity field data (time_steps, grid_size, grid_size)
            title: Image title
            save_name: Save filename
            fps: Frame rate (frames per second)
            show_colorbar: Whether to show colorbar
            vmin: Minimum value, auto-computed from full time trajectory if None
            vmax: Maximum value, auto-computed from full time trajectory if None
        """
        import matplotlib.animation as animation

        if vmin is None:
            vmin = min(u_pred.min(), u_true.min())
        if vmax is None:
            vmax = max(u_pred.max(), u_true.max())
        
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        
        time_steps = u_pred.shape[0]
        
        error_data = np.abs(u_pred - u_true)
        error_vmin, error_vmax = error_data.min(), error_data.max()
        
        if error_vmin == error_vmax:
            error_vmin = 0
            error_vmax = max(0.1, error_vmax + 0.1)
        
        if show_colorbar:
            fig = plt.figure(figsize=(20, 6))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], 
                                 left=0.05, right=0.85, wspace=0.3)
            axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        extent = [0, self.grid_size[1]-1, 0, self.grid_size[0]-1]

        im1 = axes[0].imshow(u_pred[0], cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.8)
        im2 = axes[1].imshow(u_true[0], cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.8)
        im3 = axes[2].imshow(error_data[0], cmap='Reds', vmin=error_vmin, vmax=error_vmax, alpha=0.8,
                                extent=extent,
                                origin='lower',
                                interpolation='bilinear')
        
        axes[0].set_title('Predicted (t=0)', fontsize=12, fontweight='bold')
        axes[1].set_title('Ground Truth (t=0)', fontsize=12, fontweight='bold')
        axes[2].set_title('Error (t=0)', fontsize=12, fontweight='bold')
        
        for ax in axes:
            ax.set_aspect('equal')
        
        if show_colorbar:
            cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8, pad=0.02)
            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8, pad=0.02)
            cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8, pad=0.02)
            
            cbar1.set_label('Predicted Value', rotation=270, labelpad=15)
            cbar2.set_label('True Value', rotation=270, labelpad=15)
            cbar3.set_label('Absolute Error', rotation=270, labelpad=15)
        
        def animate(frame):
            im1.set_array(u_pred[frame])
            im2.set_array(u_true[frame])
            
            error_frame = np.abs(u_pred[frame] - u_true[frame])
            im3.set_array(error_frame)
            
            axes[0].set_title(f'Predicted (t={frame})', fontsize=12, fontweight='bold')
            axes[1].set_title(f'Ground Truth (t={frame})', fontsize=12, fontweight='bold')
            axes[2].set_title(f'Error (t={frame})', fontsize=12, fontweight='bold')
            
            return [im1, im2, im3]
        
        ani = animation.FuncAnimation(fig, animate, frames=time_steps, 
                                     interval=1000//fps, blit=False, repeat=True)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        ani.save(save_path, writer='pillow', fps=fps)
        plt.close()
        print(f"Time series GIF saved to: {save_path}")

    def visualize_sdf_field(self, 
                                   sdf_pred: np.ndarray, 
                                   sdf_true: np.ndarray = None,
                                   title: str = "SDF Contour Field",
                                   save_name: str = "sdf_contour_field.png",
                                   cmap: str = 'RdBu_r',
                                   colorbar: bool = True,
                                   vmin: Optional[float] = None,
                                   vmax: Optional[float] = None,
                                   index: Optional[int] = None,
                                   use_interpolation: str = 'bilinear', # bilinear  
                                   zero_tolerance: float = 1e-6) -> None:
        """
        Visualize SDF field
        
        Args:
            sdf_pred: Predicted SDF field (grid_size, grid_size)
            sdf_true: Ground truth SDF field (grid_size, grid_size)
            title: Image title
            save_name: Save filename
            cmap: Color map
            colorbar: Whether to show colorbar
            vmin: Colorbar minimum value, auto-computed if None
            vmax: Colorbar maximum value, auto-computed if None
            index: Index number, added to filename if provided
            zero_tolerance: Tolerance for SDF=0 (currently unused)
        """
        if vmin is None:
            vmin = min(sdf_pred.min(), sdf_true.min())
        if vmax is None:
            vmax = max(sdf_pred.max(), sdf_true.max())
        
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        extent = [0, self.grid_size[1]-1, 0, self.grid_size[0]-1]
        
        import matplotlib.colors as mcolors
        levels = np.linspace(vmin, vmax, 21)
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap if isinstance(cmap, int) else plt.cm.get_cmap(cmap).N)
        
        im1 = axes[0].imshow(
            sdf_pred, 
            cmap=cmap, 
            alpha=0.8, 
            extent=extent,
            origin='lower',
            interpolation=use_interpolation
        )
        
        solid_mask_pred = (sdf_pred < self.args.sdf_threshold)

        if solid_mask_pred.any():
            axes[0].contour(
                solid_mask_pred.astype(float), 
                levels=[0.5], 
                colors='yellow', 
                linewidths=1,
                origin='lower'
            )
        
        if colorbar:
            cbar1 = plt.colorbar(im1, ax=axes[0])
        axes[0].set_title('Predicted SDF', fontsize=12, fontweight='bold')
        axes[0].set_aspect('equal')
        
        im2 = axes[1].imshow(
            sdf_true, 
            cmap=cmap, 
            alpha=0.8, 
            extent=extent,
            origin='lower',
            interpolation=use_interpolation
        )
        
        solid_mask_true = (sdf_true < self.args.sdf_threshold)
        
        if solid_mask_true.any():
            axes[1].contour(
                solid_mask_true.astype(float), 
                levels=[0.5], 
                colors='yellow', 
                linewidths=1,
                origin='lower'
            )
        
        if colorbar:
            cbar2 = plt.colorbar(im2, ax=axes[1])
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].set_aspect('equal')
        
        error = np.abs(sdf_pred - sdf_true)
        im3 = axes[2].imshow(
            error, 
            cmap='Reds', 
            alpha=0.8,
            extent=extent,
            origin='lower',
            interpolation=use_interpolation
        )
        if colorbar:
            cbar3 = plt.colorbar(im3, ax=axes[2])
        axes[2].set_title('Absolute Error', fontsize=12, fontweight='bold')
        axes[2].set_aspect('equal')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if index is not None:
            name, ext = os.path.splitext(save_name)
            save_name = f"{name}_{index}{ext}"
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"SDF contour field image saved to: {save_path}")

    def visualize_sdf_time_series_gif(self, 
                                     sdf_pred: np.ndarray, 
                                     sdf_true: np.ndarray = None,
                                     title: str = "SDF Time Series Animation",
                                     save_name: str = "sdf_time_series.gif",
                                     fps: int = 2,
                                     cmap: str = 'RdBu_r',
                                     vmin: Optional[float] = None,
                                     vmax: Optional[float] = None,
                                     index: Optional[int] = None,
                                     show_colorbar: bool = True) -> None:
        """
        Generate SDF field time series GIF animation
        
        Args:
            sdf_pred: Predicted SDF field data (time_steps, grid_size, grid_size)
            sdf_true: Ground truth SDF field data (time_steps, grid_size, grid_size)
            title: Image title
            save_name: Save filename
            fps: Frame rate (frames per second)
            cmap: Color map
            vmin: Colorbar minimum value, auto-computed if None
            vmax: Colorbar maximum value, auto-computed if None
            index: Index number, added to filename if provided
            show_colorbar: Whether to show colorbar
        """
        import matplotlib.animation as animation
        
        if sdf_true is None:
            sdf_true = np.zeros_like(sdf_pred)
        
        time_steps = sdf_pred.shape[0]
        
        if vmin is None:
            vmin = min(sdf_pred.min(), sdf_true.min())
        if vmax is None:
            vmax = max(sdf_pred.max(), sdf_true.max())
        
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        
        error_data = np.abs(sdf_pred - sdf_true)
        error_vmin, error_vmax = error_data.min(), error_data.max()
        
        import matplotlib.colors as mcolors
        levels = np.linspace(vmin, vmax, 21)
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap if isinstance(cmap, int) else plt.cm.get_cmap(cmap).N)
        
        error_levels = np.linspace(error_vmin, error_vmax, 21)
        error_norm = mcolors.BoundaryNorm(error_levels, ncolors=plt.cm.get_cmap('Reds').N)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        if show_colorbar:
            temp_im1 = axes[0].imshow(sdf_pred[0], cmap=cmap, 
                                       norm=norm, alpha=0.8, origin='lower')
            temp_im2 = axes[1].imshow(sdf_true[0], cmap=cmap,
                                       norm=norm, alpha=0.8, origin='lower')
            temp_im3 = axes[2].imshow(error_data[0], cmap='Reds',
                                       norm=error_norm, alpha=0.8, origin='lower')
            
            plt.colorbar(temp_im1, ax=axes[0], shrink=0.8)
            plt.colorbar(temp_im2, ax=axes[1], shrink=0.8)
            plt.colorbar(temp_im3, ax=axes[2], shrink=0.8)
            
            for ax in axes:
                ax.clear()
        
        def animate(frame):
            for ax in axes:
                ax.clear()
            
            im1 = axes[0].imshow(
                sdf_pred[frame], 
                cmap=cmap, 
                alpha=0.8, 
                norm=norm,
                origin='lower'
            )
            
            solid_mask_pred = sdf_pred[frame] < self.args.sdf_threshold

            if solid_mask_pred.any():
                axes[0].contour(
                    solid_mask_pred.astype(float), 
                    levels=[0.5], 
                    colors='yellow', 
                    linewidths=1,
                    origin='lower'
                )
            
            axes[0].set_title(f'Predicted SDF (t={frame})', fontsize=12, fontweight='bold')
            axes[0].set_aspect('equal')
            
            im2 = axes[1].imshow(
                sdf_true[frame], 
                cmap=cmap, 
                alpha=0.8, 
                norm=norm,
                origin='lower'
            )
            
            solid_mask_true = sdf_true[frame] < self.args.sdf_threshold
            
            if solid_mask_true.any():
                axes[1].contour(
                    solid_mask_true.astype(float), 
                    levels=[0.5], 
                    colors='yellow', 
                    linewidths=1,
                    origin='lower'
                )
            
            axes[1].set_title(f'Ground Truth SDF (t={frame})', fontsize=12, fontweight='bold')
            axes[1].set_aspect('equal')
            
            error = np.abs(sdf_pred[frame] - sdf_true[frame])
            im3 = axes[2].imshow(
                error, 
                cmap='Reds', 
                alpha=0.8,
                norm=error_norm,
                origin='lower'
            )
            
            axes[2].set_title(f'Absolute Error (t={frame})', fontsize=12, fontweight='bold')
            axes[2].set_aspect('equal')
            
            return [im1, im2, im3]
        
        ani = animation.FuncAnimation(fig, animate, frames=time_steps, 
                                     interval=1000//fps, blit=False, repeat=True)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if index is not None:
            name, ext = os.path.splitext(save_name)
            save_name = f"{name}_{index}{ext}"
        
        save_path = os.path.join(self.save_dir, save_name)
        ani.save(save_path, writer='pillow', fps=fps)
        plt.close()
        print(f"SDF time series GIF saved to: {save_path}")



def generate_test_fluid_data(grid_size: int = 64, time_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test fluid data
    
    Args:
        grid_size: Grid size
        time_steps: Number of time steps
    
    Returns:
        velocity_data: Velocity field data (time_steps, 2, grid_size, grid_size)
        pressure_data: Pressure field data (time_steps, grid_size, grid_size)
    """
    x = np.linspace(0, 2*np.pi, grid_size)
    y = np.linspace(0, 2*np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    
    velocity_data = np.zeros((time_steps, 2, grid_size, grid_size))
    pressure_data = np.zeros((time_steps, grid_size, grid_size))
    
    for t in range(time_steps):
        time_factor = 1.0 + 0.1 * t
        
        u = -np.sin(Y) * np.cos(X) * time_factor
        v = np.cos(Y) * np.sin(X) * time_factor
        
        noise_u = 0.1 * np.random.randn(grid_size, grid_size)
        noise_v = 0.1 * np.random.randn(grid_size, grid_size)
        
        velocity_data[t, 0] = u + noise_u
        velocity_data[t, 1] = v + noise_v
        
        pressure = -0.5 * (u**2 + v**2) + 0.1 * np.sin(X + Y) * time_factor
        pressure_data[t] = pressure
    
    return velocity_data, pressure_data


def generate_test_fluid_data(grid_size: int = 64, time_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test fluid data
    
    Args:
        grid_size: Grid size
        time_steps: Number of time steps
    
    Returns:
        velocity_data: Velocity field data (time_steps, 2, grid_size, grid_size)
        pressure_data: Pressure field data (time_steps, grid_size, grid_size)
    """
    x = np.linspace(0, 2*np.pi, grid_size)
    y = np.linspace(0, 2*np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    
    velocity_data = np.zeros((time_steps, 2, grid_size, grid_size))
    pressure_data = np.zeros((time_steps, grid_size, grid_size))
    
    for t in range(time_steps):
        time_factor = 1.0 + 0.1 * t
        
        u = -np.sin(Y) * np.cos(X) * time_factor
        v = np.cos(Y) * np.sin(X) * time_factor
        
        noise_u = 0.1 * np.random.randn(grid_size, grid_size)
        noise_v = 0.1 * np.random.randn(grid_size, grid_size)
        
        velocity_data[t, 0] = u + noise_u
        velocity_data[t, 1] = v + noise_v
        
        pressure = -0.5 * (u**2 + v**2) + 0.1 * np.sin(X + Y) * time_factor
        pressure_data[t] = pressure
    
    return velocity_data, pressure_data


if __name__ == "__main__":
    print("Starting test data generation and visualization...")
    
    visualizer = FluidFieldVisualizer(grid_size=[64,64], save_dir="/path/to/visualization_results", create_timestamp_folder=False)
    
    velocity_data, pressure_data = generate_test_fluid_data(grid_size=64, time_steps=10)
    
    print(f"Generated data shape: velocity field {velocity_data.shape}, pressure field {pressure_data.shape}")

    print("Generating velocity field plots...")
    visualizer.visualize_field(
        u_pred=velocity_data[0, 0], 
        u_true=None, 
        title="Test Velocity Field (t=0)",
        save_name="test_velocity_field_x.png"
    )
    visualizer.visualize_field(
        u_pred=velocity_data[0, 1], 
        u_true=None, 
        title="Test Velocity Field (t=0)",
        save_name="test_velocity_field_y.png"
    )
    
    print("Generating pressure field plots...")
    visualizer.visualize_field(
        u_pred=pressure_data[0], 
        u_true=None, 
        title="Test Pressure Field (t=0)",
        save_name="test_pressure_field.png"
    )
    
    print("Generating time series GIF animation...")
    visualizer.visualize_time_series_gif(
        u_pred=velocity_data[:,0],
        u_true=None,
        title="Time Series Animation Test",
        save_name="time_series_animation.gif",
        fps=2
    )

    print(f"\nAll visualization results saved to: {visualizer.save_dir}")
    print("Visualization test completed!")


# Data caching utilities
def generate_cache_dir_name(args, dataset_name):
    """Generate cache directory name based on dataset arguments"""
    cache_params = [
        f"length_{args.length}",
        f"input_{args.input_step}",
        f"output_{args.output_step}",
        f"stride_{args.stride}",
        f"stage_{args.stage}",
        f"delta_t_{args.num_delta_t}",
        f"dt_{args.dt}"
    ]
    cache_dir = f"{dataset_name}_{'_'.join(cache_params)}"
    return cache_dir


def get_cache_path(cache_dir_name, mode='train'):
    """Get full cache path for a given mode"""
    cache_base = os.path.join("cache_data", cache_dir_name, mode)
    return cache_base


def check_cache_exists(cache_dir_name, mode='train'):
    """Check if cache exists for given parameters"""
    cache_path = get_cache_path(cache_dir_name, mode)
    return os.path.exists(cache_path) and len(os.listdir(cache_path)) > 0


def create_data_cache(dataset, cache_dir_name, mode='train', batch_size=512):
    """Create cache for dataset by processing all samples"""
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback if tqdm is not available
        def tqdm(iterable, desc=""):
            print(f"{desc}: Processing {len(iterable)} items")
            return iterable
    import pickle

    cache_path = get_cache_path(cache_dir_name, mode)
    os.makedirs(cache_path, exist_ok=True)

    print(f"Creating cache for {mode} dataset at {cache_path}")

    # Save dataset metadata
    metadata = {
        'length': len(dataset),
        'files_path': dataset.files_path,
        'input_size': dataset.input_size,
        'output_size': dataset.output_size,
        'stage': dataset.stage,
        'num_delta_t': dataset.num_delta_t,
        'dt': dataset.dt
    }

    with open(os.path.join(cache_path, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # Process all samples
    for i in tqdm(range(len(dataset)), desc=f"Caching {mode} data"):
        sample = dataset[i]

        # Save each sample as individual torch file for fast loading
        sample_data = {
            'x': sample[0],  # input data
            'y': sample[1],  # target data
            'grid_x': sample[2],  # grid coordinates
            'grid_y': sample[3],  # grid coordinates
            'attrs': sample[4]  # attributes
        }

        torch.save(sample_data, os.path.join(cache_path, f'sample_{i:06d}.pt'))

    print(f"Cache creation completed for {mode} dataset")


def load_cached_sample(cache_path, index):
    """Load a single cached sample"""
    sample_path = os.path.join(cache_path, f'sample_{index:06d}.pt')
    return torch.load(sample_path, weights_only=False)  # Keep False for now to allow full tensor loading