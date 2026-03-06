import os
import sys
import time
import datetime as dt
import csv
import logging
from natsort import natsorted
import json
import shutil

# 获取FDGO根目录
FDGO_BASE_DIR = os.path.split(os.path.split(__file__)[0])[0]


class Logger:
    """
    训练日志记录器

    功能：
    - 配置全局logging系统（使所有print和logging都输出到console和文件）
    - 记录残差数据到CSV（带缓冲区）
    - 保存和加载模型状态
    - 可选：复制源代码到实验目录
    """

    def __init__(
        self,
        name,
        datetime=None,
        use_dat=False,
        params=None,
        saving_path=None,
        copy_code=True,
        log_level="INFO",
    ):
        """
        Logger logs metrics to CSV files / log files
        :name: logging name (e.g. model name / dataset name / ...)
        :datetime: date and time of logging start (useful in case of multiple runs). Default: current date and time is picked
        :use_dat: log output to dat files (for plotting)
        :params: parameter object
        :saving_path: custom saving path (default: auto-generate)
        :copy_code: copy source code to experiment directory
        :log_level: logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.name = name
        self.params = params

        if datetime:
            self.datetime = datetime
        else:
            self.datetime = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        if saving_path is not None:
            self.saving_path = saving_path
        else:
            # 使用FDGO目录作为基础路径
            self.saving_path = os.path.join(FDGO_BASE_DIR, "Logger", name, self.datetime)

        # 创建保存目录
        os.makedirs(self.saving_path, exist_ok=True)

        # 配置logging系统
        self._setup_logging(log_level)

        # 复制源代码
        if copy_code:
            self._copy_source_code()

        self.use_dat = use_dat
        if use_dat:
            os.makedirs(f"{self.saving_path}/logs", exist_ok=True)

        self.best_loss = 0

        # 初始化残差记录系统
        self._init_residual_logging()

    def _setup_logging(self, log_level):
        """
        配置全局logging系统

        所有模块使用 logging.getLogger(__name__) 都会自动输出到console和文件
        """
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        # 配置根logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # 清除已有的handlers
        if root_logger.handlers:
            root_logger.handlers.clear()

        # 格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler - 保存所有日志到文件
        log_file = os.path.join(self.saving_path, 'training.log')
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # 创建当前实例的logger
        self.logger = logging.getLogger(f"FDGO.{self.name}")

        # 记录初始化信息
        self.logger.info(f"Logger initialized")
        self.logger.info(f"Saving path: {self.saving_path}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("=" * 60)

    def _copy_source_code(self):
        """复制源代码到实验目录（遵守.gitignore规则）"""
        source_dir = FDGO_BASE_DIR
        target_dir = f"{self.saving_path}/source"

        try:
            os.makedirs(target_dir, exist_ok=True)
            shutil.copytree(
                source_dir,
                target_dir,
                ignore=self.ignore_files_and_folders,
                dirs_exist_ok=True,
            )
            self.logger.info(f"Source code copied to {target_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to copy source code: {e}")

    def ignore_files_and_folders(self, dir_name, names):
        """定义要忽略的文件和文件夹（基于.gitignore）"""
        ignored = set()
        
        # 读取.gitignore
        gitignore_path = os.path.join(FDGO_BASE_DIR, ".gitignore")
        gitignore_patterns = set()
        
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释
                    if line and not line.startswith('#'):
                        # 移除尾部的/
                        pattern = line.rstrip('/')
                        gitignore_patterns.add(pattern)
        
        # 默认忽略的模式
        default_ignores = {"__pycache__", ".git", ".vscode", "Logger"}
        gitignore_patterns.update(default_ignores)
        
        for name in names:
            # 检查是否匹配gitignore模式
            for pattern in gitignore_patterns:
                # 简单匹配：检查名称是否与模式匹配
                if name == pattern or name.endswith(pattern):
                    ignored.add(name)
                    break
        
        return ignored

    def _init_residual_logging(self):
        """初始化残差记录系统"""
        self.residual_buffer = []
        self.residual_buffer_size = 100
        self.residual_csv_path = os.path.join(self.saving_path, "residuals.csv")
        self.residual_headers = None
        self.residual_headers_written = False

    # ==================== Logging代理方法 ====================

    def info(self, msg, *args, **kwargs):
        """记录INFO级别日志"""
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """记录DEBUG级别日志"""
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """记录WARNING级别日志"""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """记录ERROR级别日志"""
        self.logger.error(msg, *args, **kwargs)

    # ==================== 残差记录 ====================

    def log_residuals(self, **kwargs):
        """
        记录残差数据到缓冲区（满时自动写入CSV）

        Args:
            **kwargs: 任意键值对，例如:
                time_step=0, inner_iter=10, loss_cont=0.001, loss_mom=0.002

        Example:
            logger.log_residuals(
                time_step=0,
                inner_iter=10,
                loss_total=0.001,
                loss_cont_L2=0.002,
                loss_mom_L2=0.003
            )
        """
        if not kwargs:
            return

        # 第一次调用时确定表头
        if self.residual_headers is None:
            # 确保time_step作为第一列
            if 'time_step' in kwargs:
                self.residual_headers = ['time_step'] + sorted([k for k in kwargs.keys() if k != 'time_step'])
            else:
                self.residual_headers = sorted(kwargs.keys())

        # 按表头顺序创建数据行
        row_data = [kwargs.get(h, 0.0) for h in self.residual_headers]
        self.residual_buffer.append(row_data)

        # 缓冲区满时写入
        if len(self.residual_buffer) >= self.residual_buffer_size:
            self._flush_residual_buffer()

    def _flush_residual_buffer(self):
        """将缓冲区数据写入CSV文件"""
        if not self.residual_buffer or self.residual_headers is None:
            return

        mode = 'w' if not self.residual_headers_written else 'a'

        with open(self.residual_csv_path, mode, newline='') as f:
            writer = csv.writer(f)

            if not self.residual_headers_written:
                writer.writerow(self.residual_headers)
                self.residual_headers_written = True

            writer.writerows(self.residual_buffer)

        self.residual_buffer.clear()

    def finalize_residuals(self):
        """完成残差记录，写入剩余数据"""
        if self.residual_buffer:
            self._flush_residual_buffer()
            self.logger.info(f"Residuals finalized: saved to {self.residual_csv_path}")

    # ==================== dat文件记录（向后兼容） ====================

    def log(self, item, value, index):
        """
        log index value couple for specific item into dat file
        :item: string describing item (e.g. "training_loss","test_loss")
        :value: value to log
        :index: index (e.g. batchindex / epoch)
        """
        if self.use_dat:
            filename = f"{self.saving_path}/logs/{item}.dat"

            if os.path.exists(filename):
                append_write = "a"
            else:
                append_write = "w"

            with open(filename, append_write) as log_file:
                log_file.write("{}, {}\n".format(index, value))

    def log_items(self, index, **items):

        filename = f"Logger/{self.name}/{self.datetime}/logs/results.dat"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        write_header = not os.path.exists(filename)
        with open(filename, "a") as f:
            if write_header:
                header = ["epoch"] + list(items.keys())
                f.write(", ".join(header) + "\n")
            values = [index] + [items[k] for k in items.keys()]
            f.write(", ".join(str(v) for v in values) + "\n")
        
    # ==================== 模型状态管理 ====================

    def save_state(self, model, optimizer, scheduler, index="final", loss=None):
        """
        saves state of model and optimizer
        :model: model to save (if list: save multiple models)
        :optimizer: optimizer (if list: save multiple optimizers)
        :index: index of state to save (e.g. specific epoch)
        """
        os.makedirs(self.saving_path + "/states", exist_ok=True)
        path = self.saving_path + "/states"

        with open(path + "/commandline_args.json", "wt") as f:
            json.dump(
                {**vars(self.params)}, f, indent=4, ensure_ascii=False
            )
        if scheduler is not None:
            model.save_checkpoint(path + "/{}.state".format(index), optimizer, scheduler)
        else:
            model.save_checkpoint(path + "/{}.state".format(index), optimizer)
        return path + "/{}.state".format(index)

    def load_state(
        self,
        model,
        optimizer,
        scheduler,
        datetime=None,
        index=None,
        continue_datetime=False,
        device=None,
    ):
        """
        loads state of model and optimizer
        """
        if datetime is None:
            for _, dirs, _ in os.walk(f"{FDGO_BASE_DIR}/Logger/{self.name}/"):
                datetime = sorted(dirs)[-1]
                if datetime == self.datetime:
                    datetime = sorted(dirs)[-2]
                break

        if continue_datetime:
            self.datetime = datetime

        if index is None:
            for _, _, files in os.walk(
                f"{FDGO_BASE_DIR}/Logger/{self.name}/{datetime}/states/"
            ):
                index = os.path.splitext(natsorted(files)[-1])[0]
                break

        path = f"{FDGO_BASE_DIR}/Logger/{self.name}/{datetime}/states/{index}.state"

        if scheduler is not None:
            model.load_checkpoint(
                optimizer=optimizer, scheduler=scheduler, ckpdir=path, device=device
            )
        else:
            model.load_checkpoint(
                optimizer=optimizer, ckpdir=path, device=device
            )

        return datetime, index


t_start = 0


def t_step():
    """
    returns delta t from last call of t_step()
    """
    global t_start
    t_end = time.perf_counter()
    delta_t = t_end - t_start
    t_start = t_end
    return delta_t
