import setproctitle
import argparse
import os

import sys
import torch
sys.path.insert(0, sys.path[0]+'/..')
print(sys.path)
from mmengine.config import Config, DictAction
from mmpl.engine.runner import PLRunner
import os.path as osp
from mmpl.registry import RUNNERS
from mmpl.utils import register_all_modules

torch.set_float32_matmul_precision('high')
register_all_modules()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pl model')
    parser.add_argument('--config', default='../configs//SegSAMPLerEmbAdd_config.py',
                        help='train config file path')
    parser.add_argument('--is-debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--ckpt-path', default=None, help='checkpoint path')
    parser.add_argument('--status', default='fit', help='fit or test', choices=['fit', 'test', 'predict', 'validate'])
    parser.add_argument('--work-dir', default="test", help='the dir to save logs and mmpl')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')    
    
    args = parser.parse_args()
    return args

import numpy as np
import random
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    if args.work_dir is not None:
        cfg.trainer_cfg['default_root_dir'] = args.work_dir
    elif cfg.trainer_cfg.get('default_root_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.trainer_cfg['default_root_dir'] = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.is_debug:
        cfg.trainer_cfg['fast_dev_run'] = True
        cfg.trainer_cfg['logger'] = None

    if 'runner_type' not in cfg:
        runner = PLRunner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.run(args.status, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()

