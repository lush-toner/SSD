import argparse
import logging
import os

import torch
import torch.distributed as dist
torch.backends.cudnn.benchmark = True

from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.engine.trainer import do_train
from ssd.modeling.detector import build_detection_model
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool

import random
import numpy as np

######### Set Seeds ###########
random_seed = 8138
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set the GPUs 2 and 3 to use

def train(cfg, args):
    logger = logging.getLogger('SSD.trainer')
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    print("=" * 70)
    print(model)
    print("=" * 70)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = make_optimizer(cfg, model, lr)

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS] # [80000, 100000]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(model, optimizer, scheduler, save_dir = cfg.OUTPUT_DIR, save_to_disk = save_to_disk, logger = logger)
    extra_checkpoint_data = checkpointer.load()
    # Define and get iteration
    arguments = {"iteration": 0}
    arguments.update(extra_checkpoint_data) # update 

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    train_loader = make_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter, start_iter=arguments['iteration'])

    model = do_train(cfg, model, train_loader, optimizer, scheduler, checkpointer, device, arguments, args)
    return model


def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument("--config-file", default="./configs/vgg_ssd300_voc0712.yaml", metavar="FILE", help="path to config file",type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=2500, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument("--skip-test", dest="skip_test", help="Do not test the final model", action="store_true")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1 # system environ var 
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    # logger.info(args)

    print('============================ args ============================')
    for k, v in sorted(vars(args).items()):
        print('%s: %s' % (str(k), str(v)))
    print('============================ End ============================')

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    print('=' * 50)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args)

    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, model, distributed=args.distributed)


if __name__ == '__main__':
    main()
