import os
import random
import torch.distributed as dist
import torch as th

def set_seed_logger_random(args):
    '''
    training or evaluation on multiple GPUs requires different randomness
    '''
    if os.path.exists(args.output_dir) == False and dist.get_rank() == 0:
        os.makedirs(args.output_dir)
    # random.seed(args.seed)
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # np.random.seed(args.seed)
    # th.manual_seed(args.seed)
    # th.cuda.manual_seed(args.seed)
    # th.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

    # if dist.get_rank() == 0:
    #     logger.log("Effective parameters:")
    #     for key in sorted(args.__dict__):
    #         logger.log("  <<< {}: {}".format(key, args.__dict__[key]))
    return args

