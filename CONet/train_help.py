from typing import Tuple
from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path
import os
import platform
import time
import pandas as pd
import gc
import numpy as np
import global_vars as GLOBALS
from AdaS import AdaS
from test import test_main, accuracy, AverageMeter
from optim.sls import SLS
from optim.sps import SPS
from optim import get_optimizer_scheduler
from early_stop import EarlyStop
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from utils import parse_config
from metrics import Metrics
from models import get_net
from data import get_data
from openpyxl import load_workbook
import yaml


import re

from utils import CrossEntropyLabelSmooth



def get_args(sub_parser: _SubParsersAction):
    # print("\n---------------------------------")
    # print("AdaS Train Args")
    # print("---------------------------------\n")
    # sub_parser.add_argument(
    #     '-vv', '--very-verbose', action='store_true',
    #     dest='very_verbose',
    #     help="Set flask debug mode")
    # sub_parser.add_argument(
    #     '-v', '--verbose', action='store_true',
    #     dest='verbose',
    #     help="Set flask debug mode")
    # sub_parser.set_defaults(verbose=False)
    # sub_parser.set_defaults(very_verbose=False)
    sub_parser.add_argument(
        '--config', dest='config',
        default='config.yaml', type=str,
        help="Set configuration file path: Default = 'config.yaml'")
    sub_parser.add_argument(
        '--data', dest='data',
        default='.adas-data', type=str,
        help="Set data directory path: Default = '.adas-data'")
    sub_parser.add_argument(
        '--output', dest='output',
        default='adas_search', type=str,
        help="Set output directory path: Default = '.adas-output'")
    sub_parser.add_argument(
        '--checkpoint', dest='checkpoint',
        default='.adas-checkpoint', type=str,
        help="Set checkpoint directory path: Default = '.adas-checkpoint")
    sub_parser.add_argument(
        '--root', dest='root',
        default='.', type=str,
        help="Set root path of project that parents all others: Default = '.'")
    sub_parser.set_defaults(resume=False)
    sub_parser.add_argument(
        '--cpu', action='store_true',
        dest='cpu',
        help="Flag: CPU bound training")
    sub_parser.set_defaults(cpu=False)
    sub_parser.add_argument(
        '--resume-search', action='store_true',
        dest='resume_search',
        help="Flag: Resume searching from latest trial"
    )
    sub_parser.set_defaults(resume_search=False)
    sub_parser.add_argument(
        '--train-num', type=int,
        dest='train_num',
        help="Number of times to run full train"
    )
    sub_parser.set_defaults(train_num=-1)


def build_paths(args: APNamespace):

    root_path = Path(args.root).expanduser()
    config_path = Path(args.config).expanduser()
    data_path = root_path / Path(args.data).expanduser()
    output_path = root_path / Path(args.output).expanduser()
    # global checkpoint_path, config
    GLOBALS.CHECKPOINT_PATH = root_path / Path(args.checkpoint).expanduser()
    #checks
    if not config_path.exists():
        # logging.critical(f"AdaS: Config path {config_path} does not exist")
        print(os.getcwd())
        print(config_path)
        raise ValueError(f"AdaS: Config path {config_path} does not exist")
    if not data_path.exists():
        print(f"AdaS: Data dir {data_path} does not exist, building")
        data_path.mkdir(exist_ok=True, parents=True)
    if not output_path.exists():
        print(f"AdaS: Output dir {output_path} does not exist, building")
        output_path.mkdir(exist_ok=True, parents=True)
    if not GLOBALS.CHECKPOINT_PATH.exists():
        if args.resume:
            raise ValueError(f"AdaS: Cannot resume from checkpoint without " +
                             "specifying checkpoint dir")
        GLOBALS.CHECKPOINT_PATH.mkdir(exist_ok=True, parents=True)

    with config_path.open() as f:
        GLOBALS.CONFIG = parse_config(yaml.load(f))

    return output_path

def initialize(args: APNamespace, new_network, beta=None, new_threshold=None, new_threshold_kernel=None, scheduler=None, init_lr=None, load_config=True, trial=-1):
    def get_loss(loss: str) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss() if loss == 'cross_entropy' else None

    root_path = Path(args.root).expanduser()
    data_path = root_path / Path(args.data).expanduser()
    config_path = Path(args.config).expanduser()
    #parse from yaml

    if load_config:
        with config_path.open() as f:
            GLOBALS.CONFIG = parse_config(yaml.load(f))

    if scheduler != None:
        GLOBALS.CONFIG['lr_scheduler']='StepLR'

    if init_lr != None:
        GLOBALS.CONFIG['init_lr'] = init_lr
    
    print('~~~GLOBALS.CONFIG:~~~')
    print(GLOBALS.CONFIG)
    print("Adas: Argument Parser Options")
    print("-"*45)
    print(f"    {'config':<20}: {args.config:<40}")
    print(f"    {'data':<20}: {str(Path(args.root) / args.data):<40}")
    print(f"    {'output':<20}: {str(Path(args.root) / args.output):<40}")
    #print(f"    {'checkpoint':<20}: " + No checkpoints used
    #      f"{str(Path(args.root) / args.checkpoint):<40}")
    print(f"    {'root':<20}: {args.root:<40}")
    #print(f"    {'resume':<20}: {'True' if args.resume else 'False':<20}") No checkpoints / resumes used
    
    print("\nAdas: Train: Config")
    print(f"    {'Key':<20} {'Value':<20}")
    print("-"*45)
    
    for k, v in GLOBALS.CONFIG.items():
        if isinstance(v, list):
            print(f"    {k:<20} {v}")
        else:
            print(f"    {k:<20} {v:<20}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"AdaS: Pytorch device is set to {device}")
    # global best_acc
    GLOBALS.BEST_ACC = 0  # best test accuracy - needs to be set to 0 every trial / full train

    '''
    Early stopping stuff
    '''
    if np.less(float(GLOBALS.CONFIG['early_stop_threshold']), 0):
        print(
            "AdaS: Notice: early stop will not be used as it was set to " +
            f"{GLOBALS.CONFIG['early_stop_threshold']}, training till " +
            "completion")
    elif GLOBALS.CONFIG['optim_method'] != 'SGD' and \
            GLOBALS.CONFIG['lr_scheduler'] != 'AdaS':
        print(
            "AdaS: Notice: early stop will not be used as it is not SGD with" +
            " AdaS, training till completion")
        GLOBALS.CONFIG['early_stop_threshold'] = -1

    train_loader, test_loader = get_data(
                root=data_path,
                dataset=GLOBALS.CONFIG['dataset'],
                mini_batch_size=GLOBALS.CONFIG['mini_batch_size'],
                cutout=GLOBALS.CONFIG['cutout'],
                cutout_length = GLOBALS.CONFIG['cutout_length'])

    GLOBALS.PERFORMANCE_STATISTICS = {}
    #Gets initial conv size list (string) from config yaml file and converts into int list
    init_conv = [int(conv_size) for conv_size in GLOBALS.CONFIG['init_conv_setting'].split(',')]

    '''if GLOBALS.CONFIG['blocks_per_superblock']==2:
        GLOBALS.super1_idx = [64,64,64,64,64]
        GLOBALS.super2_idx = [64,64,64,64]
        GLOBALS.super3_idx = [64,64,64,64]
        GLOBALS.super4_idx = [64,64,64,64]
    else:
        GLOBALS.super1_idx = [64,64,64,64,64,64,64]
        GLOBALS.super2_idx = [64,64,64,64,64,64]
        GLOBALS.super3_idx = [64,64,64,64,64,64]
        GLOBALS.super4_idx = [64,64,64,64,64,64]'''

    GLOBALS.index_used = GLOBALS.super1_idx + GLOBALS.super2_idx + GLOBALS.super3_idx + GLOBALS.super4_idx

    """
    if GLOBALS.FIRST_INIT == True and new_network == 0:
        print('FIRST_INIT==True, GETTING NET FROM CONFIG')
        GLOBALS.NET = get_net(
                    GLOBALS.CONFIG['network'], num_classes=10 if
                    GLOBALS.CONFIG['dataset'] == 'CIFAR10' else 100 if
                    GLOBALS.CONFIG['dataset'] == 'CIFAR100'
                    else 1000, init_adapt_conv_size=init_conv)
        GLOBALS.FIRST_INIT = False
    else:
        print('GLOBALS.FIRST_INIT IS FALSE LOADING IN NETWORK FROM UPDATE (Fresh weights)')
        GLOBALS.NET = new_network
    """

    GLOBALS.METRICS = Metrics(list(new_network.parameters()),p=GLOBALS.CONFIG['p'])
    # print("Memory before sending model to cuda:", torch.cuda.memory_allocated(0))
    model = new_network.to(device)
    # print("Memory after sending model to cuda:", torch.cuda.memory_allocated(0))
    GLOBALS.CRITERION = get_loss(GLOBALS.CONFIG['loss'])
    GLOBALS.CRITERION_SMOOTH = CrossEntropyLabelSmooth(1000, GLOBALS.CONFIG['label_smooth'])

    if beta != None:
        GLOBALS.CONFIG['beta']=beta

    if new_threshold != None:
        GLOBALS.CONFIG['delta_threshold']=new_threshold

    if new_threshold_kernel != None:
        GLOBALS.CONFIG['delta_threshold_kernel']=new_threshold_kernel

    if args.train_num > 0:
        GLOBALS.CONFIG['train_num'] = args.train_num

    optimizer, scheduler = get_optimizer_scheduler(
            net_parameters=model.parameters(),
            listed_params=list(model.parameters()),
            # init_lr=learning_rate,
            # optim_method=GLOBALS.CONFIG['optim_method'],
            # lr_scheduler=GLOBALS.CONFIG['lr_scheduler'],
            train_loader_len=len(train_loader),
            config=GLOBALS.CONFIG)

    GLOBALS.EARLY_STOP = EarlyStop(
                patience=int(GLOBALS.CONFIG['early_stop_patience']),
                threshold=float(GLOBALS.CONFIG['early_stop_threshold']))

    #GLOBALS.OPTIMIZER = optimizer
    if device == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

    return train_loader,test_loader,device,optimizer,scheduler,model

def free_cuda_memory():

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def get_latest_completed_trial(trial_dir):
    file_list = os.listdir(trial_dir)
    interrupted_trial = None
    latest_trial = 0
    for file in file_list:
        if file.startswith("AdaS_adapt_trial=") and file.endswith('.xlsx'):
            #Check if this is the latest trial!
            current_trial = int(re.findall(r"\d+", file)[0])
            latest_trial = max(latest_trial, current_trial)

            file_path = os.path.join(trial_dir, file)
            df_trial = pd.read_excel(file_path, index_col=0)
            check = [col for col in df_trial if col == "test_acc_epoch_{}".format(GLOBALS.CONFIG['epochs_per_trial'] - 1)]
            if len(check) == 0 and interrupted_trial == None:
                interrupted_trial = current_trial
            elif len(check) == 0 and interrupted_trial != None:
                interrupted_trial = min(interrupted_trial, current_trial)

    if interrupted_trial == None:
        interrupted_trial = latest_trial + 1

    print("Interrupted Trial found:", interrupted_trial)
    return interrupted_trial


"""
def get_max_ranks_by_layer(path=GLOBALS.EXCEL_PATH):
    '''
    -returns 2 36-lists of the maximum value of the input and output ranks from the datafile produced after one adapting trial
    '''
    sheet = pd.read_excel(path,index_col=0)
    out_rank_col = [col for col in sheet if col.startswith('out_rank')]
    in_rank_col = [col for col in sheet if col.startswith('in_rank')]

    out_ranks = sheet[out_rank_col]
    in_ranks = sheet[in_rank_col]

    out_max_ranks = out_ranks.max(axis=1)
    in_max_ranks = in_ranks.max(axis=1)

    out_max_ranks = out_max_ranks.tolist()
    in_max_ranks = in_max_ranks.tolist()

    return in_max_ranks,out_max_ranks

def new_output_sizes(current_conv_sizes,ranks,threshold):
    scaling_factor=np.subtract(ranks,threshold)
    new_conv_sizes = np.multiply(current_conv_sizes,np.add(1,scaling_factor))
    new_conv_sizes = [int(i) for i in new_conv_sizes]
    print(scaling_factor,'Scaling Factor')
    print(current_conv_sizes, 'CURRENT CONV SIZES')
    print(new_conv_sizes,'NEW CONV SIZES')
    return new_conv_sizes

def nearest_upper_odd(list_squared_kernel_size):
    for superblock in range(len(list_squared_kernel_size)):
        list_squared_kernel_size[superblock] = (np.ceil(np.sqrt(list_squared_kernel_size[superblock])) // 2 * 2 + 1).tolist()
    return list_squared_kernel_size
"""
"""
def create_full_data_file(new_network,full_save_file,full_fresh_file,output_path_string_full_train, debug=False):
    parameter_data = pd.DataFrame(columns=['Training Accuracy (%)','Training Loss','Test Accuracy (%)','Test Loss','GMacs','GFlops','Parameter Count (M)'])

    #full_save_dfs=pd.read_excel(full_save_file)
    full_fresh_dfs=pd.read_excel(full_fresh_file)

    #final_epoch_save=full_save_dfs.columns[-1][(full_save_dfs.columns[-1].index('epoch_')+6):]
    final_epoch_fresh=full_fresh_dfs.columns[-1][(full_fresh_dfs.columns[-1].index('epoch_')+6):]

    #full_save_accuracy = full_save_dfs['test_acc_epoch_'+str(final_epoch_save)][0]*100
    full_train_accuracy = full_fresh_dfs['train_acc_epoch_'+str(final_epoch_fresh)][0]*100
    full_test_accuracy = full_fresh_dfs['test_acc_epoch_' + str(final_epoch_fresh)][0] * 100
    #full_save_loss = full_save_dfs['train_loss_epoch_'+str(final_epoch_save)][0]
    full_train_loss = full_fresh_dfs['train_loss_epoch_'+str(final_epoch_fresh)][0]
    full_test_loss = full_fresh_dfs['test_loss_epoch_' + str(final_epoch_fresh)][0]

    if debug==True:
        macs=0
        params=0
    else:
        macs, params = get_model_complexity_info(new_network, (3,32,32), as_strings=False,print_per_layer_stat=False, verbose=True)

    #save_parameter_size_list = [full_save_accuracy,full_save_loss,int(macs)/1000000000,2*int(macs)/1000000000,int(params)/1000000]
    fresh_parameter_size_list = [full_train_accuracy,full_train_loss, full_test_accuracy, full_test_loss, int(macs)/1000000000,2*int(macs)/1000000000,int(params)/1000000]
    #parameter_data.loc[len(parameter_data)] = save_parameter_size_list
    parameter_data.loc[len(parameter_data)] = fresh_parameter_size_list
    if platform.system() == 'Windows':
        parameter_data.to_excel(output_path_string_full_train+'\\'+'adapted_parameters.xlsx')
    else:
        parameter_data.to_excel(output_path_string_full_train+'/'+'adapted_parameters.xlsx')

    return True
"""

"""
def get_output_sizes(file_name):
    outputs=pd.read_excel(file_name)
    output_sizes=outputs.iloc[-1,1:]
    output_sizes=output_sizes.tolist()
    output_sizes_true=[ast.literal_eval(i) for i in output_sizes]
    print(output_sizes_true,'Output sizes frome excel')
    return output_sizes_true
"""

def run_epochs(trial, model, epochs, train_loader, test_loader,
               device, optimizer, scheduler, output_path):
    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'
    print('------------------------------' + slash)
    if GLOBALS.CONFIG['lr_scheduler'] == 'AdaS':
        if GLOBALS.FULL_TRAIN == False:
            xlsx_name = \
                slash + f"AdaS_adapt_trial={trial}_" +\
                f"net={GLOBALS.CONFIG['network']}_" +\
                f"{GLOBALS.CONFIG['init_lr']}_dataset=" +\
                f"{GLOBALS.CONFIG['dataset']}.xlsx"
        else:
            if GLOBALS.FULL_TRAIN_MODE == 'last_trial':
                xlsx_name = \
                    slash + f"AdaS_last_iter_fulltrain_trial={trial}_" +\
                    f"net={GLOBALS.CONFIG['network']}_" +\
                    f"dataset=" +\
                    f"{GLOBALS.CONFIG['dataset']}.xlsx"
            elif GLOBALS.FULL_TRAIN_MODE == 'fresh':
                xlsx_name = \
                    slash + f"AdaS_fresh_fulltrain_trial={trial}_" +\
                    f"net={GLOBALS.CONFIG['network']}_" +\
                    f"beta={GLOBALS.CONFIG['beta']}_" +\
                    f"dataset=" +\
                    f"{GLOBALS.CONFIG['dataset']}.xlsx"
            else:
                print('ERROR: INVALID FULL_TRAIN_MODE | Check that the correct full_train_mode strings have been initialized in main file | Should be either fresh, or last_trial')
                sys.exit()
    else:
        if GLOBALS.FULL_TRAIN == False:
            xlsx_name = \
                slash + f"StepLR_adapt_trial={trial}_" +\
                f"net={GLOBALS.CONFIG['network']}_" +\
                f"{GLOBALS.CONFIG['init_lr']}_dataset=" +\
                f"{GLOBALS.CONFIG['dataset']}.xlsx"
        else:
            if GLOBALS.FULL_TRAIN_MODE == 'last_trial':
                xlsx_name = \
                    slash + f"StepLR_last_iter_fulltrain_trial={trial}_" +\
                    f"net={GLOBALS.CONFIG['network']}_" +\
                    f"dataset=" +\
                    f"{GLOBALS.CONFIG['dataset']}.xlsx"
            elif GLOBALS.FULL_TRAIN_MODE == 'fresh':
                xlsx_name = \
                    slash + f"StepLR_fresh_fulltrain_trial={trial}_" +\
                    f"net={GLOBALS.CONFIG['network']}_" +\
                    f"dataset=" +\
                    f"{GLOBALS.CONFIG['dataset']}.xlsx"
            else:
                print('ERROR: INVALID FULL_TRAIN_MODE | Check that the correct full_train_mode strings have been initialized in main file | Should be either fresh, or last_trial')
                sys.exit()
    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'
    xlsx_path = str(output_path) +slash+ xlsx_name

    if GLOBALS.FULL_TRAIN == False:
        filename = \
            slash + f"stats_net={GLOBALS.CONFIG['network']}_AdaS_trial={trial}_" +\
            f"beta={GLOBALS.CONFIG['beta']}_initlr={GLOBALS.CONFIG['init_lr']}_" +\
            f"dataset={GLOBALS.CONFIG['dataset']}.csv"
    else:
        if GLOBALS.FULL_TRAIN_MODE == 'last_trial':
            filename = \
                slash + f"stats_last_iter_net={GLOBALS.CONFIG['network']}_StepLR_trial={trial}_" +\
                f"beta={GLOBALS.CONFIG['beta']}_" +\
                f"dataset={GLOBALS.CONFIG['dataset']}.csv"
        elif GLOBALS.FULL_TRAIN_MODE == 'fresh':
            filename = \
                slash + f"stats_fresh_net={GLOBALS.CONFIG['network']}_StepLR_trial={trial}_" +\
                f"beta={GLOBALS.CONFIG['beta']}_" +\
                f"dataset={GLOBALS.CONFIG['dataset']}.csv"
    GLOBALS.EXCEL_PATH = xlsx_path
    print(GLOBALS.EXCEL_PATH,'SET GLOBALS EXCEL PATH')

    for epoch in epochs:
        start_time = time.time()
        # print(f"AdaS: Epoch {epoch}/{epochs[-1]} Started.")

        # New - Drop Path
        if GLOBALS.CONFIG['drop_path']:
            model.drop_path_prob = GLOBALS.CONFIG['drop_path_prob'] * epoch / GLOBALS.CONFIG['max_epoch']

        train_loss, train_accuracy, test_loss, test_accuracy = \
            epoch_iteration(trial, model, train_loader, test_loader,epoch, device, optimizer, scheduler)

        end_time = time.time()

        if GLOBALS.CONFIG['lr_scheduler'] == 'StepLR':
            scheduler.step()
        total_time = time.time()
        print(
            f"AdaS: Trial {trial}/{GLOBALS.total_trials - 1} | " +
            f"Epoch {epoch}/{epochs[-1]} Ended | " +
            "Total Time: {:.3f}s | ".format(total_time - start_time) +
            "Epoch Time: {:.3f}s | ".format(end_time - start_time) +
            "~Time Left: {:.3f}s | ".format(
                (total_time - start_time) * (epochs[-1] - epoch)),
            "Train Loss: {:.4f}% | Train Acc. {:.4f}% | ".format(
                train_loss,
                train_accuracy) +
            "Test Loss: {:.4f}% | Test Acc. {:.4f}%".format(test_loss,
                                                            test_accuracy))
        df = pd.DataFrame(data=GLOBALS.PERFORMANCE_STATISTICS)

        df.to_excel(xlsx_path)
        if GLOBALS.EARLY_STOP(train_loss):
            print("AdaS: Early stop activated.")
            break

#@Profiler
def epoch_iteration(trial, model, train_loader, test_loader, epoch: int,
                    device, optimizer,scheduler) -> Tuple[float, float]:
    # logging.info(f"Adas: Train: Epoch: {epoch}")
    # global net, performance_statistics, metrics, adas, config
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    top1 = AverageMeter()
    top5 = AverageMeter()

    """train CNN architecture"""
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if GLOBALS.CONFIG['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + batch_idx / len(train_loader))
        optimizer.zero_grad()
        # if GLOBALS.CONFIG['optim_method'] == 'SLS':
        if isinstance(optimizer, SLS):
            def closure():
                outputs = model(inputs)
                loss = GLOBALS.CRITERION(outputs, targets)
                return loss, outputs
            loss, outputs = optimizer.step(closure=closure)
        else:

            #TODO: Revert this if statement when creating separate files
            if GLOBALS.CONFIG['network'] == 'DARTS' or GLOBALS.CONFIG['network'] == 'DARTSPlus':
                outputs, outputs_aux = model(inputs)
                if GLOBALS.CONFIG['dataset'] == 'ImageNet':
                    loss = GLOBALS.CRITERION_SMOOTH(outputs, targets)
                else:
                    loss = GLOBALS.CRITERION(outputs, targets)

                if GLOBALS.CONFIG['auxiliary']:
                    if GLOBALS.CONFIG['dataset'] == 'ImageNet':
                        loss_aux = GLOBALS.CRITERION_SMOOTH(outputs_aux, targets)
                    else:
                        loss_aux = GLOBALS.CRITERION(outputs_aux, targets)
                    loss = loss + GLOBALS.CONFIG['auxiliary_weight'] * loss_aux
                loss.backward()
                if GLOBALS.CONFIG['grad_clip']:
                    nn.utils.clip_grad_norm(model.parameters(), GLOBALS.CONFIG['grad_clip_threshold'])

            else:
                outputs = model(inputs)
                loss = GLOBALS.CRITERION(outputs, targets)
                loss.backward()

            # if GLOBALS.ADAS is not None:
            #     optimizer.step(GLOBALS.METRICS.layers_index_todo,
            #                    GLOBALS.ADAS.lr_vector)
            if isinstance(scheduler, AdaS):
                optimizer.step(GLOBALS.METRICS.layers_index_todo,
                               scheduler.lr_vector)
            # elif GLOBALS.CONFIG['optim_method'] == 'SPS':
            elif isinstance(optimizer, SPS):
                optimizer.step(loss=loss)
            else:
                optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        acc1_temp, acc5_temp = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1_temp[0], inputs.size(0))
        top5.update(acc5_temp[0], inputs.size(0))

        GLOBALS.TRAIN_LOSS = train_loss
        GLOBALS.TRAIN_CORRECT = correct
        GLOBALS.TRAIN_TOTAL = total

        if GLOBALS.CONFIG['lr_scheduler'] == 'OneCycleLR':
            scheduler.step()
        #Update optimizer
        #GLOBALS.OPTIMIZER = optimizer

        # progress_bar(batch_idx, len(train_loader),
        #              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1),
        #                  100. * correct / total, correct, total))

    acc = top1.avg.cpu().item() / 100
    acc5 = top5.avg.cpu().item() / 100

    GLOBALS.PERFORMANCE_STATISTICS[f'train_acc_epoch_{epoch}'] = \
        float(acc)
    GLOBALS.PERFORMANCE_STATISTICS[f'train_acc5_epoch_{epoch}'] = \
        float(acc5)
    GLOBALS.PERFORMANCE_STATISTICS[f'train_loss_epoch_{epoch}'] = \
        train_loss / (batch_idx + 1)

    io_metrics = GLOBALS.METRICS.evaluate(epoch)
    GLOBALS.PERFORMANCE_STATISTICS[f'in_S_epoch_{epoch}'] = \
        io_metrics.input_channel_S
    GLOBALS.PERFORMANCE_STATISTICS[f'out_S_epoch_{epoch}'] = \
        io_metrics.output_channel_S
    GLOBALS.PERFORMANCE_STATISTICS[f'mode12_S_epoch_{epoch}'] = \
        io_metrics.mode_12_channel_S
    GLOBALS.PERFORMANCE_STATISTICS[f'fc_S_epoch_{epoch}'] = \
        io_metrics.fc_S
    GLOBALS.PERFORMANCE_STATISTICS[f'in_rank_epoch_{epoch}'] = \
        io_metrics.input_channel_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'out_rank_epoch_{epoch}'] = \
        io_metrics.output_channel_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'mode12_rank_epoch_{epoch}'] = \
        io_metrics.mode_12_channel_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'fc_rank_epoch_{epoch}'] = \
        io_metrics.fc_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'in_condition_epoch_{epoch}'] = \
        io_metrics.input_channel_condition
    GLOBALS.PERFORMANCE_STATISTICS[f'out_condition_epoch_{epoch}'] = \
        io_metrics.output_channel_condition
    GLOBALS.PERFORMANCE_STATISTICS[f'mode12_condition_epoch_{epoch}'] = \
        io_metrics.mode_12_channel_condition
    # if GLOBALS.ADAS is not None:

    if isinstance(scheduler, AdaS):
        lrmetrics = scheduler.step(epoch, GLOBALS.METRICS)
        GLOBALS.PERFORMANCE_STATISTICS[f'rank_velocity_epoch_{epoch}'] = \
            lrmetrics.rank_velocity
        GLOBALS.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
            lrmetrics.r_conv
    else:
        # if GLOBALS.CONFIG['optim_method'] == 'SLS' or \
        #         GLOBALS.CONFIG['optim_method'] == 'SPS':
        if isinstance(optimizer, SLS) or isinstance(optimizer, SPS):
            GLOBALS.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
                optimizer.state['step_size']
        else:
            GLOBALS.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
                optimizer.param_groups[0]['lr']
    test_loss, test_accuracy, test_acc5 = test_main(model, test_loader, epoch, device, optimizer)

    return (train_loss / (batch_idx + 1), 100. * acc,
            test_loss, 100 * test_accuracy)
