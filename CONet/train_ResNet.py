from typing import Tuple
from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from train_help import *
from pathlib import Path
import os
import platform
import pandas as pd
import global_vars as GLOBALS
from adaptive_graph import create_adaptive_graphs,create_plot,adapted_info_graph,trial_info_graph, stacked_bar_plot
from ptflops import get_model_complexity_info
from models.own_network import DASNet34,DASNet50
import copy
import torch
from resnet_scaling_algorithm import channel_size_adjust_algorithm as delta_scaling
import matplotlib.pyplot as plt
import re
from shutil import copyfile
"""
Contains training code for ResNet
"""

def update_network(new_channel_sizes,new_kernel_sizes):
    new_network=None

    class_num = 0
    if GLOBALS.CONFIG['dataset'] == 'CIFAR10':
        class_num = 10
    elif GLOBALS.CONFIG['dataset'] == 'CIFAR100':
        class_num = 100

    if GLOBALS.CONFIG['network']=='DASNet34':
        new_network=DASNet34(num_classes_input=class_num,new_output_sizes=new_channel_sizes,new_kernel_sizes=new_kernel_sizes)
    elif GLOBALS.CONFIG['network']=='DASNet50':
        new_network=DASNet50(num_classes_input=class_num,new_output_sizes=new_channel_sizes,new_kernel_sizes=new_kernel_sizes)
    return new_network

def create_train_output_file(new_network, full_fresh_file,output_path_string_full_train, debug=False):
    output_file='default.xlsx'
    if platform.system() == 'Windows':
        performance_output_file=output_path_string_full_train + '\\' + 'performance.xlsx'
        auxilary_output_file = output_path_string_full_train + '\\' + 'auxilary.xlsx'
    else:
        performance_output_file = output_path_string_full_train +'/'+ 'performance.xlsx'
        auxilary_output_file = output_path_string_full_train + '/' + 'auxilary.xlsx'
    writer_performance = pd.ExcelWriter(performance_output_file, engine='openpyxl')
    wb_per = writer_performance.book
    writer_auxilary = pd.ExcelWriter(auxilary_output_file, engine='openpyxl')
    wb_aux = writer_auxilary.book

    full_fresh_dfs = pd.read_excel(full_fresh_file)
    final_epoch_fresh = full_fresh_dfs.columns[-1][(full_fresh_dfs.columns[-1].index('epoch_') + 6):]
    performance_data={}
    auxilary_data={}
    if debug==True:
        macs=0
        params=0
    else:
        macs, params = get_model_complexity_info(new_network, (3,32,32), as_strings=False,print_per_layer_stat=False)
    performance_data['Gmac']=int(macs) / 1000000000
    performance_data['GFlop']=2 * int(macs) / 1000000000
    performance_data['parameter count (M)'] = int(params) / 1000000

    num_layer=len(full_fresh_dfs['train_acc_epoch_' + str(0)])
    layer_list=list(range(0,num_layer))
    auxilary_data['layer_index'] = layer_list

    for i in range(int(final_epoch_fresh)+1):
        performance_data['train_acc_epoch_' + str(i)+ " (%)"] = [full_fresh_dfs['train_acc_epoch_' + str(i)][0] *100]
        performance_data['train_loss_epoch_' + str(i)] = [full_fresh_dfs['train_loss_epoch_' + str(i)][0]]
        performance_data['test_acc_epoch_' + str(i)+" (%)"] = [full_fresh_dfs['test_acc_epoch_' + str(i)][0] *100]
        performance_data['test_loss_epoch_' + str(i)] = [full_fresh_dfs['test_loss_epoch_' + str(i)][0]]

        auxilary_data['in_KG_epcho'+str(i)] = full_fresh_dfs['in_S_epoch_' + str(i)]
        auxilary_data['out_KG_epcho'+str(i)] = full_fresh_dfs['out_S_epoch_' + str(i)]
        auxilary_data['in_rank_epcho'+str(i)] = full_fresh_dfs['in_rank_epoch_' + str(i)]
        auxilary_data['out_rank_epcho'+str(i)] = full_fresh_dfs['out_rank_epoch_' + str(i)]
        auxilary_data['in_condition_epcho'+str(i)] = full_fresh_dfs['in_condition_epoch_' + str(i)]
        auxilary_data['out_condition_epcho'+str(i)] = full_fresh_dfs['out_condition_epoch_' + str(i)]

    df_per = pd.DataFrame(performance_data)
    df_per.to_excel(writer_performance, index=False)
    wb_per.save(performance_output_file)

    df_aux = pd.DataFrame(auxilary_data)
    df_aux.to_excel(writer_auxilary, index=False)
    wb_aux.save(auxilary_output_file)

    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'

    #copyfile(GLOBALS.OUTPUT_PATH_STRING + slash + '..' + slash + '..' + slash + '.adas-checkpoint' + slash + 'ckpt.pth',
    #         output_path_string_full_train + slash + 'ckpt.pth')

def run_fresh_full_train(output_sizes,kernel_sizes,epochs,output_path_fulltrain):
    """
    Perform model evaluation for ResNet
    """
    class_num=0
    if GLOBALS.CONFIG['dataset']=='CIFAR10':
        class_num=10
    elif GLOBALS.CONFIG['dataset']=='CIFAR100':
        class_num=100

    if GLOBALS.CONFIG['network']=='DASNet34':
        new_network=DASNet34(num_classes_input=class_num,new_output_sizes=output_sizes,new_kernel_sizes=kernel_sizes)
    elif GLOBALS.CONFIG['network']=='DASNet50':
        new_network=DASNet50(num_classes_input=class_num,new_output_sizes=output_sizes,new_kernel_sizes=kernel_sizes)

    GLOBALS.FIRST_INIT = False

    #optimizer,scheduler=network_initialize(new_network,train_loader)
    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()
    #free_cuda_memory()
    GLOBALS.CONFIG['mini_batch_size'] = GLOBALS.CONFIG['mini_batch_size_full']
    GLOBALS.CONFIG['weight_decay'] = GLOBALS.CONFIG['weight_decay_full']
    train_loader,test_loader,device,optimizer,scheduler, model = initialize(args,new_network,beta=GLOBALS.CONFIG['beta_full'],
                                                                            scheduler=GLOBALS.CONFIG['lr_scheduler_full'], init_lr=GLOBALS.CONFIG['init_lr_full'],
                                                                            load_config=False)

    GLOBALS.FULL_TRAIN = True
    GLOBALS.PERFORMANCE_STATISTICS = {}
    GLOBALS.FULL_TRAIN_MODE = 'fresh'
    GLOBALS.EXCEL_PATH = ''

    for param_tensor in model.state_dict():
        val=param_tensor.find('bn')
        if val==-1:
            continue
        print(param_tensor, "\t", model.state_dict()[param_tensor].size(), 'FRESH')
        #print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor], 'FRESH')
        break

    run_epochs(0, model, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_fulltrain)
    # print("Memory allocated full train:", torch.cuda.memory_allocated(0))
    return model


def create_graphs(trial_info_file_name, adapted_kernel_file_name, adapted_conv_file_name, rank_final_file_name,
                  rank_stable_file_name, out_folder):
    if platform.system == "Windows":
        slash = '\\'
    else:
        slash = '/'
    create_adaptive_graphs(trial_info_file_name, GLOBALS.CONFIG['epochs_per_trial'], GLOBALS.total_trials, out_folder)
    kernel_path = out_folder + slash + 'dynamic_kernel_Size_Plot.png'
    conv_path = out_folder + slash + 'dynamic_layer_Size_Plot.png'
    rank_final_path = out_folder + slash + 'dynamic_rank_final.png'
    rank_stable_path = out_folder + slash + 'dynamic_rank_stable.png'
    output_condition_path = out_folder + slash + 'dynamic_output_condition.png'
    input_condition_path = out_folder + slash + 'dynamic_input_condition.png'
    network_visualize_path = out_folder + slash + 'dynamic_network_Size_Plot.png'
    '''create_layer_plot(conv_data_file_name,GLOBALS.CONFIG['adapt_trials'],conv_path, 'Layer Size')
    #create_layer_plot(rank_final_file_name,GLOBALS.CONFIG['adapt_trials'],rank_final_path, 'Final Rank')
    #create_layer_plot(rank_stable_file_name,GLOBALS.CONFIG['adapt_trials'],rank_stable_path, 'Stable Rank')'''

    last_epoch = GLOBALS.CONFIG['epochs_per_trial'] - 1
    stable_epoch = GLOBALS.CONFIG['stable_epoch']

    shortcut_indexes = []
    old_conv_size_list = [GLOBALS.super1_idx, GLOBALS.super2_idx, GLOBALS.super3_idx, GLOBALS.super4_idx]
    counter = -1
    for j in old_conv_size_list:
        if len(shortcut_indexes) == len(old_conv_size_list) - 1:
            break
        counter += len(j) + 1
        shortcut_indexes += [counter]
    plt.clf()
    stacked_bar_plot(adapted_conv_file_name, network_visualize_path)
    if GLOBALS.CONFIG['kernel_adapt'] != 0:
        plt.clf()
        adapted_info_graph(adapted_kernel_file_name, GLOBALS.CONFIG['adapt_trials_kernel'], kernel_path, 'Kernel Size',
                           last_epoch)
    plt.clf()
    adapted_info_graph(adapted_conv_file_name, GLOBALS.CONFIG['adapt_trials'], conv_path, 'Layer Size', last_epoch)
    plt.clf()
    trial_info_graph(trial_info_file_name, GLOBALS.total_trials, len(GLOBALS.index_used) + 3, rank_final_path,
                     'Final Rank', 'out_rank_epoch_', shortcut_indexes, last_epoch)
    plt.clf()
    trial_info_graph(trial_info_file_name, GLOBALS.total_trials, len(GLOBALS.index_used) + 3, rank_stable_path,
                     'Stable Rank', 'out_rank_epoch_', shortcut_indexes, stable_epoch)
    plt.clf()
    trial_info_graph(trial_info_file_name, GLOBALS.total_trials, len(GLOBALS.index_used) + 3, output_condition_path,
                     'Output Condition', 'out_condition_epoch_', shortcut_indexes, last_epoch)
    plt.clf()
    trial_info_graph(trial_info_file_name, GLOBALS.total_trials, len(GLOBALS.index_used) + 3, input_condition_path,
                     'Input Condition', 'in_condition_epoch_', shortcut_indexes, last_epoch)
    plt.clf()
    return True


def run_trials(epochs, output_path_train, new_threshold=None):

    """
    Perform Channel Search for ResNet
    """
    last_operation, factor_scale, delta_percentage, last_operation_kernel, factor_scale_kernel, delta_percentage_kernel = [], [], [], [], [], []
    parameter_type = GLOBALS.CONFIG['parameter_type']
    trial_dir = os.path.join(GLOBALS.OUTPUT_PATH_STRING, 'Trials')
    print(trial_dir)

    kernel_begin_trial = 0

    def check_last_operation(last_operation, last_operation_kernel, kernel_begin_trial):
        all_channels_stopped = True
        for blah in last_operation:
            for inner in blah:
                if inner != 0:
                    all_channels_stopped = False
        all_kernels_stopped = True
        # if kernel_begin_trial!=0:
        for blah in last_operation_kernel:
            for inner in blah:
                if inner != 0:
                    all_kernels_stopped = False
        return all_channels_stopped, all_kernels_stopped

    def get_shortcut_indexes(conv_size_list):
        shortcut_indexes = []
        counter = -1
        for j in conv_size_list:
            if len(shortcut_indexes) == len(conv_size_list) - 1:
                break
            counter += len(j) + 1
            shortcut_indexes += [counter]
        return shortcut_indexes

    def initialize_dataframes_and_lists():
        conv_data = pd.DataFrame(columns=['superblock1', 'superblock2', 'superblock3', 'superblock4'])
        kernel_data = pd.DataFrame(columns=['superblock1', 'superblock2', 'superblock3', 'superblock4'])
        rank_final_data = pd.DataFrame(columns=['superblock1', 'superblock2', 'superblock3', 'superblock4'])
        rank_stable_data = pd.DataFrame(columns=['superblock1', 'superblock2', 'superblock3', 'superblock4'])
        conv_size_list = [GLOBALS.super1_idx, GLOBALS.super2_idx, GLOBALS.super3_idx, GLOBALS.super4_idx]
        kernel_size_list = [GLOBALS.super1_kernel_idx, GLOBALS.super2_kernel_idx, GLOBALS.super3_kernel_idx,
                            GLOBALS.super4_kernel_idx]

        conv_data.loc[0] = conv_size_list
        kernel_data.loc[0] = kernel_size_list
        delta_info = pd.DataFrame(columns=['delta_percentage', 'factor_scale', 'last_operation'])
        delta_info_kernel = pd.DataFrame(
            columns=['delta_percentage_kernel', 'factor_scale_kernel', 'last_operation_kernel'])
        return conv_data, kernel_data, rank_final_data, rank_stable_data, delta_info, delta_info_kernel, conv_size_list, kernel_size_list

    def should_break(i, all_channels_stopped, all_kernels_stopped, kernel_begin_trial, parameter_type):
        break_loop = False
        if (all_channels_stopped == True and kernel_begin_trial == 0) or i == GLOBALS.CONFIG['adapt_trials']:
            GLOBALS.CONFIG['adapt_trials'] = i
            parameter_type = 'kernel'
            kernel_begin_trial = i
            if GLOBALS.CONFIG['adapt_trials_kernel'] == 0 or GLOBALS.CONFIG['kernel_adapt'] == 0:
                print('ACTIVATED IF STATEMENT 1 FOR SOME STUPID REASON')
                break_loop = True

        if (all_kernels_stopped == True or i == kernel_begin_trial + GLOBALS.CONFIG[
            'adapt_trials_kernel']) and kernel_begin_trial != 0:  # and kernel_begin_trial!=0:
            print('ACTIVATED IF STATEMENT 2 FOR SOME EVEN STUPIDER REASON')
            break_loop = True
        return kernel_begin_trial, parameter_type, break_loop

    #####################################################################################################################################
    conv_data, kernel_data, rank_final_data, rank_stable_data, delta_info, delta_info_kernel, conv_size_list, kernel_size_list = initialize_dataframes_and_lists()
    shortcut_indexes = get_shortcut_indexes(conv_size_list)
    # print("Memory before allocation:", torch.cuda.memory_allocated(0))
    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()
    if GLOBALS.CONFIG['dataset'] == 'CIFAR10':
        class_num = 10
    elif GLOBALS.CONFIG['dataset'] == 'CIFAR100':
        class_num = 100
    new_network = DASNet34(num_classes_input=class_num)
    train_loader, test_loader, device, optimizer, scheduler, model = initialize(args, new_network)

    interrupted_trial = 0 # Determines at which trial we will resume!
    if args.resume_search is False:
        run_epochs(0, model, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_train)
        # print("Memory after first trial:", torch.cuda.memory_allocated(0))
    else:
        interrupted_trial = get_latest_completed_trial(trial_dir)

    del model
    del train_loader
    del test_loader
    del optimizer
    del scheduler

    free_cuda_memory()

    print('~~~First run_epochs done.~~~')

    if (GLOBALS.CONFIG['kernel_adapt'] == 0):
        GLOBALS.CONFIG['adapt_trials_kernel'] = 0

    GLOBALS.total_trials = GLOBALS.CONFIG['adapt_trials'] + GLOBALS.CONFIG['adapt_trials_kernel']
    for i in range(1, GLOBALS.total_trials):
        """
        if (GLOBALS.CONFIG['kernel_adapt'] == 0):
            GLOBALS.CONFIG['adapt_trials_kernel'] = 0
        if kernel_begin_trial != 0:
            if (i > (GLOBALS.total_trials // 2 - kernel_begin_trial)) and all_channels_stopped == True:
                GLOBALS.min_kernel_size_1 = GLOBALS.CONFIG['min_kernel_size']
                GLOBALS.CONFIG['min_kernel_size'] = GLOBALS.CONFIG['min_kernel_size_2']
                
        """
        '------------------------------------------------------------------------------------------------------------------------------------------------'
        """
        last_operation, last_operation_kernel, factor_scale, factor_scale_kernel, new_channel_sizes, new_kernel_sizes, delta_percentage, delta_percentage_kernel, rank_averages_final, rank_averages_stable = delta_scaling(
            conv_size_list, kernel_size_list, shortcut_indexes, last_operation, factor_scale, delta_percentage,
            last_operation_kernel, factor_scale_kernel, delta_percentage_kernel, parameter_type=parameter_type)
        """

        new_channel_sizes, delta_percentage, last_operation, factor_scale, \
        cell_list_rank  = \
            delta_scaling(conv_size_list, GLOBALS.CONFIG['delta_threshold'], \
                            GLOBALS.CONFIG['min_scale_limit'], GLOBALS.CONFIG['mapping_condition_threshold'], \
                            GLOBALS.CONFIG['min_conv_size'], GLOBALS.CONFIG['max_conv_size'],
                            trial_dir, i - 1, last_operation, factor_scale)
        '------------------------------------------------------------------------------------------------------------------------------------------------'

        print(last_operation_kernel, 'LAST OPERATION KERNEL FOR TRIAL ' + str(i))

        """
        all_channels_stopped, all_kernels_stopped = check_last_operation(last_operation, last_operation_kernel,
                                                                         kernel_begin_trial)
        print(all_channels_stopped, all_kernels_stopped, 'BREAK VALUES!')
        kernel_begin_trial, parameter_type, break_loop = should_break(i, all_channels_stopped, all_kernels_stopped,
                                                                      kernel_begin_trial, parameter_type)
        if break_loop == True:
            GLOBALS.total_trials = i
            break
        """

        last_operation_copy, factor_scale_copy, delta_percentage_copy = copy.deepcopy(
            last_operation), copy.deepcopy(factor_scale), copy.deepcopy(delta_percentage)
       # last_operation_kernel_copy, factor_scale_kernel_copy, delta_percentage_kernel_copy = copy.deepcopy(
       #     last_operation_kernel), copy.deepcopy(factor_scale_kernel), copy.deepcopy(delta_percentage_kernel)
        conv_size_list = copy.deepcopy(new_channel_sizes)
        # old_kernel_size_list = copy.deepcopy(kernel_size_list)
        # kernel_size_list = copy.deepcopy(new_kernel_sizes)

        print('~~~Writing to Dataframe~~~')
        if parameter_type == 'channel':
            conv_data.loc[i] = new_channel_sizes
            delta_info.loc[i] = [delta_percentage_copy, factor_scale_copy, last_operation_copy]
        
        # rank_final_data.loc[i] = rank_averages_final_copy
        # rank_stable_data.loc[i] = rank_averages_stable_copy

        print('~~~Starting Conv parameter_typements~~~')

        new_network = update_network(new_channel_sizes, None)

        print('~~~Initializing the new model~~~')

        train_loader, test_loader, device, optimizer, scheduler, model = initialize(args, new_network,
                                                                                    new_threshold_kernel=new_threshold)
        #print("Memory allocated before trial:", torch.cuda.memory_allocated(0))
        epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])

        if i < interrupted_trial:
            print('~~~Using previous training data~~~')

        else:
            print('~~~Training with new model~~~')
            run_epochs(i, model, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_train)
        # print("Memory allocated after trial:", torch.cuda.memory_allocated(0))

        del model
        del train_loader
        del test_loader
        del optimizer
        del scheduler

        free_cuda_memory()

        #Use default
        new_kernel_sizes = [GLOBALS.super1_kernel_idx,GLOBALS.super2_kernel_idx,GLOBALS.super3_kernel_idx,GLOBALS.super4_kernel_idx]

    return kernel_data, conv_data, rank_final_data, rank_stable_data, new_channel_sizes, new_kernel_sizes, delta_info, delta_info_kernel


def create_trial_data_file(kernel_data, conv_data, delta_info_kernel, delta_info, rank_final_data, rank_stable_data,
                           output_path_string_trials, output_path_string_graph_files, output_path_string_modelweights):
    # parameter_data.to_excel(output_path_string_trials+'\\'+'adapted_parameters.xlsx')
    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'
    try:
        delta_info_kernel.to_excel(output_path_string_trials + slash + 'adapted_delta_info_kernel.xlsx')
        delta_info.to_excel(output_path_string_trials + slash + 'adapted_delta_info.xlsx')
        # kernel_data.to_excel(output_path_string_trials + slash + 'adapted_kernels.xlsx')
        conv_data.to_excel(output_path_string_trials + slash + 'adapted_architectures.xlsx')
        # rank_final_data.to_excel(output_path_string_trials + slash + 'adapted_rank_final.xlsx')
        # rank_stable_data.to_excel(output_path_string_trials + slash + 'adapted_rank_stable.xlsx')
        """
        create_graphs(GLOBALS.EXCEL_PATH, output_path_string_trials + slash + 'adapted_kernels.xlsx',
                      output_path_string_trials + slash + 'adapted_architectures.xlsx',
                      output_path_string_trials + slash + 'adapted_rank_final.xlsx',
                      output_path_string_trials + slash + 'adapted_rank_stable.xlsx', output_path_string_graph_files)
        """
    except Exception as ex:
        print('COULD NOT CREATE GRAPHS')
        print(ex)
    # torch.save(GLOBALS.NET.state_dict(), output_path_string_modelweights+'\\'+'model_state_dict')