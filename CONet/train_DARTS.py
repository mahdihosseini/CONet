from typing import Tuple
from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from train_help import *
from pathlib import Path
import os
import platform
import time
import pandas as pd
import numpy as np
import global_vars as GLOBALS
from ptflops import get_model_complexity_info
import copy
import torch

#DARTS Model files
from model import NetworkCIFAR as Network
from model import NetworkImageNet as Network_ImageNet
import genotypes
from darts_scaling_algorithms import channel_size_adjust_algorithm as DARTS_algorithm
from dartsplus_scaling_algorithms import channel_size_adjust_algorithm as DARTSPlus_algorithm
from shutil import copyfile
import re


"""
Contains training code for DARTS and DARTSPlus
"""

def update_network_DARTS(new_cell_list = None,new_sep_conv_list = None):

    if GLOBALS.CONFIG['network'] == 'DARTS':
        arch = "DARTS"
    elif GLOBALS.CONFIG['network'] == 'DARTSPlus':
        arch = "DARTS_PLUS_CIFAR100"
    genotype = eval("genotypes.%s" % arch)

    if GLOBALS.CONFIG["dataset"] == 'CIFAR10':
        fc_dim = 10
    elif GLOBALS.CONFIG["dataset"] == 'CIFAR100':
        fc_dim = 100
    elif GLOBALS.CONFIG["dataset"] == 'ImageNet':
        fc_dim = 1000

    assert GLOBALS.CONFIG["num_cells"] == 7 or GLOBALS.CONFIG["num_cells"] == 14 or GLOBALS.CONFIG["num_cells"] == 20


    if new_cell_list == None:
        if GLOBALS.CONFIG["num_cells"] == 20:
            if arch == "DARTS":
                new_cell_list = GLOBALS.DARTS_cell_list_20
            else:
                new_cell_list = GLOBALS.DARTSPlus_cell_list_20

        elif GLOBALS.CONFIG["num_cells"] == 14:
            if arch == "DARTS":
                new_cell_list = GLOBALS.DARTS_cell_list_14
            else:
                new_cell_list = GLOBALS.DARTSPlus_cell_list_14
        else:
            if arch == "DARTS":
                new_cell_list = GLOBALS.DARTS_cell_list_7
            else:
                new_cell_list = GLOBALS.DARTSPlus_cell_list_7
    if new_sep_conv_list == None:
        if GLOBALS.CONFIG["num_cells"] == 20:
            if arch == "DARTS":
                new_sep_conv_list = GLOBALS.DARTS_sep_conv_list_20
            else:
                new_sep_conv_list = GLOBALS.DARTSPlus_sep_conv_list_20
        elif GLOBALS.CONFIG["num_cells"] == 14:
            if arch == "DARTS":
                new_sep_conv_list = GLOBALS.DARTS_sep_conv_list_14
            else:
                new_sep_conv_list = GLOBALS.DARTSPlus_sep_conv_list_14
        else:
            if arch == "DARTS":
                new_sep_conv_list = GLOBALS.DARTS_sep_conv_list_7
            else:
                new_sep_conv_list = GLOBALS.DARTSPlus_sep_conv_list_7

    #The 10 is the number of classes in CIFAR10
    if GLOBALS.CONFIG["dataset"] == 'CIFAR10' or GLOBALS.CONFIG["dataset"] == 'CIFAR100':
        new_network = Network(new_cell_list, new_sep_conv_list, fc_dim, GLOBALS.CONFIG["num_cells"], GLOBALS.CONFIG['auxiliary'], genotype, arch)
    elif GLOBALS.CONFIG["dataset"] == 'ImageNet':
        new_network = Network_ImageNet(new_cell_list, new_sep_conv_list, fc_dim, GLOBALS.CONFIG["num_cells"],
                          GLOBALS.CONFIG['auxiliary'], genotype, arch)
    print("Cell List:", new_cell_list)
    print("Sep Conv List:", new_sep_conv_list)
    new_network.drop_path_prob = 0  # Need to update this
    return new_network

def find_best_acc_epoch(df):
    test_accs = list()
    cols = [col for col in df if col.startswith('test_acc_epoch')]
    for col in cols:
        temp = float(df[col][0])*100
        test_accs.append(temp)
    return np.array(test_accs).argmax()

def create_full_data_file_DARTS(new_network,full_fresh_file,output_path_string_full_train):
    parameter_data = pd.DataFrame(columns=['Accuracy (%)','Training Loss','GMacs','GFlops','Parameter Count (M)'])

    #full_save_dfs=pd.read_excel(full_save_file)
    full_fresh_dfs=pd.read_excel(full_fresh_file)

    #final_epoch_save=full_save_dfs.columns[-1][(full_save_dfs.columns[-1].index('epoch_')+6):]
    final_epoch_fresh=full_fresh_dfs.columns[-1][(full_fresh_dfs.columns[-1].index('epoch_')+6):]
    #best acc
    best_epoch_fresh = find_best_acc_epoch(full_fresh_dfs)

    #full_save_accuracy = full_save_dfs['test_acc_epoch_'+str(final_epoch_save)][0]*100
    full_fresh_accuracy = full_fresh_dfs['test_acc_epoch_'+str(best_epoch_fresh)][0]*100
    #full_save_loss = full_save_dfs['train_loss_epoch_'+str(final_epoch_save)][0]
    full_fresh_loss = full_fresh_dfs['train_loss_epoch_'+str(best_epoch_fresh)][0]


    if GLOBALS.CONFIG['dataset'] == 'CIFAR10' or GLOBALS.CONFIG['dataset'] == 'CIFAR100':
        macs, params = get_model_complexity_info(new_network, (3,32,32), as_strings=False,print_per_layer_stat=False)
    elif GLOBALS.CONFIG['dataset'] == 'ImageNet':
        macs, params = get_model_complexity_info(new_network, (3, 224, 224), as_strings=False,
                                                 print_per_layer_stat=False)

    #save_parameter_size_list = [full_save_accuracy,full_save_loss,int(macs)/1000000000,2*int(macs)/1000000000,int(params)/1000000]
    fresh_parameter_size_list = [full_fresh_accuracy,full_fresh_loss,int(macs)/1000000000,2*int(macs)/1000000000,int(params)/1000000]
    #parameter_data.loc[len(parameter_data)] = save_parameter_size_list
    parameter_data.loc[len(parameter_data)] = fresh_parameter_size_list
    if platform.system() == 'Windows':
        parameter_data.to_excel(output_path_string_full_train+'\\'+'adapted_parameters.xlsx')
    else:
        parameter_data.to_excel(output_path_string_full_train+'/'+'adapted_parameters.xlsx')

    # Copied from master
    output_file = 'default.xlsx'
    if platform.system() == 'Windows':
        performance_output_file = output_path_string_full_train + '\\' + 'performance.xlsx'
        auxilary_output_file = output_path_string_full_train + '\\' + 'auxilary.xlsx'
    else:
        performance_output_file = output_path_string_full_train + '/' + 'performance.xlsx'
        auxilary_output_file = output_path_string_full_train + '/' + 'auxilary.xlsx'
    writer_performance = pd.ExcelWriter(performance_output_file, engine='openpyxl')
    wb_per = writer_performance.book
    writer_auxilary = pd.ExcelWriter(auxilary_output_file, engine='openpyxl')
    wb_aux = writer_auxilary.book

    performance_data = {}
    auxilary_data = {}
    performance_data['Gmac'] = int(macs) / 1000000000
    performance_data['GFlop'] = 2 * int(macs) / 1000000000
    performance_data['parameter count (M)'] = int(params) / 1000000

    num_layer = len(full_fresh_dfs['train_acc_epoch_' + str(0)])
    layer_list = list(range(0, num_layer))
    auxilary_data['layer_index'] = layer_list

    for i in range(int(final_epoch_fresh) + 1):
        performance_data['train_acc_epoch_' + str(i) + " (%)"] = [full_fresh_dfs['train_acc_epoch_' + str(i)][0] * 100]
        performance_data['train_loss_epoch_' + str(i)] = [full_fresh_dfs['train_loss_epoch_' + str(i)][0]]
        performance_data['test_acc_epoch_' + str(i) + " (%)"] = [full_fresh_dfs['test_acc_epoch_' + str(i)][0] * 100]
        performance_data['test_loss_epoch_' + str(i)] = [full_fresh_dfs['test_loss_epoch_' + str(i)][0]]

        auxilary_data['in_KG_epcho' + str(i)] = full_fresh_dfs['in_S_epoch_' + str(i)]
        auxilary_data['out_KG_epcho' + str(i)] = full_fresh_dfs['out_S_epoch_' + str(i)]
        auxilary_data['in_rank_epcho' + str(i)] = full_fresh_dfs['in_rank_epoch_' + str(i)]
        auxilary_data['out_rank_epcho' + str(i)] = full_fresh_dfs['out_rank_epoch_' + str(i)]
        auxilary_data['in_condition_epcho' + str(i)] = full_fresh_dfs['in_condition_epoch_' + str(i)]
        auxilary_data['out_condition_epcho' + str(i)] = full_fresh_dfs['out_condition_epoch_' + str(i)]

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

    #Hard coded path, copy into adas search folder
    # copyfile(GLOBALS.OUTPUT_PATH_STRING +slash+'..'+slash+'..'+slash+'.adas-checkpoint'+slash+'ckpt.pth', output_path_string_full_train + slash + 'ckpt.pth')

    return True

def run_fresh_full_train_DARTS(epochs,output_path_fulltrain, cell_list = None, sep_conv_list = None):
    """
    Perform model evaluation for DARTS/DARTS+
    """

    GLOBALS.FIRST_INIT = False

    #optimizer,scheduler=network_initialize(new_network,train_loader)
    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()
    #Just to build directories. Settings get overwritten below

    """
    if cell_list != None:
        print ("Full Train Cell Architecture:", cell_list)
    else:
        if GLOBALS.CONFIG["num_cells"] == 20:
            print ("Full Train Cell Architecture:", GLOBALS.cell_list_20)
        else:
            print ("Full Train Cell Architecture:", GLOBALS.cell_list_7)


    if sep_conv_list != None:
        print ("Full Train Sep Conv Architecture:", sep_conv_list)
    else:
        if GLOBALS.CONFIG["num_cells"] == 20:
            print("Full Train Cell Architecture:", GLOBALS.sep_conv_list_20)
        else:
            print("Full Train Cell Architecture:", GLOBALS.sep_conv_list_7)
            
    """

    #Set all DARTS Hyperparamter to true for full train
    GLOBALS.CONFIG['drop_path'] = GLOBALS.CONFIG['drop_path_full']
    GLOBALS.CONFIG['auxiliary'] = GLOBALS.CONFIG['auxiliary_full']
    GLOBALS.CONFIG['cutout'] = GLOBALS.CONFIG['cutout_full']
    GLOBALS.CONFIG['grad_clip'] = GLOBALS.CONFIG['grad_clip_full']
    GLOBALS.CONFIG['mini_batch_size'] = GLOBALS.CONFIG['mini_batch_size_full']
    GLOBALS.CONFIG['weight_decay'] = GLOBALS.CONFIG['weight_decay_full']

    new_network = update_network_DARTS(cell_list, sep_conv_list)


    train_loader,test_loader,device,optimizer,scheduler, model = initialize(args, new_network,beta=GLOBALS.CONFIG['beta_full'],scheduler=GLOBALS.CONFIG['lr_scheduler_full'], load_config=False)

    GLOBALS.FULL_TRAIN = True
    GLOBALS.PERFORMANCE_STATISTICS = {}
    GLOBALS.FULL_TRAIN_MODE = 'fresh' #
    GLOBALS.EXCEL_PATH = ''


    run_epochs(0, model, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_fulltrain)

    #Initializing again to remove auxiliary head so it does not get added in param / GMAC count.
    print("Running initialize again to remove auxiliary head for param / gmac count")
    GLOBALS.CONFIG['auxiliary'] = False
    #initialize(args_true,beta= GLOBALS.CONFIG['beta_full'],new_cell_list=cell_list, new_sep_conv_list=sep_conv_list, scheduler="StepLR", load_config=False)
    new_network = update_network_DARTS(cell_list, sep_conv_list)

    return new_network

def run_trials_DARTS(epochs,output_path_train):
    """
    Perform Channel Search for DARTS/DARTS+
    """

    cell_list_average_slope, cell_list_prev_ops , cell_list_factor , sep_conv_list_average_slope, sep_conv_list_prev_ops , sep_conv_list_factor, cell_list_rank, sep_conv_list_rank = [],[],[],[],[],[],[],[]

    trial_dir = os.path.join(GLOBALS.OUTPUT_PATH_STRING, 'Trials')
    print (trial_dir)

    parameter_type=GLOBALS.CONFIG['parameter_type']

    def initialize_dataframes_and_lists():

        #[C0, C1, C2, C3] Sizes

        #Default 7 cells
        if GLOBALS.CONFIG["num_cells"] == 7:
            cell_list_columns = ['STEM', 'cell0', 'cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6']
            sep_conv_list_columns = ['cell0', 'cell1', 'cell3', 'cell5', 'cell6']
            if GLOBALS.CONFIG['network'] == 'DARTS':
                initial_cell_list = GLOBALS.DARTS_cell_list_7
                initial_sep_conv_list = GLOBALS.DARTS_sep_conv_list_7
            else:
                initial_cell_list = GLOBALS.DARTSPlus_cell_list_7
                initial_sep_conv_list = GLOBALS.DARTSPlus_sep_conv_list_7

        elif GLOBALS.CONFIG["num_cells"] == 14:
            cell_list_columns = ['STEM','cell0','cell1','cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell7', \
                                'cell8','cell9','cell10', 'cell11', 'cell12', 'cell13']
            sep_conv_list_columns = ['cell0','cell1','cell2', 'cell3', 'cell5', 'cell6', 'cell7', \
                                    'cell8', 'cell10', 'cell11', 'cell12', 'cell13']
            if GLOBALS.CONFIG['network'] == 'DARTS':
                initial_cell_list = GLOBALS.DARTS_cell_list_14
                initial_sep_conv_list = GLOBALS.DARTS_sep_conv_list_14
            else:
                initial_cell_list = GLOBALS.DARTSPlus_cell_list_14
                initial_sep_conv_list = GLOBALS.DARTSPlus_sep_conv_list_14

        #Config for 20 cells
        elif GLOBALS.CONFIG["num_cells"] == 20:
            cell_list_columns = ['STEM','cell0','cell1','cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell7', \
                                'cell8','cell9','cell10', 'cell11', 'cell12', 'cell13', 'cell14', 'cell15', \
                                'cell16','cell17','cell18', 'cell19']
            sep_conv_list_columns = ['cell0','cell1','cell2', 'cell3', 'cell4', 'cell5',  'cell7', \
                                    'cell8','cell9','cell10', 'cell11', 'cell12', 'cell14', 'cell15', \
                                    'cell16','cell17','cell18', 'cell19']
            if GLOBALS.CONFIG['network'] == 'DARTS':
                initial_cell_list = GLOBALS.DARTS_cell_list_20
                initial_sep_conv_list = GLOBALS.DARTS_sep_conv_list_20
            else:
                initial_cell_list = GLOBALS.DARTSPlus_cell_list_20
                initial_sep_conv_list = GLOBALS.DARTSPlus_sep_conv_list_20


        cell_list_data = pd.DataFrame(columns=cell_list_columns)
        sep_conv_list_data = pd.DataFrame(columns=sep_conv_list_columns)


        #Final ranks used to calculate [C0, C1, C2, C3] Sizes for all 20 cells
        cell_rank_data = pd.DataFrame(columns=cell_list_columns)

        # Final ranks used to calculate intermediate sep conv sizes
        sep_conv_rank_data  = pd.DataFrame(columns=sep_conv_list_columns)


        cell_list_data.loc[0] = initial_cell_list
        sep_conv_list_data.loc[0] = initial_sep_conv_list

        cell_delta_info = pd.DataFrame(columns=['delta_percentage','last_operation','factor_scale'])
        sep_conv_delta_info = pd.DataFrame(columns=['delta_percentage','last_operation','factor_scale'])

        return cell_list_data, sep_conv_list_data, cell_rank_data, sep_conv_rank_data, cell_delta_info, sep_conv_delta_info, \
               initial_cell_list, initial_sep_conv_list


    # Train for 1 trial first to collect data
    cell_list_data, sep_conv_list_data, cell_rank_data, sep_conv_rank_data, cell_delta_info, sep_conv_delta_info, initial_cell_list, initial_sep_conv_list =initialize_dataframes_and_lists()
    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()

    current_cell_list = copy.deepcopy(initial_cell_list)
    current_sep_conv_list = copy.deepcopy(initial_sep_conv_list)

    new_network = update_network_DARTS()

    train_loader, test_loader, device, optimizer, scheduler, model = initialize(args, new_network)
    interrupted_trial = 0  # Determines at which trial we will resume!
    if args.resume_search is False:
        run_epochs(0, model, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_train)
        # print("Memory after first trial:", torch.cuda.memory_allocated(0))
    else:
        interrupted_trial = get_latest_completed_trial(trial_dir)
    print('~~~First run_epochs done.~~~')
    del model
    del train_loader
    del test_loader
    del optimizer
    del scheduler

    free_cuda_memory()

    if (GLOBALS.CONFIG['kernel_adapt']==0):
        GLOBALS.CONFIG['adapt_trials_kernel']=0

    GLOBALS.total_trials=GLOBALS.CONFIG['adapt_trials']+GLOBALS.CONFIG['adapt_trials_kernel']

    for i in range(1,GLOBALS.total_trials):

        # Apply CONet Algorithm and get new channel sizes

        if GLOBALS.CONFIG['network'] == 'DARTS':
            new_cell_list,new_sep_conv_list, cell_list_average_slope, cell_list_prev_ops,cell_list_factor, \
            sep_conv_list_average_slope, sep_conv_list_prev_ops,sep_conv_list_factor, cell_list_rank, sep_conv_list_rank = \
                DARTS_algorithm (current_cell_list ,current_sep_conv_list ,GLOBALS.CONFIG['delta_threshold'], GLOBALS.CONFIG['factor_scale'],\
                                             GLOBALS.CONFIG['min_scale_limit'],GLOBALS.CONFIG['mapping_condition_threshold'], \
                                               GLOBALS.CONFIG['min_conv_size'], GLOBALS.CONFIG['max_conv_size'],
                                                trial_dir,i-1,cell_list_prev_ops ,cell_list_factor ,
                                                 sep_conv_list_prev_ops,sep_conv_list_factor)
        elif GLOBALS.CONFIG['network'] == 'DARTSPlus':
            new_cell_list, new_sep_conv_list, cell_list_average_slope, cell_list_prev_ops, cell_list_factor, \
            sep_conv_list_average_slope, sep_conv_list_prev_ops, sep_conv_list_factor, cell_list_rank, sep_conv_list_rank = \
                DARTSPlus_algorithm(current_cell_list, current_sep_conv_list, GLOBALS.CONFIG['delta_threshold'], GLOBALS.CONFIG['factor_scale'],\
                                GLOBALS.CONFIG['min_scale_limit'], GLOBALS.CONFIG['mapping_condition_threshold'], \
                                GLOBALS.CONFIG['min_conv_size'], GLOBALS.CONFIG['max_conv_size'],
                                trial_dir, i - 1, cell_list_prev_ops, cell_list_factor,
                                sep_conv_list_prev_ops, sep_conv_list_factor)
        #Need to do trial - 1 for trial num because we need the results from the previous trial

        current_cell_list = copy.deepcopy(new_cell_list)
        current_sep_conv_list = copy.deepcopy(new_sep_conv_list)

        cell_list_average_slope_copy = copy.deepcopy(cell_list_average_slope)
        cell_list_prev_ops_copy = copy.deepcopy(cell_list_prev_ops)
        cell_list_factor_copy = copy.deepcopy(cell_list_factor)

        sep_conv_list_average_slope_copy = copy.deepcopy(sep_conv_list_average_slope)
        sep_conv_list_prev_ops_copy = copy.deepcopy(sep_conv_list_prev_ops)
        sep_conv_list_factor_copy = copy.deepcopy(sep_conv_list_factor)

        cell_list_rank_copy = copy.deepcopy(cell_list_rank)
        sep_conv_list_rank_copy = copy.deepcopy(sep_conv_list_rank)


        print ("Cell List:", current_cell_list)
        print ("Conv_sep_list", current_sep_conv_list)
        #print (sep_conv_list_factor)
        #print (sep_conv_list_prev_ops)

        print('~~~Writing to Dataframe~~~')
        if parameter_type=='channel':
            cell_list_data.loc[i] = current_cell_list
            sep_conv_list_data.loc[i] = current_sep_conv_list
            cell_delta_info.loc[i] = [cell_list_average_slope_copy, cell_list_prev_ops_copy, cell_list_factor_copy]
            sep_conv_delta_info.loc[i] = [sep_conv_list_average_slope_copy, sep_conv_list_prev_ops_copy, sep_conv_list_factor_copy]

            cell_rank_data.loc[i] = cell_list_rank_copy
            sep_conv_rank_data.loc[i] = sep_conv_list_rank_copy


        print('~~~Starting Conv parameter_typements~~~')
        print('~~~Initializing the new model~~~')

        # Apply new channel sizes to model
        new_network = update_network_DARTS(current_cell_list, current_sep_conv_list)
        train_loader, test_loader, device, optimizer, scheduler, model = initialize(args, new_network)
        epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])

        if i < interrupted_trial:
            print('~~~Using previous training data~~~')

        else:
            print('~~~Training with new model~~~')
            run_epochs(i, model, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_train)
        del model
        del train_loader
        del test_loader
        del optimizer
        del scheduler

        free_cuda_memory()

    return current_cell_list, current_sep_conv_list, cell_list_data, sep_conv_list_data, cell_delta_info, sep_conv_delta_info, cell_rank_data, sep_conv_rank_data

def create_trial_data_file_DARTS(cell_data, sep_conv_data,  cell_delta_info, sep_conv_delta_info, cell_rank_data, sep_conv_rank_data, output_path_string_trials):
    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'
    #try:
    cell_data.to_excel(output_path_string_trials + slash + 'cell_architectures.xlsx')
    cell_delta_info.to_excel(output_path_string_trials + slash + 'cell_delta_info.xlsx')
    cell_rank_data.to_excel(output_path_string_trials + slash + 'cell_average_rank.xlsx')

    sep_conv_data.to_excel(output_path_string_trials + slash + 'sep_conv_architectures.xlsx')
    sep_conv_delta_info.to_excel(output_path_string_trials + slash + 'sep_conv_delta_info.xlsx')
    sep_conv_rank_data.to_excel(output_path_string_trials + slash + 'sep_conv_average_rank.xlsx')
