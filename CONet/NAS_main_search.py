from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path
import os
import platform


from train_ResNet import *
from train_DARTS import *
import  global_vars as GLOBALS

if __name__ == '__main__':

    
    GLOBALS.FIRST_INIT = True
    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()
    output_path = build_paths(args)

    output_path = output_path / f"conv_{GLOBALS.CONFIG['init_conv_setting']}_deltaThresh={GLOBALS.CONFIG['delta_threshold']}_kernelthresh={GLOBALS.CONFIG['delta_threshold_kernel']}_epochpert={GLOBALS.CONFIG['epochs_per_trial']}_adaptnum={GLOBALS.CONFIG['adapt_trials']}"
    GLOBALS.OUTPUT_PATH_STRING = str(output_path)

    if not os.path.exists(GLOBALS.OUTPUT_PATH_STRING):
        os.mkdir(GLOBALS.OUTPUT_PATH_STRING)

    print('~~~Initialization Complete. Beginning first training~~~')

    epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])
    full_train_epochs = range(0, GLOBALS.CONFIG['max_epoch'])
    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'
    output_path_string_trials = GLOBALS.OUTPUT_PATH_STRING +slash+ 'Trials'
    output_path_string_modelweights = GLOBALS.OUTPUT_PATH_STRING +slash+ 'model_weights'
    output_path_string_graph_files = GLOBALS.OUTPUT_PATH_STRING +slash+ 'graph_files'
    output_path_string_full_train = GLOBALS.OUTPUT_PATH_STRING +slash+ 'full_train'
    output_path_train = output_path / f"Trials"
    output_path_fulltrain = output_path / f"full_train"

    if not os.path.exists(output_path_string_trials):
        os.mkdir(output_path_string_trials)

    if not os.path.exists(output_path_string_modelweights):
        os.mkdir(output_path_string_modelweights)

    if not os.path.exists(output_path_string_graph_files):
        os.mkdir(output_path_string_graph_files)

    if not os.path.exists(output_path_string_full_train):
        os.mkdir(output_path_string_full_train)

    if GLOBALS.CONFIG['network'] == 'DASNet34':
        if GLOBALS.CONFIG['full_train_only']==False:
            print('Starting Trials')
            kernel_data,conv_data,rank_final_data,rank_stable_data,output_sizes,kernel_sizes,delta_info,delta_info_kernel=run_trials(epochs,output_path_train)
            create_trial_data_file(kernel_data,conv_data,delta_info_kernel,delta_info,rank_final_data,rank_stable_data,output_path_string_trials,output_path_string_graph_files,output_path_string_modelweights)
            print('Done Trials.')
        else:
            k = 3
            output_sizes=[GLOBALS.super1_idx, GLOBALS.super2_idx, GLOBALS.super3_idx, GLOBALS.super4_idx]
            kernel_sizes = [GLOBALS.super1_kernel_idx,GLOBALS.super2_kernel_idx,GLOBALS.super3_kernel_idx,GLOBALS.super4_kernel_idx]
        #output_sizes=[[64,64,64,64,64],[64,64,64,64],[64,64,64,64],[64,64,64,64],[64,64,64,64]]

        model = run_fresh_full_train(output_sizes,kernel_sizes,full_train_epochs,output_path_fulltrain)

        create_train_output_file(model,output_path_string_full_train+slash+f"StepLR_fresh_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx",
                                     output_path_string_full_train)

        for i in range (1, GLOBALS.CONFIG['train_num']):

            # Why is there 2 variables???
            output_path_fulltrain = output_path / "full_train_{}".format(i)
            output_path_string_full_train = GLOBALS.OUTPUT_PATH_STRING + slash + "full_train_{}".format(i)
            if not os.path.exists(output_path_string_full_train):
                os.mkdir(output_path_string_full_train)

            model = run_fresh_full_train(output_sizes, kernel_sizes, full_train_epochs, output_path_fulltrain)

            create_train_output_file(model,
                                     output_path_string_full_train + slash + f"StepLR_fresh_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx",
                                     output_path_string_full_train)

    elif GLOBALS.CONFIG['network'] == 'DARTS' or GLOBALS.CONFIG['network'] == 'DARTSPlus':
        cell_list = None
        sep_conv_list = None

        if GLOBALS.CONFIG['full_train_only'] == False:
            print('Starting Trials')
            cell_list, sep_conv_list, cell_data, sep_conv_data, cell_delta_info, sep_conv_delta_info, cell_rank_data, sep_conv_rank_data = run_trials_DARTS(epochs, output_path_train)
            create_trial_data_file_DARTS(cell_data, sep_conv_data, cell_delta_info, sep_conv_delta_info, cell_rank_data,
                                   sep_conv_rank_data, output_path_string_trials)
            print('Done Trials.')

        model = run_fresh_full_train_DARTS(full_train_epochs, output_path_fulltrain, cell_list=cell_list, sep_conv_list=sep_conv_list)

        # paths are hard coded
        create_full_data_file_DARTS(model,
                              output_path_string_full_train + slash + f"{GLOBALS.CONFIG['lr_scheduler']}_fresh_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx",
                              output_path_string_full_train)

        for i in range (1, GLOBALS.CONFIG['train_num']):
            output_path_fulltrain = output_path / "full_train_{}".format(i)
            output_path_string_full_train = GLOBALS.OUTPUT_PATH_STRING + slash + "full_train_{}".format(i)
            if not os.path.exists(output_path_string_full_train):
                os.mkdir(output_path_string_full_train)

            model = run_fresh_full_train_DARTS(full_train_epochs, output_path_fulltrain, cell_list=cell_list,
                                               sep_conv_list=sep_conv_list)
            # paths are hard coded
            create_full_data_file_DARTS(model,
                                        output_path_string_full_train + slash + f"{GLOBALS.CONFIG['lr_scheduler']}_fresh_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx",
                                        output_path_string_full_train)


    print('Done Full Train')
