import copy
import pandas as pd
import numpy as np
import os
#import global_vars as GLOBALS

# Find the first conv layer of each cells. Returns list of first conv layer where idx is cell idx
# NOTE: first cell in the list is the STEM cell!
def find_cell_first_conv_idx(cell_list):
    normal_cell_layers = 26
    reduction_cell_layers = 8
    red_normal_cell_layers = 27
    cur_layer_count = 0
    reduction_cell_idx = [(len(cell_list)-1)//3+1, (len(cell_list)-1)*2//3+1] # 3,5 for 7 cells+1 stem
    #print("reduction cell idx: ",reduction_cell_idx)
    prev_cell_is_reduction = False
    cells_first_layer_idx = []
    for cur_cell_idx,cur_cell in enumerate(cell_list):
        if cur_cell_idx == 0:
            cur_layer_count = 0
        elif cur_cell_idx == 1:
            cur_layer_count = 1
        elif cur_cell_idx-1 in reduction_cell_idx:
            prev_cell_is_reduction = True
            cur_layer_count += reduction_cell_layers
        elif prev_cell_is_reduction:
            cur_layer_count += red_normal_cell_layers
            prev_cell_is_reduction = False
        else:
            cur_layer_count += normal_cell_layers

        cells_first_layer_idx.append(cur_layer_count)
    return copy.deepcopy(cells_first_layer_idx)

# cell types: stem, norm_cell, red_cell, norm_red_cell
# returns a list of string of type for corresponding cell idx
def get_cells_type(cell_list):
    cells_type = []
    reduction_cell_idx = [(len(cell_list) - 1) // 3 + 1, (len(cell_list) - 1) * 2 // 3 + 1]
    cur_cell_type = 'stem'
    for cur_cell_idx, cur_cell in enumerate(cell_list):
        if cur_cell_idx == 0:
            cur_cell_type = 'stem'
        elif cur_cell_idx in reduction_cell_idx:
            cur_cell_type = 'red_cell'
        elif cur_cell_idx-1 in reduction_cell_idx:
            cur_cell_type = 'norm_red_cell'
        else:
            cur_cell_type = 'norm_cell'
        cells_type.append(cur_cell_type)
    return copy.deepcopy(cells_type)

# Read the trial excel file to get in rank and out rank for each layer for all epochs
# returns in and out rank of each layer indexed as [layer#,epoch#]
def get_layer_conv_ranks(trial_dir,cur_trial):
    cur_trial = 'AdaS_adapt_trial=%s'%str(cur_trial)
    file_list = os.listdir(trial_dir)
    in_rank = list()
    out_rank = list()
    for file in file_list:
        if file.startswith(cur_trial) and file.endswith('.xlsx'):
            file_path = os.path.join(trial_dir,file)
            df_trial = pd.read_excel(file_path, index_col=0)
            # find all in_rank at each layer for all epochs
            cols = [col for col in df_trial if col.startswith('in_rank')]
            for col in cols:
                temp = df_trial.loc[:, col].tolist()
                in_rank.append(copy.deepcopy(np.array(temp)))

            # find all out_rank at each layer for all epochs
            cols = [col for col in df_trial if col.startswith('out_rank')]
            for col in cols:
                temp = df_trial.loc[:, col].tolist()
                out_rank.append(copy.deepcopy(np.array(temp)))

            break
    in_rank = np.array(in_rank).transpose((1,0)) # [epoch,layer] -> [layer,epoch]
    out_rank = np.array(out_rank).transpose((1,0))
    return copy.deepcopy(in_rank), copy.deepcopy(out_rank)

# return last epoch's mapping condition for each layer
def get_layer_mapping(trial_dir,cur_trial):
    cur_trial = 'AdaS_adapt_trial=%s'%str(cur_trial)
    file_list = os.listdir(trial_dir)
    in_cond = list()
    out_cond = list()
    for file in file_list:
        if file.startswith(cur_trial) and file.endswith('.xlsx'):
            file_path = os.path.join(trial_dir,file)
            df_trial = pd.read_excel(file_path, index_col=0)
            # find all in_rank at each layer for all epochs
            cols = [col for col in df_trial if col.startswith('in_condition')]
            for col in cols:
                temp = df_trial.loc[:, col].tolist()
                in_cond.append(copy.deepcopy(np.array(temp)))

            # find all out_rank at each layer for all epochs
            cols = [col for col in df_trial if col.startswith('out_condition')]
            for col in cols:
                temp = df_trial.loc[:, col].tolist()
                out_cond.append(copy.deepcopy(np.array(temp)))

            break
    # in_cond = np.array(in_cond).transpose((1,0)) # [epoch,layer] -> [layer,epoch]
    # out_cond = np.array(out_cond).transpose((1,0))
    return copy.deepcopy(in_cond[-1]), copy.deepcopy(out_cond[-1]) # only return last epoch

# for a cell list: [cell0,cell1,cell2,...]
# each cell has list of adjustable channel size: [C0,C1,C2,C3]
# output is rank avg of each adjustable channel size by dependency mapping:
# [[avg_rank],[avg_rank,avg_rank,avg_rank,avg_rank],...] in the format of cell_list
def get_cell_list_ranks_by_dependency(in_rank,out_rank,cell_first_layer_idx,cells_type):
    # cell types: stem, norm_cell, red_cell, norm_red_cell
    # encoding is '{in/out}_rank-{conv_layer_offset}'. NOTE: conv_layer_idx = cell_first_conv_idx + conv_layer_offset
    chan_size_dependency_mapping = {
        'stem': [['out_rank-0']],
        'norm_cell': [['out_rank-0','out_rank-5','in_rank-7','out_rank-9','in_rank-15','in_rank-19'], # C0
                      ['out_rank-1','in_rank-3','in_rank-11'], # C1
                      ['out_rank-13','out_rank-17','in_rank-23'], # C2
                      ['out_rank-21','out_rank-25'] # C3
                      ],
        'red_cell': [['out_rank-0','out_rank-2','out_rank-3','in_rank-5','out_rank-5','in_rank-7','out_rank-7'],
                     ['out_rank-1','in_rank-2','in_rank-3']],
        'norm_red_cell': [['out_rank-0','out_rank-1','out_rank-6','in_rank-8','out_rank-10','in_rank-16','in_rank-20'],
                          ['out_rank-2','in_rank-4','in_rank-12'],
                          ['out_rank-14','out_rank-18','in_rank-24'],
                          ['out_rank-22','out_rank-26']
                          ]
    }
    total_epoch = in_rank.shape[1]
    channel_size_avg_ranks = list()
    # explore each cells we have
    for cur_cell_idx, first_conv_idx in enumerate(cell_first_layer_idx):
        cur_cell_avg_ranks = list()
        cur_cell_type = cells_type[cur_cell_idx]
        dependency_mapping = chan_size_dependency_mapping[cur_cell_type]
        # for each cell, check the dependency for each conv size
        for j,chan_dependencies in enumerate(dependency_mapping):
            avg_rank = np.zeros(total_epoch) # initialize avg rank for current channel size
            # for each dependency mapping of a conv size, calculate the avg rank
            # if reduction cell, need to handle differently
            if cur_cell_type == 'red_cell':
                i = 0
                for idx in range(len(chan_dependencies)-1):
                    # need to consider the concat of conv2 and conv3 as a single conv
                    if i == 1:
                        rank_type = chan_dependencies[i].split('-')[0]
                        conv_layer_offset_1 = int(chan_dependencies[i].split('-')[1])
                        conv_layer_idx_1 = first_conv_idx + conv_layer_offset_1
                        conv_layer_offset_2 = int(chan_dependencies[i+1].split('-')[1])
                        conv_layer_idx_2 = first_conv_idx + conv_layer_offset_2
                        if rank_type == 'in_rank':
                            rank_list = in_rank
                        else:
                            rank_list = out_rank
                        temp = (rank_list[conv_layer_idx_1] + rank_list[conv_layer_idx_2]) / 2
                        avg_rank += temp
                        i += 2
                        continue
                    rank_type = chan_dependencies[i].split('-')[0]
                    conv_layer_offset = int(chan_dependencies[i].split('-')[1])
                    conv_layer_idx = first_conv_idx+conv_layer_offset
                    if rank_type == 'in_rank':
                        rank_list = in_rank
                    else:
                        rank_list = out_rank
                    avg_rank += rank_list[conv_layer_idx]
                    i += 1
                i -= 2 # need to consider 2 concat conv layer as 1, minus 2 since we have plus 1 in averaging
                # print(i)
                # print(avg_rank)
            else:
                # if it's reduction cell, need to first calculate the avg of the first 2 dependency due to concatenation
                if cur_cell_type == 'norm_red_cell' and j==0:
                    red_dependencies = chan_dependencies[:2] # will only avg the first 2
                    for encoding in red_dependencies:
                        rank_type = encoding.split('-')[0]
                        conv_layer_offset = int(encoding.split('-')[1])
                        conv_layer_idx = first_conv_idx + conv_layer_offset
                        if rank_type == 'in_rank':
                            rank_list = in_rank
                        else:
                            rank_list = out_rank
                        avg_rank += rank_list[conv_layer_idx]
                        # print(encoding, conv_layer_idx)
                        # print('rank list: ', rank_list[conv_layer_idx])
                        # print(avg_rank)
                    avg_rank /= 2
                    # print(avg_rank)
                    chan_dependencies = chan_dependencies[2:] # will continue after the first 2
                # find avg rank of the dependent layers
                for i,encoding in enumerate(chan_dependencies):
                    rank_type = encoding.split('-')[0]
                    conv_layer_offset = int(encoding.split('-')[1])
                    conv_layer_idx = first_conv_idx+conv_layer_offset
                    if rank_type == 'in_rank':
                        rank_list = in_rank
                    else:
                        rank_list = out_rank
                    avg_rank += rank_list[conv_layer_idx]
                    # if cur_cell_type == 'norm_red_cell' and j == 0:
                    #     print(i,encoding,conv_layer_idx)
                    #     print('rank list: ',rank_list[conv_layer_idx])
                    #     print(avg_rank)
                if cur_cell_type == 'norm_red_cell' and j==0:
                    i += 1 # consider the first pair as 1 value
            avg_rank /= (i+1) # calculate avg of dependent ranks
            cur_cell_avg_ranks.append(copy.deepcopy(avg_rank))
        channel_size_avg_ranks.append(copy.deepcopy(np.array(cur_cell_avg_ranks)))
    return copy.deepcopy(channel_size_avg_ranks)

# for a sep_conv_list = [norm_cell1,norm_cell2,....]
# each normal/normal reduction cell has adjustable sep conv size [sep1,sep2,sep3,sep4,sep5]
# output is rank avg of each adjustable channel size by dependency mapping:
# [[avg_rank,avg_rank,avg_rank,avg_rank,avg_rank],...] in the format of sep_conv_list
def get_sep_conv_list_ranks_by_dependency(in_rank,out_rank,cell_first_layer_idx,cells_type):
    # cell types: stem, norm_cell, red_cell, norm_red_cell
    # encoding is '{in/out}_rank-{conv_layer_offset}'. NOTE: conv_layer_idx = cell_first_conv_idx + conv_layer_offset
    # sep conv channel size dependency mapping:
    norm_cell_sep_conv_mapping = [['out_rank-3', 'in_rank-5'],
                                  ['out_rank-7', 'in_rank-9'],
                                  ['out_rank-11', 'in_rank-13'],
                                  ['out_rank-15', 'in_rank-17'],
                                  ['out_rank-19', 'in_rank-21'],
                                  ['out_rank-23', 'in_rank-25']
                                  ]
    norm_red_cell_sep_conv_mapping = [['out_rank-4', 'in_rank-6'],
                                      ['out_rank-8', 'in_rank-10'],
                                      ['out_rank-12', 'in_rank-14'],
                                      ['out_rank-16', 'in_rank-18'],
                                      ['out_rank-20', 'in_rank-22'],
                                      ['out_rank-24', 'in_rank-26']
                                      ]

    total_epoch = in_rank.shape[1]
    sep_conv_avg_rank = list()
    # explore each cells we have
    for cur_cell_idx, first_conv_idx in enumerate(cell_first_layer_idx):
        cur_sep_conv_avg_ranks = list()
        cur_cell_type = cells_type[cur_cell_idx]
        if cur_cell_type == 'norm_cell':
            mapping = norm_cell_sep_conv_mapping
        elif cur_cell_type == 'norm_red_cell':
            mapping = norm_red_cell_sep_conv_mapping
        # for each sep_conv group, find avg rank
        if cur_cell_type == 'norm_cell' or cur_cell_type == 'norm_red_cell':
            for sep_conv_dependencies in mapping:
                avg_rank = np.zeros(total_epoch)  # initialize avg rank for current channel size
                for i,encoding in enumerate(sep_conv_dependencies):
                    rank_type = encoding.split('-')[0]
                    conv_layer_offset = int(encoding.split('-')[1])
                    conv_layer_idx = first_conv_idx + conv_layer_offset
                    if rank_type == 'in_rank':
                        rank_list = in_rank
                    else:
                        rank_list = out_rank
                    avg_rank += rank_list[conv_layer_idx]
                avg_rank /= (i+1)
                cur_sep_conv_avg_ranks.append(copy.deepcopy(avg_rank))
            sep_conv_avg_rank.append(copy.deepcopy(np.array(cur_sep_conv_avg_ranks)))
    # print(sep_conv_avg_rank[2])
    return copy.deepcopy(sep_conv_avg_rank)

# returns cell avg rank slope in format of the input cell_list or sep_conv_list format
def find_rank_avg_slope(cell_avg_ranks):
    cell_rank_slopes = list()
    for i,cell in enumerate(cell_avg_ranks):
        cur_cell_rank_slopes = list()
        for avg_rank in cell:
            max_rank_epoch = np.argmax(avg_rank) # find the index (epoch#) where the max avg rank is
            rank_delta = avg_rank[max_rank_epoch] - avg_rank[0]
            epoch_delta = max_rank_epoch - 0
            rank_slope = rank_delta/ (epoch_delta + 0.0001)

            # if i == 1:
            #     print(max_rank_epoch)
            #     print(avg_rank[max_rank_epoch])
            #     print(rank_slope)
            cur_cell_rank_slopes.append(rank_slope)
        cell_rank_slopes.append(copy.deepcopy(np.array(cur_cell_rank_slopes)))
    # print(cell_avg_ranks[1])
    # print(cell_rank_slopes[1])
    return copy.deepcopy(cell_rank_slopes)

def get_cell_list_mapping_condition(in_cond,out_cond,cell_first_layer_idx,cells_type):
    # cell types: stem, norm_cell, red_cell, norm_red_cell
    # encoding is '{in/out}_rank-{conv_layer_offset}'. NOTE: conv_layer_idx = cell_first_conv_idx + conv_layer_offset
    chan_size_dependency_mapping = {
        'stem': [['out_cond-0']],
        'norm_cell': [['out_cond-0','out_cond-5','in_cond-7','out_cond-9','in_cond-15','in_cond-19'], # C0
                      ['out_cond-1','in_cond-3','in_cond-11'], # C1
                      ['out_cond-13','out_cond-17','in_cond-23'], # C2
                      ['out_cond-21','out_cond-25'] # C3
                      ],
        'red_cell': [['out_cond-0','out_cond-2','out_cond-3','in_cond-5','out_cond-5','in_cond-7','out_cond-7'],
                     ['out_cond-1','in_cond-2','in_cond-3']],
        'norm_red_cell': [['out_cond-0','out_cond-1','out_cond-6','in_cond-8','out_cond-10','in_cond-16','in_cond-20'],
                          ['out_cond-2','in_cond-4','in_cond-12'],
                          ['out_cond-14','out_cond-18','in_cond-24'],
                          ['out_cond-22','out_cond-26']
                          ]
    }
    cell_cond = list()
    # explore each cells we have
    for cur_cell_idx, first_conv_idx in enumerate(cell_first_layer_idx):
        cur_cell_avg_conds = list()
        cur_cell_type = cells_type[cur_cell_idx]
        dependency_mapping = chan_size_dependency_mapping[cur_cell_type]
        # for each cell, check the dependency for each conv size
        for j,chan_dependencies in enumerate(dependency_mapping):
            avg_cond = 0 # initialize avg rank for current channel size
            # for each dependency mapping of a conv size, calculate the avg rank
            # if reduction cell, need to handle differently
            if cur_cell_type == 'red_cell':
                i = 0
                for idx in range(len(chan_dependencies)-1):
                    # need to consider the concat of conv2 and conv3 as a single conv
                    if i == 1:
                        cond_type = chan_dependencies[i].split('-')[0]
                        conv_layer_offset_1 = int(chan_dependencies[i].split('-')[1])
                        conv_layer_idx_1 = first_conv_idx + conv_layer_offset_1
                        conv_layer_offset_2 = int(chan_dependencies[i+1].split('-')[1])
                        conv_layer_idx_2 = first_conv_idx + conv_layer_offset_2
                        if cond_type == 'in_cond':
                            cond_list = in_cond
                        else:
                            cond_list = out_cond
                        temp = (cond_list[conv_layer_idx_1] + cond_list[conv_layer_idx_2]) / 2
                        avg_cond += temp
                        i += 2
                        continue
                    cond_type = chan_dependencies[i].split('-')[0]
                    conv_layer_offset = int(chan_dependencies[i].split('-')[1])
                    conv_layer_idx = first_conv_idx+conv_layer_offset
                    if cond_type == 'in_cond':
                        cond_list = in_cond
                    else:
                        cond_list = out_cond
                    avg_cond += cond_list[conv_layer_idx]
                    i += 1
                i -= 2 # need to consider 2 concat conv layer as 1, minus 2 since we have plus 1 in averaging
                # print(i)
                # print(avg_cond)
            else:
                # if it's reduction cell, need to first calculate the avg of the first 2 dependency due to concatenation
                if cur_cell_type == 'norm_red_cell' and j==0:
                    red_dependencies = chan_dependencies[:2] # will only avg the first 2
                    for encoding in red_dependencies:
                        cond_type = encoding.split('-')[0]
                        conv_layer_offset = int(encoding.split('-')[1])
                        conv_layer_idx = first_conv_idx + conv_layer_offset
                        if cond_type == 'in_cond':
                            cond_list = in_cond
                        else:
                            cond_list = out_cond
                        avg_cond += cond_list[conv_layer_idx]
                        # print(encoding, conv_layer_idx)
                        # print('rank list: ', cond_list[conv_layer_idx])
                        # print(avg_cond)
                    avg_cond /= 2
                    # print(avg_cond)
                    chan_dependencies = chan_dependencies[2:] # will continue after the first 2
                # find avg rank of the dependent layers
                for i,encoding in enumerate(chan_dependencies):
                    cond_type = encoding.split('-')[0]
                    conv_layer_offset = int(encoding.split('-')[1])
                    conv_layer_idx = first_conv_idx+conv_layer_offset
                    if cond_type == 'in_cond':
                        cond_list = in_cond
                    else:
                        cond_list = out_cond
                    avg_cond += cond_list[conv_layer_idx]
                    # if cur_cell_type == 'norm_red_cell' and j == 0:
                    #     print(i,encoding,conv_layer_idx)
                    #     print('rank list: ',cond_list[conv_layer_idx])
                    #     print(avg_cond)
                if cur_cell_type == 'norm_red_cell' and j==0:
                    i += 1 # consider the first pair as 1 value
            avg_cond /= (i+1) # calculate avg of dependent ranks
            cur_cell_avg_conds.append(avg_cond)
        cell_cond.append(copy.deepcopy(np.array(cur_cell_avg_conds)))
    return copy.deepcopy(cell_cond)

def get_sep_conv_list_mapping_condition(in_cond,out_cond,cell_first_layer_idx,cells_type):
    # cell types: stem, norm_cell, red_cell, norm_red_cell
    # encoding is '{in/out}_cond-{conv_layer_offset}'. NOTE: conv_layer_idx = cell_first_conv_idx + conv_layer_offset
    # sep conv channel size dependency mapping:
    norm_cell_sep_conv_mapping = [['out_cond-3', 'in_cond-5'],
                                  ['out_cond-7', 'in_cond-9'],
                                  ['out_cond-11', 'in_cond-13'],
                                  ['out_cond-15', 'in_cond-17'],
                                  ['out_cond-19', 'in_cond-21'],
                                  ['out_cond-23', 'in_cond-25']
                                  ]
    norm_red_cell_sep_conv_mapping = [['out_cond-4', 'in_cond-6'],
                                      ['out_cond-8', 'in_cond-10'],
                                      ['out_cond-12', 'in_cond-14'],
                                      ['out_cond-16', 'in_cond-18'],
                                      ['out_cond-20', 'in_cond-22'],
                                      ['out_cond-24', 'in_cond-26']
                                      ]

    sep_conv_avg_cond = list()
    # explore each cells we have
    for cur_cell_idx, first_conv_idx in enumerate(cell_first_layer_idx):
        cur_sep_conv_avg_conds = list()
        cur_cell_type = cells_type[cur_cell_idx]
        if cur_cell_type == 'norm_cell':
            mapping = norm_cell_sep_conv_mapping
        elif cur_cell_type == 'norm_red_cell':
            mapping = norm_red_cell_sep_conv_mapping
        # for each sep_conv group, find avg cond
        if cur_cell_type == 'norm_cell' or cur_cell_type == 'norm_red_cell':
            for sep_conv_dependencies in mapping:
                avg_cond = 0  # initialize avg cond for current channel size
                for i,encoding in enumerate(sep_conv_dependencies):
                    cond_type = encoding.split('-')[0]
                    conv_layer_offset = int(encoding.split('-')[1])
                    conv_layer_idx = first_conv_idx + conv_layer_offset
                    if cond_type == 'in_cond':
                        cond_list = in_cond
                    else:
                        cond_list = out_cond
                    avg_cond += cond_list[conv_layer_idx]
                avg_cond /= (i+1)
                cur_sep_conv_avg_conds.append(avg_cond)
            sep_conv_avg_cond.append(copy.deepcopy(np.array(cur_sep_conv_avg_conds)))
    # print(sep_conv_avg_cond[2])
    return copy.deepcopy(sep_conv_avg_cond)

# round to nearest even number for chan size
def round_even(number):
    return int(round(number / 2) * 2)

# provide cell_list or sep_conv_list to generate a prev operation list and factor list
# prev_op initalized to be 0 (do nothing/stop), factor_list inialized to be 0.2 (maximum)
def initialize_algorithm(conv_list):
    prev_op_list = list()
    factor_list = list()
    for cell in conv_list:
        cell_adjustable_convs = len(cell)
        cell_op = np.zeros(cell_adjustable_convs,dtype=int)
        cell_factor = np.full(cell_adjustable_convs,0.2)
        prev_op_list.append(copy.deepcopy(cell_op))
        factor_list.append((copy.deepcopy(cell_factor)))
    return copy.deepcopy(prev_op_list),copy.deepcopy(factor_list)

# returns new cell_list, sep_conv_list, and previous operation and scaling for both lists
# chan sizes are rounded to nearest even int
def channel_size_adjust_algorithm(cell_list,sep_conv_list,delta_threshold,min_scale_limit,map_cond_threshold,
                                  min_conv_size,max_conv_size,trial_dir,cur_trial,cell_list_prev_ops=None,
                                  cell_list_factor=None,sep_conv_list_prev_ops=None,sep_conv_list_factor=None):
    first_trial = False
    if len(cell_list_prev_ops) == 0 or cell_list_prev_ops == None:
        first_trial = True
        cell_list_prev_ops,cell_list_factor = initialize_algorithm(cell_list)
        sep_conv_list_prev_ops,sep_conv_list_factor = initialize_algorithm(sep_conv_list)
    # assert len(sep_conv_list_prev_ops) != 0 and sep_conv_list_prev_ops != None

    # get the avg rank slope of each adjustable channel sizes
    cell_first_layer_idx = find_cell_first_conv_idx(cell_list)
    cells_type = get_cells_type(cell_list)
    layers_in_rank, layers_out_rank = get_layer_conv_ranks(trial_dir, cur_trial)
    cell_list_avg_ranks = get_cell_list_ranks_by_dependency(layers_in_rank, layers_out_rank, cell_first_layer_idx,
                                                            cells_type)
    sep_conv_list_avg_ranks = get_sep_conv_list_ranks_by_dependency(layers_in_rank, layers_out_rank,
                                                                    cell_first_layer_idx, cells_type)
    cell_list_rank_slopes = find_rank_avg_slope(cell_list_avg_ranks)
    sep_conv_list_rank_slopes = find_rank_avg_slope(sep_conv_list_avg_ranks)

    # get the avg mapping condition for each adjustable channel sizes
    in_cond, out_cond = get_layer_mapping(trial_dir, cur_trial)
    cell_list_cond = get_cell_list_mapping_condition(in_cond, out_cond, cell_first_layer_idx, cells_type)
    sep_conv_list_cond = get_sep_conv_list_mapping_condition(in_cond, out_cond, cell_first_layer_idx, cells_type)

    # algorithm: operation = expand:1/shrink:-1/stop:0
    for cell_idx,cell in enumerate(cell_list_rank_slopes):
        for chan_idx,avg_rank_slope in enumerate(cell):
            map_cond = cell_list_cond[cell_idx][chan_idx]
            chan_size = cell_list[cell_idx][chan_idx]
            prev_op = cell_list_prev_ops[cell_idx][chan_idx]
            chan_scale = cell_list_factor[cell_idx][chan_idx]
            if not first_trial and prev_op == 0:
                continue # if threshold already satisfied, skip this conv
            cur_op = 1 # initialize operation to expand first.
            if ((avg_rank_slope < delta_threshold or map_cond>=map_cond_threshold) and chan_size > min_conv_size) or (chan_size > max_conv_size):
                cur_op = -1
            if prev_op != cur_op and not first_trial:
                if chan_scale < min_scale_limit:
                    cur_op = 0
                cell_list_factor[cell_idx][chan_idx] = chan_scale/2
            cell_list_prev_ops[cell_idx][chan_idx] = cur_op
            new_cell_size = round_even(chan_size * (1 + cell_list_factor[cell_idx][chan_idx] * cur_op))
            if new_cell_size > max_conv_size:
                new_cell_size = max_conv_size
            elif new_cell_size < min_conv_size:
                new_cell_size = min_conv_size
            cell_list[cell_idx][chan_idx] = new_cell_size

    # do same thing for sep_conv_list
    for cell_idx,cell in enumerate(sep_conv_list_rank_slopes):
        for chan_idx,avg_rank_slope in enumerate(cell):
            map_cond = sep_conv_list_cond[cell_idx][chan_idx]
            chan_size = sep_conv_list[cell_idx][chan_idx]
            prev_op = sep_conv_list_prev_ops[cell_idx][chan_idx]
            chan_scale = sep_conv_list_factor[cell_idx][chan_idx]
            if not first_trial and prev_op == 0:
                continue # if already really close to threshold, skip this conv
            cur_op = 1 # initialize operation to expand first.
            if ((avg_rank_slope < delta_threshold or map_cond>=map_cond_threshold) and chan_size > min_conv_size) or (chan_size > max_conv_size):
                cur_op = -1
            if prev_op != cur_op and not first_trial:
                if chan_scale < min_scale_limit:
                    cur_op = 0
                sep_conv_list_factor[cell_idx][chan_idx] = chan_scale/2
            sep_conv_list_prev_ops[cell_idx][chan_idx] = cur_op
            new_sep_size = round_even(chan_size * (1 + sep_conv_list_factor[cell_idx][chan_idx] * cur_op))
            if new_sep_size > max_conv_size:
                new_sep_size = max_conv_size
            elif new_sep_size < min_conv_size:
                new_sep_size = min_conv_size
            sep_conv_list[cell_idx][chan_idx] = new_sep_size

    return cell_list,sep_conv_list,cell_list_rank_slopes, cell_list_prev_ops,cell_list_factor, sep_conv_list_rank_slopes, sep_conv_list_prev_ops,sep_conv_list_factor, cell_list_avg_ranks, sep_conv_list_avg_ranks







''' main is only for testing purpose, can run on its own '''
if __name__ == '__main__':
    cell_list = [[20], [4, 6, 8, 10], [12, 14, 16, 18], [20, 22], [24, 26, 28, 30], [32, 34], [36, 38, 40, 42],
                 [44, 46, 48, 50]]
    sep_conv_list = [[2,4,6,8,10,12],[14,16,18,20,22,24],[26,28,30,32,34,36],[38,40,42,44,46,48],[50,52,54,56,58,60]]
    print("orig cell list:")
    print(cell_list)
    print("orig sep list")
    print(sep_conv_list)

    '''MUST NAME THE TRIALS EXCEL FILES STARTING WITH AdaS_adapt_trial=#_, # is the trial count'''
    path = os.getcwd()
    print(path)
    #trial_dir = ".\\adas_search\\conv_3,3_deltaThresh=0.025_kernelthresh=0.75_epochpert=2_adaptnum=3\\Trials\\"  # directory of the Trials folder
    trial_dir = ".\\"
    cur_trial = 0 # current trial count

    '''ONLY PUT IT HERE FOR DEBUGGING PURPOSES. MY FUNCTION WILL FIND THESE INSIDE'''
    # get the avg rank slope of each adjustable channel sizes
    cell_first_layer_idx = find_cell_first_conv_idx(cell_list)
    cells_type = get_cells_type(cell_list)
    print("first conv idx of each cell: ",cell_first_layer_idx)
    print("cell types: ",cells_type)
    layers_in_rank, layers_out_rank = get_layer_conv_ranks(trial_dir, cur_trial)
    # print(layers_in_rank)
    # print(layers_out_rank)

    cell_list_avg_ranks = get_cell_list_ranks_by_dependency(layers_in_rank, layers_out_rank, cell_first_layer_idx,
                                                            cells_type)
    sep_conv_list_avg_ranks = get_sep_conv_list_ranks_by_dependency(layers_in_rank, layers_out_rank,
                                                                    cell_first_layer_idx, cells_type)
    in_cond, out_cond = get_layer_mapping(trial_dir, cur_trial)
    cell_cond = get_cell_list_mapping_condition(in_cond, out_cond, cell_first_layer_idx, cells_type)
    print("cell list mapping condition")
    print(cell_cond)
    sep_cond = get_sep_conv_list_mapping_condition(in_cond, out_cond, cell_first_layer_idx, cells_type)
    print("sep conv list mapping condition")
    print(sep_cond)

    cell_list_rank_slopes = find_rank_avg_slope(cell_list_avg_ranks)
    sep_conv_list_rank_slopes = find_rank_avg_slope(sep_conv_list_avg_ranks)
    #print(cell_list_avg_ranks)
    print("cell list avg rank slopes:")
    print(cell_list_rank_slopes)
    print("sep list avg rank slopes:")
    print(sep_conv_list_rank_slopes)


    # thresholds for testing
    delta_threshold = 0.025
    min_scale_limit = 0.001
    map_cond_threshold = 10
    max_conv_size = 256
    min_conv_size = 32
    cell_list_prev_ops = []
    cell_list_factor = []
    sep_conv_list_prev_ops = []
    sep_conv_list_factor = []
    cell_list, sep_conv_list, cell_list_rank_slopes, cell_list_prev_ops, cell_list_factor, sep_conv_list_rank_slopes, sep_conv_list_prev_ops, sep_conv_list_factor, cell_list_avg_ranks, sep_conv_list_avg_ranks = channel_size_adjust_algorithm(cell_list, sep_conv_list, delta_threshold, min_scale_limit, map_cond_threshold, min_conv_size, max_conv_size,
                                                                                                                                              trial_dir, cur_trial, cell_list_prev_ops, cell_list_factor,
                                                                                                                                              sep_conv_list_prev_ops, sep_conv_list_factor)

    print("delta threshold is: ",delta_threshold)
    print("min conv size is: ",min_conv_size)
    print("max conv size is: ",max_conv_size)
    print("mapping condition threshold is: ",map_cond_threshold)
    print("new cell_list:")
    print(cell_list)
    print("new sep_list:")
    print(sep_conv_list)
    print("cell list prev op:")
    print(cell_list_prev_ops)
    print("sep list prev op:")
    print(sep_conv_list_prev_ops)
    print("cell list factor:")
    print(cell_list_factor)
    print("sep list factor:")
    print(sep_conv_list_factor)