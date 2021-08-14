import copy
import pandas as pd
import numpy as np
import os

# find the first layer of each cell(superblock)
# layer idx 0 is the gate layer
def find_cell_first_conv_idx(conv_size_list):
    cells_first_layer_idx = [0]
    counter = -1
    for j in conv_size_list:
        if len(cells_first_layer_idx) == len(conv_size_list):
            break
        counter += len(j) + 1
        cells_first_layer_idx += [counter]
    return cells_first_layer_idx

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
    return in_rank, out_rank

# find the avg rank of the entire cell
def get_cell_ranks_avg(in_rank,out_rank,first_conv_idx,last_conv_idx,total_epoch):
    cell_avg = np.zeros(total_epoch)
    in_rank_true = True # flips from true to false and back to true
    shortcut_layer = first_conv_idx + 2
    count = 0
    # skip layer 0 (gate layer)
    if first_conv_idx == 0:
        first_conv_idx = 1
    for layer_idx in range(first_conv_idx, last_conv_idx+1):
        # skip the shortcut layer
        if layer_idx == shortcut_layer and first_conv_idx != 1:
            continue
        if in_rank_true:
            rank = in_rank[layer_idx]
            in_rank_true = False
            # print("in layer ",layer_idx)
        else:
            rank = out_rank[layer_idx]
            in_rank_true = True
            # print("out layer ", layer_idx)
        cell_avg += rank
        count += 1
    return cell_avg/count

# get the cell layer ranks by the dependency (mapping) indicated in paper
# returns avg ranks of each adjustable conv sizes in the format of conv_size_list
def get_cell_layer_ranks_by_dependency(in_rank,out_rank,cell_first_layer_idx):
    total_epoch = in_rank.shape[1]
    total_layer = in_rank.shape[0]
    channel_size_avg_ranks = list()

    for cur_cell_idx, first_conv_idx in enumerate(cell_first_layer_idx):
        cur_cell_avg_ranks = list()
        if cur_cell_idx == len(cell_first_layer_idx) - 1:
            last_conv_idx = total_layer - 1
        else:
            last_conv_idx = cell_first_layer_idx[cur_cell_idx+1] - 1
        #print(first_conv_idx, last_conv_idx)
        shortcut_layer_idx = first_conv_idx + 2
        # average of the entire cell ranks, used for C0 (as noted in paper)
        cell_avg = get_cell_ranks_avg(in_rank,out_rank,first_conv_idx,last_conv_idx,total_epoch)

        # first cell starts with a gate layer and the conv size is C0
        if cur_cell_idx == 0:
            cur_cell_avg_ranks.append(copy.deepcopy(cell_avg))
            first_conv_idx = 1
        is_C0 = False # alternates between False and True
        for layer_idx in range(first_conv_idx, last_conv_idx+1):
            # skip shortcut layers
            if layer_idx == shortcut_layer_idx and cur_cell_idx != 0:
                #print("is shortcut")
                continue
            # if the channel size is C0, use the entire cell rank avg
            if is_C0:
                cur_cell_avg_ranks.append(copy.deepcopy(cell_avg))
                is_C0 = False
                #print("is C0")
                continue
            # if its pair is shortcut layer, skip it (shouldn't be invoked?)
            if layer_idx + 1 == shortcut_layer_idx and cur_cell_idx != 0:
                pair_next = layer_idx + 2
            else:
                pair_next = layer_idx + 1
            pair_avg = (out_rank[layer_idx] + in_rank[pair_next])/2
            cur_cell_avg_ranks.append(copy.deepcopy(pair_avg))
            #print("out ", layer_idx, ",in ", pair_next)
            is_C0 = True
        channel_size_avg_ranks.append(copy.deepcopy(np.array(cur_cell_avg_ranks)))
    return channel_size_avg_ranks

# returns cell avg rank slope in format of the conv_size_list
def find_rank_avg_slope(cell_avg_ranks):
    cell_rank_slopes = list()
    for i,cell in enumerate(cell_avg_ranks):
        cur_cell_rank_slopes = list()
        for avg_rank in cell:
            max_rank_epoch = np.argmax(avg_rank) # find the index (epoch#) where the max avg rank is
            rank_delta = avg_rank[max_rank_epoch] - avg_rank[0]
            epoch_delta = max_rank_epoch - 0
            rank_slope = rank_delta/ (epoch_delta + 0.0001)
            cur_cell_rank_slopes.append(rank_slope)
            # if i == 0:
            #     print("max rank epoch: ",max_rank_epoch)
            #     print("max rank: ", avg_rank[max_rank_epoch])
            #     print("initial rank: ", avg_rank[0])
        cell_rank_slopes.append(copy.deepcopy(np.array(cur_cell_rank_slopes)))
    # print(cell_avg_ranks[1])
    # print(cell_rank_slopes[1])
    return cell_rank_slopes

# return last epoch's mapping condition for each layer
def get_layer_cond(trial_dir,cur_trial):
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

# find the avg mapping condition of the entire cell
def get_cell_cond_avg(in_cond,out_cond,first_conv_idx,last_conv_idx,total_epoch):
    cell_avg = 0
    in_cond_true = True # flips from true to false and back to true
    shortcut_layer = first_conv_idx + 2
    count = 0
    # skip layer 0 (gate layer)
    if first_conv_idx == 0:
        first_conv_idx = 1
    for layer_idx in range(first_conv_idx, last_conv_idx+1):
        # skip the shortcut layer
        if layer_idx == shortcut_layer and first_conv_idx != 1:
            continue
        if in_cond_true:
            cond = in_cond[layer_idx]
            in_cond_true = False
            # print("in layer ",layer_idx)
        else:
            cond = out_cond[layer_idx]
            in_cond_true = True
            # print("out layer ", layer_idx)
        cell_avg += cond
        count += 1
    return cell_avg/count

# get the mapping condition of the conv sizes in each cell
def get_cell_layer_cond_by_dependency(in_cond,out_cond,cell_first_layer_idx):
    total_epoch = 0 #in_cond and out_cond only has the values for the LAST layer
    total_layer = in_cond.shape[0]
    channel_size_avg_cond = list()

    for cur_cell_idx, first_conv_idx in enumerate(cell_first_layer_idx):
        cur_cell_avg_cond = list()
        if cur_cell_idx == len(cell_first_layer_idx) - 1:
            last_conv_idx = total_layer - 1
        else:
            last_conv_idx = cell_first_layer_idx[cur_cell_idx+1] - 1
        #print(first_conv_idx, last_conv_idx)
        shortcut_layer_idx = first_conv_idx + 2
        # average of the entire cell ranks, used for C0 (as noted in paper)
        cell_avg = get_cell_cond_avg(in_cond,out_cond,first_conv_idx,last_conv_idx,total_epoch)

        # first cell starts with a gate layer and the conv size is C0
        if cur_cell_idx == 0:
            cur_cell_avg_cond.append(copy.deepcopy(cell_avg))
            first_conv_idx = 1
        is_C0 = False # alternates between False and True
        for layer_idx in range(first_conv_idx, last_conv_idx+1):
            # skip shortcut layers
            if layer_idx == shortcut_layer_idx and cur_cell_idx != 0:
                #print("is shortcut")
                continue
            # if the channel size is C0, use the entire cell rank avg
            if is_C0:
                cur_cell_avg_cond.append(copy.deepcopy(cell_avg))
                is_C0 = False
                #print("is C0")
                continue
            # if its pair is shortcut layer, skip it (shouldn't be invoked?)
            if layer_idx + 1 == shortcut_layer_idx and cur_cell_idx != 0:
                pair_next = layer_idx + 2
            else:
                pair_next = layer_idx + 1
            pair_avg = (out_cond[layer_idx] + in_cond[pair_next])/2
            cur_cell_avg_cond.append(pair_avg)
            #print("out ", layer_idx, ",in ", pair_next)
            is_C0 = True
        channel_size_avg_cond.append(copy.deepcopy(np.array(cur_cell_avg_cond)))
    return channel_size_avg_cond

# provide conv_size_list to generate a prev operation list and factor list
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
    return prev_op_list, factor_list

# round to nearest even number for chan size
def round_even(number):
    return int(round(number / 2) * 2)

# returns new conv_size_list, and previous operation and scaling
# chan sizes are rounded to nearest even int
def channel_size_adjust_algorithm(conv_size_list, delta_threshold, min_scale_limit, map_cond_threshold, min_conv_size, max_conv_size,
                                    trial_dir, cur_trial, cell_layer_prev_ops, cell_layer_factor):
    first_trial = False
    if len(cell_layer_prev_ops) == 0 or cell_layer_prev_ops == None:
        first_trial = True
        cell_layer_prev_ops, cell_layer_factor = initialize_algorithm(conv_size_list)

    # get the avg rank slope of each adjustable channel sizes
    cells_first_layer_idx = find_cell_first_conv_idx(conv_size_list)
    in_rank, out_rank = get_layer_conv_ranks(trial_dir, cur_trial)
    cell_layer_avg_ranks = get_cell_layer_ranks_by_dependency(in_rank, out_rank, cells_first_layer_idx)
    cell_layer_rank_slopes = find_rank_avg_slope(cell_layer_avg_ranks)

    # get the avg mapping condition for each adjustable channel sizes
    in_cond, out_cond = get_layer_cond(trial_dir, cur_trial)
    cell_layer_cond = get_cell_layer_cond_by_dependency(in_cond, out_cond, cells_first_layer_idx)

    # algorithm: operation = expand:1/shrink:-1/stop:0
    for cell_idx, cell in enumerate(cell_layer_rank_slopes):
        for chan_idx, avg_rank_slope in enumerate(cell):
            map_cond = cell_layer_cond[cell_idx][chan_idx]
            chan_size = conv_size_list[cell_idx][chan_idx]
            prev_op = cell_layer_prev_ops[cell_idx][chan_idx]
            chan_scale = cell_layer_factor[cell_idx][chan_idx]
            if not first_trial and prev_op == 0:
                continue  # if threshold already satisfied, skip this conv
            cur_op = 1  # initialize operation to expand first.
            if ((avg_rank_slope < delta_threshold or map_cond >= map_cond_threshold) and chan_size > min_conv_size) or (
                    chan_size > max_conv_size):
                cur_op = -1
            if prev_op != cur_op and not first_trial:
                if chan_scale < min_scale_limit:
                    cur_op = 0
                cell_layer_factor[cell_idx][chan_idx] = chan_scale / 2
            cell_layer_prev_ops[cell_idx][chan_idx] = cur_op
            new_cell_size = round_even(chan_size * (1 + cell_layer_factor[cell_idx][chan_idx] * cur_op))
            if new_cell_size > max_conv_size:
                new_cell_size = max_conv_size
            elif new_cell_size < min_conv_size:
                new_cell_size = min_conv_size
            conv_size_list[cell_idx][chan_idx] = new_cell_size

    return conv_size_list, cell_layer_rank_slopes, cell_layer_prev_ops, cell_layer_factor, cell_layer_avg_ranks


if __name__ == '__main__':
    super1_idx = [32, 32, 32, 32, 32, 32, 32]
    super2_idx = [32, 32, 32, 32, 32, 32, 32, 32]
    super3_idx = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
    super4_idx = [32, 32, 32, 32, 32, 32]

    conv_size_list = [super1_idx, super2_idx, super3_idx, super4_idx]
    cells_first_layer_idx = find_cell_first_conv_idx(conv_size_list)
    print("First layer of each cell (superblock): ",cells_first_layer_idx)
    print("Original conv_size_list: ",conv_size_list)

    trial_dir = ".\\"
    cur_trial = 0
    in_rank, out_rank = get_layer_conv_ranks(trial_dir, cur_trial)
    # print(in_rank)
    # print(out_rank)
    cell_layer_avg_ranks = get_cell_layer_ranks_by_dependency(in_rank, out_rank, cells_first_layer_idx)
    cell_layer_rank_slopes = find_rank_avg_slope(cell_layer_avg_ranks)
    print("cell layer avg rank slopes:")
    print(cell_layer_rank_slopes)
    # # print(cell_layer_avg_ranks[0])
    # print(cell_layer_rank_slopes[0])
    # total_epoch = in_rank.shape[1]
    # first_conv_idx = 0
    # last_conv_idx = 6
    # print(get_cell_ranks_avg(in_rank,out_rank,first_conv_idx,last_conv_idx,total_epoch))
    in_cond, out_cond = get_layer_cond(trial_dir, cur_trial)
    cell_layer_cond = get_cell_layer_cond_by_dependency(in_cond,out_cond,cells_first_layer_idx)
    print("cell layer mapping condition:")
    print(cell_layer_cond)
    # print(in_cond)
    # print(out_cond)
    # print(cell_layer_cond[0])

    # thresholds for testing
    delta_threshold = 0.025
    min_scale_limit = 0.001
    map_cond_threshold = 10
    max_conv_size = 256
    min_conv_size = 32
    cell_layer_prev_ops = []
    cell_layer_factor = []
    conv_size_list, cell_layer_rank_slopes, cell_layer_prev_ops, cell_layer_factor, cell_layer_avg_ranks = channel_size_adjust_algorithm(conv_size_list, delta_threshold, min_scale_limit, map_cond_threshold, min_conv_size, max_conv_size,
                                    trial_dir, cur_trial, cell_layer_prev_ops, cell_layer_factor)

    print("delta threshold is: ", delta_threshold)
    print("min conv size is: ", min_conv_size)
    print("max conv size is: ", max_conv_size)
    print("mapping condition threshold is: ", map_cond_threshold)
    print("new conv_size_list:")
    print(conv_size_list)
    print("prev op:")
    print(cell_layer_prev_ops)
    print("layer factor:")
    print(cell_layer_factor)





