"""
MIT License

Copyright (c) 2021 Mahdi S. Hosseini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import global_vars as GLOBALS
import ast
import copy
import platform
import time
def calculate_correct_output_sizes(input_ranks,output_ranks,conv_size_list,shortcut_indexes,threshold,final=True):
    #Note that input_ranks/output_ranks may have a different size than conv_size_list
    #threshold=GLOBALS.CONFIG['adapt_rank_threshold']
    '''
    input_ranks_layer_1, output_ranks_layer_1 = input_ranks[0], output_ranks[0]

    input_ranks_superblock_1, output_ranks_superblock_1 = input_ranks[1:shortcut_indexes[0]], output_ranks[1:shortcut_indexes[0]]
    input_ranks_superblock_2, output_ranks_superblock_2 = input_ranks[shortcut_indexes[0]+1:shortcut_indexes[1]], output_ranks[shortcut_indexes[0]+1:shortcut_indexes[1]]
    input_ranks_superblock_3, output_ranks_superblock_3 = input_ranks[shortcut_indexes[1]+1:shortcut_indexes[2]], output_ranks[shortcut_indexes[1]+1:shortcut_indexes[2]]
    input_ranks_superblock_4, output_ranks_superblock_4 = input_ranks[shortcut_indexes[2]+1:shortcut_indexes[3]], output_ranks[shortcut_indexes[2]+1:shortcut_indexes[3]]
    input_ranks_superblock_5, output_ranks_superblock_5 = input_ranks[shortcut_indexes[3]+1:], output_ranks[shortcut_indexes[2]+1:shortcut_indexes[3]]'''

    temp_shortcut_indexes=[0]+shortcut_indexes+[len(input_ranks)]
    new_input_ranks=[]
    new_output_ranks=[]
    for i in range(0,len(temp_shortcut_indexes)-1,1):
        new_input_ranks+=[input_ranks[temp_shortcut_indexes[i]+1:temp_shortcut_indexes[i+1]]]
        new_output_ranks+=[output_ranks[temp_shortcut_indexes[i]+1:temp_shortcut_indexes[i+1]]]

    #new_input_ranks = [input_ranks_superblock_1] + [input_ranks_superblock_2] + [input_ranks_superblock_3] + [input_ranks_superblock_4] + [input_ranks_superblock_5]
    #new_output_ranks = [output_ranks_superblock_1] + [output_ranks_superblock_2] + [output_ranks_superblock_3] + [output_ranks_superblock_4] + [output_ranks_superblock_5]

    #print(new_input_ranks,'INPUT RANKS WITHOUT SHORTCUTS')
    #print(new_output_ranks,'OUTPUT RANKS WITHOUT SHORTCUTS')

    block_averages=[]
    block_averages_input=[]
    block_averages_output=[]
    grey_list_input=[]
    grey_list_output=[]

    for i in range(0,len(new_input_ranks),1):
        block_averages+=[[]]
        block_averages_input+=[[]]
        block_averages_output+=[[]]
        grey_list_input+=[[]]
        grey_list_output+=[[]]
        temp_counter=0
        for j in range(1,len(new_input_ranks[i]),2):
            block_averages_input[i]=block_averages_input[i]+[new_input_ranks[i][j]]
            block_averages_output[i]=block_averages_output[i]+[new_output_ranks[i][j-1]]

            grey_list_input[i]=grey_list_input[i]+[new_input_ranks[i][j-1]]
            grey_list_output[i]=grey_list_output[i]+[new_output_ranks[i][j]]

        block_averages_input[i]=block_averages_input[i]+[np.average(np.array(grey_list_input[i]))]
        block_averages_output[i]=block_averages_output[i]+[np.average(np.array(grey_list_output[i]))]
        block_averages[i]=np.average(np.array([block_averages_input[i],block_averages_output[i]]),axis=0)

    #print(conv_size_list,'CONV SIZE LIST')
    output_conv_size_list=copy.deepcopy(conv_size_list)
    rank_averages = copy.deepcopy(conv_size_list)

    for i in range(0,len(block_averages)):
        for j in range(0,len(conv_size_list[i])):
            if (i==0):
                if (j%2==0):
                    scaling_factor=block_averages[i][-1]-threshold
                else:
                    scaling_factor=block_averages[i][int((j-1)/2)]-threshold
            else:
                if (j%2==1):
                    scaling_factor=block_averages[i][-1]-threshold
                else:
                    scaling_factor=block_averages[i][int(j/2)]-threshold
            output_conv_size_list[i][j]=even_round(output_conv_size_list[i][j]*(1+scaling_factor))
            rank_averages[i][j] = scaling_factor + threshold

    if final==True:
        GLOBALS.super1_idx = output_conv_size_list[0]
        GLOBALS.super2_idx = output_conv_size_list[1]
        GLOBALS.super3_idx = output_conv_size_list[2]
        GLOBALS.super4_idx = output_conv_size_list[3]
        GLOBALS.super5_idx = output_conv_size_list[4]
        GLOBALS.index = output_conv_size_list[0] + output_conv_size_list[1] + output_conv_size_list[2] + output_conv_size_list[3] + output_conv_size_list[4]

    #print(output_conv_size_list,'OUTPUT CONV SIZE LIST')
    return output_conv_size_list, rank_averages

def get_ranks(path = GLOBALS.EXCEL_PATH, epoch_number = -1):
    '''
    - Read from .adas-output excel file
    - Get Final epoch ranks
    '''
    sheet = pd.read_excel(path,index_col=0)
    out_rank_col = [col for col in sheet if col.startswith('out_rank')]
    in_rank_col = [col for col in sheet if col.startswith('in_rank')]

    out_ranks = sheet[out_rank_col]
    in_ranks = sheet[in_rank_col]

    last_rank_col_out = out_ranks.iloc[:,epoch_number]
    last_rank_col_in = in_ranks.iloc[:,epoch_number]

    last_rank_col_in = last_rank_col_in.tolist()
    last_rank_col_out = last_rank_col_out.tolist()

    return last_rank_col_in, last_rank_col_out
def compile_adaptive_files(file_name,num_trials):
    #CHANGE THIS VALUE FOR NUMBER OF TRIALS
    num_trials=num_trials
    adaptive_set=[]
    manipulate_index=file_name.find('trial')+6
    try:
        blah=int(file_name[manipulate_index+1])
        shift=2
    except:
        shift=1
    for trial_num in range (0,num_trials):
        adaptive_set.append(file_name[0:manipulate_index]+str(trial_num)+file_name[manipulate_index+shift:])

    return adaptive_set



def create_adaptive_graphs(file_name,num_epochs,num_trials,out_folder):
    #CHANGE THIS VALUE FOR NUMBER OF EPOCHS PER TRIAL
    total_num_epochs=num_epochs
    accuracies=[]
    epoch_num=[]
    count=0

    adaptive_set=compile_adaptive_files(file_name,num_trials)
    #print(adaptive_set,'adaptive_set')
    for trial in adaptive_set:
        dfs=pd.read_excel(trial)
        #print(dfs)
        for epoch in range (0,total_num_epochs):
            epoch_num.append(epoch+count)
            accuracies.append(dfs['test_acc_epoch_'+str(epoch)][0]*100)
            new_trial_indic=''
        count+=total_num_epochs
#    print(epoch_num)
#    print(accuracies)
    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'
    fig=plt.plot(epoch_num,accuracies, label='accuracy vs epoch', marker='o', color='r')
    #figure=plt.gcf()
    #figure.set_size_inches(16, 9)
    plt.xticks(np.arange(min(epoch_num), max(epoch_num)+1, total_num_epochs))
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Dynamic DASNet: Test Accuracy vs Epoch (init_conv_size='+GLOBALS.CONFIG['init_conv_setting']+' delta_thresh='+str(GLOBALS.CONFIG['delta_threshold'])+')')
    plt.savefig(out_folder+slash+'dynamic_accuracy_plot.png',bbox_inches='tight')
    #plt.show()

#create_adaptive_graphs()

def remove_brackets(value):
    check=']'
    val=''
    for i in range(0,len(value),1):
        if i==len(value)-1:
            val+=']'
            break
        if value[i]==check:
            if check==']':
                check='['
                if value[i+1]==check:
                    val+=', '
                    i+=2
            else:
                check=']'
        else:
            val+=value[i]
    return val
def get_trial_info(file_name,num_trials,num_layers,specified_epoch, skip_connections,info):
    adaptive_set=compile_adaptive_files(file_name,num_trials)
    final_output = []
    for i in range(0,num_trials,1):
        final_output.append([])
    for i in range(0,num_trials):
        dfs=pd.read_excel(adaptive_set[i])
        #print(dfs)
        for j in range(0,num_layers):
            if (j in skip_connections):
                continue
            final_output[i].append(dfs[info+str(specified_epoch)][j])
    return final_output

'''def create_plot(layers_size_list,num_trials,path,evo_type,specified_epoch,trial_increment=1):
    layers_list=[[]]
    if num_trials<=10:
        mult_val,temp_val=6,5
    else:
        mult_val,temp_val=(45/20)*num_trials,10

    barWidth=(0.5/6)*mult_val
    if num_trials<=10:
        trueWidth=barWidth
    else:
        trueWidth=(2/20)*num_trials

    true_temp=[]
    for i in range(0,len(layers_size_list),1):
        if i%trial_increment==0:
            true_temp+=[layers_size_list[i]]
    layers_size_list=true_temp

    for i in range(1,len(layers_size_list[0])+1,1):
        layers_list[0]+=[mult_val*i]

    for i in range(1,len(layers_size_list),1):
        temp=[x + trueWidth for x in layers_list[i-1]]
        layers_list+=[temp]

    colors=['#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045']
    plt.figure()
    for i in range(0,len(layers_size_list),1):
        plt.bar(layers_list[i],layers_size_list[i],color=colors[i],width=trueWidth, edgecolor='white',label=str('Trial '+str(trial_increment*i+1)))

    plt.xlabel('SuperBlock',fontweight='bold')
    plt.ylabel('Layer Size',fontweight='bold')
    #plt.title('DASNet:' + evo_type + ' Evolution w.r.t. Trial (init_conv_size='+GLOBALS.CONFIG['init_conv_setting']+' delta_thresh='+str(GLOBALS.CONFIG['delta_threshold'])+')')
    if num_trials<=20:
        plt.xticks([mult_val*r + temp_val*barWidth + 3 + num_trials*0.3 for r in range(len(temp))], [str(i) for i in range(len(temp))])
    else:
        plt.xticks([mult_val*r + num_trials*0.3 + 3*num_trials for r in range(len(temp))], [str(i) for i in range(len(temp))])

    plt.legend(loc='upper right')
    figure=plt.gcf()
    if num_trials<=20:
        figure.set_size_inches(15.4, 5.34)
    else:
        figure.set_size_inches(45.4, 5.34)
    #addition=str(GLOBALS.CONFIG['adapt_rank_threshold'])+'_conv_size='+GLOBALS.CONFIG['init_conv_setting']+'_epochpertrial='+str(GLOBALS.CONFIG['epochs_per_trial'])+'_beta='+str(GLOBALS.CONFIG['beta'])
    plt.savefig(path,bbox_inches='tight')
    return True

def adapted_info_graph(adapted_file_name,num_trials,path,evo_type,specified_epoch):
    layers_info=pd.read_excel(adapted_file_name) #This file_name is an adapted_blah file_name
    layers_size_list=[]

    for i in range(len(layers_info.iloc[:,0].to_numpy())):
        temp=''
        main=layers_info.iloc[i,1:].to_numpy()
        for j in main:
            temp+=j[:]
        temp=ast.literal_eval(remove_brackets(temp))
        layers_size_list+=[temp]

    create_plot(layers_size_list,num_trials,path,evo_type,specified_epoch)'''

def create_plot(layers_size_list,num_trials,path,evo_type,specified_epoch,trial_increment=1):
    layers_list=[[]]
    if num_trials<=10:
        mult_val,temp_val=6,5
    else:
        mult_val,temp_val=(45/22)*num_trials,10

    barWidth=(0.5/6)*mult_val
    if num_trials<=10:
        trueWidth=barWidth
    else:
        trueWidth=(1.2/15)*num_trials

    true_temp=[]
    for i in range(0,len(layers_size_list),1):
        if i%trial_increment==0:
            true_temp+=[layers_size_list[i]]
    layers_size_list=true_temp

    for i in range(1,len(layers_size_list[0])+1,1):
        layers_list[0]+=[mult_val*i]

    for i in range(1,len(layers_size_list),1):
        temp=[x + trueWidth for x in layers_list[i-1]]
        layers_list+=[temp]

    colors=['#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045']
    plt.figure()
    for i in range(0,len(layers_size_list),1):
        plt.bar(layers_list[i],layers_size_list[i],color=colors[i],width=trueWidth, edgecolor='white',label=str('Trial '+str(trial_increment*i+1)))

    plt.xlabel('SuperBlock',fontweight='bold')
    plt.ylabel('Layer Size',fontweight='bold')
    plt.title('DASNet:' + evo_type + ' Evolution w.r.t. Trial (init_conv_size='+GLOBALS.CONFIG['init_conv_setting']+' delta_thresh='+str(GLOBALS.CONFIG['delta_threshold'])+')')
    if num_trials<=20:
        plt.xticks([mult_val*r + temp_val*barWidth + 3 + num_trials*0.3 for r in range(len(temp))], [str(i) for i in range(len(temp))])
    else:
        plt.xticks([mult_val*r + num_trials*0.3 + 3*num_trials for r in range(len(temp))], [str(i) for i in range(len(temp))])

    plt.legend(loc='upper right')
    figure=plt.gcf()
    if num_trials<=20:
        figure.set_size_inches(15.4, 5.34)
    else:
        figure.set_size_inches(45.4, 5.34)
    #addition=str(GLOBALS.CONFIG['adapt_rank_threshold'])+'_conv_size='+GLOBALS.CONFIG['init_conv_setting']+'_epochpertrial='+str(GLOBALS.CONFIG['epochs_per_trial'])+'_beta='+str(GLOBALS.CONFIG['beta'])
    plt.savefig(path,bbox_inches='tight')
    return True

def adapted_info_graph(adapted_file_name,num_trials,path,evo_type,specified_epoch):
    layers_info=pd.read_excel(adapted_file_name) #This file_name is an adapted_blah file_name
    layers_size_list=[]

    for i in range(len(layers_info.iloc[:,0].to_numpy())):
        temp=''
        main=layers_info.iloc[i,1:].to_numpy()
        for j in main:
            temp+=j[:]
        temp=ast.literal_eval(remove_brackets(temp))
        layers_size_list+=[temp]

    create_plot(layers_size_list,num_trials,path,evo_type,specified_epoch)

def trial_info_graph(trial_file_name,num_trials,num_layers, path,evo_type,info,shortcut_indexes,specified_epoch):
    layers_size_list=get_trial_info(trial_file_name, num_trials, num_layers, specified_epoch,shortcut_indexes,info)
    create_plot(layers_size_list,num_trials,path,evo_type,specified_epoch)

def even_round(number):
    return int(round(number/2)*2)

def adaptive_stop(x_data,y_data,threshold_min,epoch_wait):
    '''From the wth epoch, If there is an increase of x in any of the next y epochs, keep going.
    If not, make the value at the wth epoch the max'''
    ranks=[0.1,0.2,0.3,0.4,0.5]
    condition=False
    for i in range(0,len(x_data)-epoch_wait,1):
        condition=False
        for j in range(i+1,epoch_wait+i+1,1):
            if ((y_data[j]-y_data[i])>threshold_min):
                condition=True
                break
        if condition==False:
            return i
        '''if condition==True:
            #final_vals=[(i,y_data[i]) for i in x_data[-epoch_wait:]]
            #final_vals=final_vals.sort(key=lambda tup: tup[1])
            #return final_vals[-1][0]
            return len(x_data)-1'''
    return len(x_data)-1


def slope(y_data,breakpoint):
    return (y_data[int(breakpoint)]-y_data[GLOBALS.CONFIG['stable_epoch']])/(breakpoint-GLOBALS.CONFIG['stable_epoch']+0.0001)

def slope_clone(y_data,breakpoint):
    return (y_data[int(breakpoint)]-y_data[0])/(breakpoint+0.0001)



def calculate_slopes(conv_size_list,shortcut_indexes,path=GLOBALS.EXCEL_PATH):
    start=time.time()
    slope_averages=[]
    for i in conv_size_list:
        slope_averages.append([0.1]*len(i))

    epoch_num=[i for i in range(GLOBALS.CONFIG['epochs_per_trial'])]
    for superblock in range (0,len(conv_size_list),1):
        for layer_num in range (0,len(conv_size_list[superblock]),1):
            yaxis=[]
            for k in range(GLOBALS.CONFIG['epochs_per_trial']):
                input_ranks,output_ranks=get_ranks(path=path,epoch_number=k)
                rank_averages=calculate_correct_output_sizes(input_ranks, output_ranks, conv_size_list, shortcut_indexes, 0.1,final=False)[1]
                yaxis+=[rank_averages[superblock][layer_num]]

            #print(yaxis,'yaxis')
            break_point = adaptive_stop(epoch_num,yaxis,0.005,4)
            #print(break_point,'breakpoint')

            slope_averages[superblock][layer_num] = slope(yaxis,break_point)
            #print(slope_averages,'SLOPE AVERAGES')
    end=time.time()
    print(end-start,'TIME ELAPSED FOR CSL')
    return slope_averages

def stacked_bar_plot(adapted_file_name, path, trial_increment=2):
    '''
    sizes_with_trials is a list of lists as follows:
    sizes_with_trials=[sizes for trial1, sizes for 2, ... sizes for trial N]
    '''
    layers_info=pd.read_excel(adapted_file_name) #This file_name is an adapted_blah file_name
    layers_size_list=[]

    for i in range(len(layers_info.iloc[:,0].to_numpy())):
        temp=''
        main=layers_info.iloc[i,1:].to_numpy()
        for j in main:
            temp+=j[:]
        temp=ast.literal_eval(remove_brackets(temp))
        layers_size_list+=[temp]
    temp=[]
    alternate=False
    sizes_with_trials=[]
    if trial_increment!=1:
        for i in range(0,len(layers_size_list),1):
            if i%trial_increment==0:
                temp+=[layers_size_list[i]]
        layers_size_list=temp
    sizes_with_trials=np.transpose(layers_size_list).tolist()

    x_values=np.arange(len(sizes_with_trials[0]))
    temp=[0 for i in x_values]
    colors=['#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#B6D094','#4d4d4e','#b51b1b','#1f639b',
            '#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5',
            '#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e',
            '#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5',
            '#fcb045','#B6D094','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#B6D094']
    barWidth=0.5
    for i in range(0,len(sizes_with_trials),1):
        if i==0:
            plt.bar(x_values,sizes_with_trials[i],color=colors[i],width=barWidth)
        else:
            plt.bar(x_values,sizes_with_trials[i],bottom=temp, color=str(colors[i]),width=barWidth)
            temp=np.add(temp,sizes_with_trials[i]).tolist()
    names=[str(trial_increment*i) for i in range(len(x_values))]
    names[0]='Baseline'
    plt.xticks(x_values, names, fontweight='bold')
    plt.xlabel('Trial Number')
    plt.ylabel('Cumulative Channel Size')

    plt.title('ResNet-like Architecture w/Channel Size ='+GLOBALS.CONFIG['init_conv_setting'][:2]+', Threshold='+str(GLOBALS.CONFIG['delta_threshold'])+', MC Threshold='+str(GLOBALS.CONFIG['mapping_condition_threshold']))
    plt.title('ResNet-like Architecture w/Channel Size =443, Threshold=, MC Threshold=')
    figure=plt.gcf()
    figure.set_size_inches(11.4, 5.34)
    plt.savefig(path,bbox_inches='tight')

def create_rank_graph(file_name,shortcut_indexes,conv_size_list):
    #superblock=4
    layer=15
    num_epochs=15
    epoch_num=[i for i in range(num_epochs)]
    yaxis=[]
    for k in range(num_epochs):

        #data=get_trial_info(file_name,1,36,k, shortcut_indexes,'mode_rank_epoch_')[0]
        #print(len(data),'DATA')

        #data=[1,2,3,4,5,6,7,8,9,10,11,....,numlayers]

        input_ranks, output_ranks = get_ranks(path=file_name, epoch_number = k)
        rank_averages=calculate_correct_output_sizes(input_ranks, output_ranks, conv_size_list, shortcut_indexes, 0.1,final=False)[1]
        yaxis+=[rank_averages[superblock][layer]]
        #yaxis+=[data[layer]]


    #print(yaxis,'YAXIS VALUES')
    break_point = adaptive_stop(epoch_num,yaxis,0.005,4)

    fig=plt.plot(epoch_num,yaxis, marker='o', color='r', label='_nolegend_')
    fig=plt.axvline(x=break_point)

    plt.ylim([0,0.35])
    #x_smooth,y_smooth=our_fit(np.asarray(epoch_num),np.asarray(yaxis))
    #fig=plt.plot(x_smooth,y_smooth,label='smooth curve', color='b')
    print(slope_clone(yaxis,break_point),'--------------------------SLOPE OF GRAPH--------------------------')

    x1, y1 = epoch_num[0], yaxis[0]
    x2, y2 = break_point, yaxis[break_point]
    m = round((y2 - y1)/(x2 - x1), 3)
    x_val = [x1, x2]
    y_val = [y1, y2]
    plt.plot(x_val, y_val, label='Slope {}'.format(m), color='g')
    plt.legend()
    plt.title('Channel Size {}'.format(conv))
    plt.xlabel('Epoch')
    plt.ylabel('Rank')
    plt.show()
    return True

# create_rank_graph('/mnt/c/users/andre/desktop/multimedia-lab/output1/conv_32,32,32,9_deltaThresh=0.02_minScaleLimit=0.01_beta=0.7_epochpert=20_adaptnum=35/Trials\AdaS_adapt_trial=1_net=DASNet34_0.1_dataset=CIFAR10.xlsx',[7,16,29], '128')
#create_rank_graph('AdaS_adapt_trial=0_net=DASNet34_0.1_dataset=CIFAR10.xlsx',[7,16,29])
#adapted_info_graph('adapted_architectures.xlsx',35,'temp.png','Layer Size',-1)

#create_rank_graph('AdaS_adapt_trial=0_net=DASNet34_0.1_dataset=CIFAR10.xlsx',[7,16,29], [[128, 34, 128, 176, 128, 160, 128],[256, 212, 248, 212, 192, 212, 142, 212],[258, 178, 256, 178, 220, 178, 148, 178, 128, 178, 156, 178],[188, 182, 144, 182, 132, 182]])
'''
shortcut_indexes=[7,16,29]
conv_size_list64=[[64,64,64,64,64,64,64],[64,64,64,64,64,64],[64,64,64,64,64,64],[64,64,64,64,64,64],[64,64,64,64,64,64]]
conv_size_list32=[[32,32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32]]
conv_size_list96=[[96,96,96,96,96,96,96],[96,96,96,96,96,96],[96,96,96,96,96,96],[96,96,96,96,96,96],[96,96,96,96,96,96]]
conv_size_list128=[[86, 80, 86, 68, 86, 74, 86],[684, 132, 162, 132, 138, 132, 126, 132],[1042, 110, 298, 110, 160, 110, 126, 110, 120, 110, 96, 110],[1042, 68, 138, 68, 66, 68]]
#create_rank_graph(conv_size_list32,shortcut_indexes,path='32.xlsx')
print(create_rank_graph(conv_size_list128,shortcut_indexes,path='AdaS_adapt_trial=11_net=AdaptiveNet_0.1_dataset=CIFAR10.xlsx'))

shortcut_indexes=[7,16,29]
output_conv_size_list,mapping_averages=calculate_correct_output_sizes(input_mapping,output_mapping,conv_size_list,shortcut_indexes,threshold,)
create_layer_plot(file_name,20,'Mapping Condition')

in_a,out_a=create_mc_plot('AdaS_adapt_trial=0_net=AdaptiveNet_0.1_dataset=CIFAR10.xlsx',20,25,19,[5,10,15,20])

print(in_a,'input mapping conditions')
print(out_a,'output mapping conditions')

in_path='adas_search\\in_mc_graph.png'
out_path='adas_search\\out_mc_graph.png'
create_mapping_plot(in_path,in_a,20,'In Mapping Condition')
create_mapping_plot(out_path,out_a,20,'out Mapping Condition')'''
