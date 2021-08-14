import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import operator

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

def stacked_bar_plot(adapted_file_name, trial_increment=2):
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
    colors=['#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#B6D094','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045']
    barWidth=0.5
    for i in range(0,len(sizes_with_trials),1):
        if i==0:
            plt.bar(x_values,sizes_with_trials[i],color=colors[i],width=barWidth)
        else:
            #Just for safety
            plt.bar(x_values,sizes_with_trials[i],bottom=temp, color=str(colors[i]),width=barWidth)
        temp=np.add(temp,sizes_with_trials[i]).tolist()
    names=[str(trial_increment*i) for i in range(len(x_values))]
    names[0]='Baseline'
    plt.xticks(x_values, names, fontweight='bold')
    plt.xlabel('Trial Number')
    plt.ylabel('Cumulative Channel Size')
    plt.title('ResNet-like Architecture w/Channel Size = 32, Threshold=0.01, MC Threshold=8')
    figure=plt.gcf()
    figure.set_size_inches(11.4, 5.34)
    plt.savefig('temp.png',bbox_inches='tight')

stacked_bar_plot('adapted_architectures.xlsx')
