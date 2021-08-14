
from typing import Any, Dict, Union, List

import torch
#from lr_scheduler import StepLR, CosineAnnealingWarmRestarts, OneCycleLR
from optim.adabound import AdaBound
from optim.sgd import SGDVec, SGD
from AdaS import AdaS

def get_optimizer_scheduler(
        net_parameters: Any,
        listed_params: List[Any],
        train_loader_len: int,
        config: Dict[str, Union[float, str, int]]) -> torch.nn.Module:
    # init_lr: float, optim_method: str,
    # lr_scheduler: str,
    # train_loader_len: int,
    # max_epochs: int) -> torch.nn.Module:
    optim_method = config['optim_method']
    lr_scheduler = config['lr_scheduler']
    init_lr = config['init_lr']
    min_lr = config['min_lr']
    max_epochs = config['max_epoch']
    adas_p = config['p']
    beta = config['beta']
    zeta = config['zeta']
    weight_decay = float(config['weight_decay'])

    step_size = config['step_size']
    gamma = config['gamma']

    print('~~~ BETA USED IN get_optimizer_scheduler: {} ~~~'.format(beta))
    print('~~~ lr_scheduler USED IN get_optimizer_scheduler: {} ~~~'.format(lr_scheduler))
    print('~~~ Weight Decay: {} ~~~'.format(weight_decay))

    if lr_scheduler == 'StepLR':
        print('~~~ LR Step Size: {} ~~~'.format(step_size))
        print('~~~ LR Gamma: {} ~~~'.format(gamma))
    optimizer = None
    scheduler = None
    if optim_method == 'SGD':
        if lr_scheduler == 'AdaS':
            optimizer = SGDVec(
                net_parameters, lr=init_lr,
                momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = SGD(
                net_parameters, lr=init_lr,
                momentum=0.9, weight_decay=weight_decay)
    elif optim_method == 'NAG':
        optimizer = SGD(
            net_parameters, lr=init_lr,
            momentum=0.9, weight_decay=weight_decay,
            nesterov=True)
    elif optim_method == 'AdaM':
        optimizer = Adam(net_parameters, lr=init_lr)
    elif optim_method == 'AdaGrad':
        optimizer = Adagrad(net_parameters, lr=init_lr)
    elif optim_method == 'RMSProp':
        optimizer = RMSprop(net_parameters, lr=init_lr)
    elif optim_method == 'AdaDelta':
        optimizer = Adadelta(net_parameters, lr=init_lr)
    elif optim_method == 'AdaBound':
        optimizer = AdaBound(net_parameters, lr=init_lr)
    elif optim_method == 'AMSBound':
        optimizer = AdaBound(net_parameters, lr=init_lr, amsbound=True)
    # below = untested
    elif optim_method == 'AdaMax':
        optimizer = Adamax(net_parameters, lr=init_lr)
    elif optim_method == 'AdaMod':
        optimizer = AdaMod(net_parameters, lr=init_lr)
    elif optim_method == 'AdaShift':
        optimizer = AdaShift(net_parameters, lr=init_lr)
    elif optim_method == 'NAdam':
        optimizer = NAdam(net_parameters, lr=init_lr)
    elif optim_method == 'NosAdam':
        optimizer = NosAdam(net_parameters, lr=init_lr)
    elif optim_method == 'NovoGrad':
        optimizer = NovoGrad(net_parameters, lr=init_lr)
    elif optim_method == 'PAdam':
        optimizer = PAdam(net_parameters, lr=init_lr)
    elif optim_method == 'RAdam':
        optimizer = RAdam(net_parameters, lr=init_lr)
    elif optim_method == 'SPS':
        optimizer = SPS(net_parameters, init_step_size=init_lr)
    elif optim_method == 'SLS':
        optimizer = SLS(net_parameters, init_step_size=init_lr)
    elif optim_method == 'LaProp':
        optimizer = LaProp(net_parameters, lr=init_lr)
    elif optim_method == 'LearningRateDropout':
        optimizer = LRD(net_parameters, lr=init_lr,
                        lr_dropout_rate=0.5)
    else:
        print(f"Adas: Warning: Unknown optimizer {optim_method}")
    if lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler == 'CosineAnnealingWarmRestarts':
        first_restart_epochs = 25
        increasing_factor = 1
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=first_restart_epochs, T_mult=increasing_factor)
    elif lr_scheduler == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer, max_lr=init_lr,
            steps_per_epoch=train_loader_len, epochs=max_epochs)
    elif lr_scheduler == 'AdaS':
        scheduler = AdaS(parameters=listed_params,
                         beta=beta,
                         zeta=zeta,
                         init_lr=init_lr,
                         min_lr=min_lr,
                         p=adas_p)
    elif lr_scheduler not in ['None', '']:
        print(f"Adas: Warning: Unknown LR scheduler {lr_scheduler}")
    return (optimizer, scheduler)
