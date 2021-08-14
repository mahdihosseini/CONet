import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

    def __init__(self, arch, genotype, C_prev_prev, C_prev, C_set, sep_conv_set, reduction, reduction_prev):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)

        if arch == "DARTS_PLUS_CIFAR100":
            if reduction_prev:
                self.preprocess0 = FactorizedReduce(C_prev_prev, C_set[0])
            else:
                self.preprocess0 = ReLUConvBN(C_prev_prev, C_set[0], 1, 1, 0)

            if reduction:
                self.preprocess1 = ReLUConvBN(C_prev, C_set[1], 1, 1, 0)  #This is kinda redundant now
            else:
                self.preprocess1 = ReLUConvBN(C_prev, C_set[1], 1, 1, 0)

            if reduction:
                op_names, indices = zip(*genotype.reduce)
                concat = genotype.reduce_concat
            else:
                op_names, indices = zip(*genotype.normal)
                concat = genotype.normal_concat
            self._compile_DARTS_PLUS(C_set, sep_conv_set, op_names, indices, concat, reduction)

        elif arch == "DARTS":
            if reduction_prev:
                self.preprocess0 = FactorizedReduce(C_prev_prev, C_set[0])
            else:
                self.preprocess0 = ReLUConvBN(C_prev_prev, C_set[0], 1, 1, 0)

            if reduction:
                self.preprocess1 = ReLUConvBN(C_prev, C_set[0], 1, 1, 0)  # This is kinda redundant now
            else:
                self.preprocess1 = ReLUConvBN(C_prev, C_set[1], 1, 1, 0)

            if reduction:
                op_names, indices = zip(*genotype.reduce)
                concat = genotype.reduce_concat
            else:
                op_names, indices = zip(*genotype.normal)
                concat = genotype.normal_concat
            self._compile_DARTS(C_set, sep_conv_set, op_names, indices, concat, reduction)

    def _compile_DARTS(self, C_set, sep_conv_set, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        # This is the final feature map size after concatenation from all intermediate nodes
        # Please be cautious of this in the future
        if reduction:
            self.concat_size = 4 * C_set[0]  # Final concat size is 4x C0 channel size
        else:
            self.concat_size = C_set[0] + C_set[0] + C_set[2] + C_set[3]  # This need to be determined analytically! Draw the genotype on a piece of paper

        self._ops = nn.ModuleList()

        if reduction:
            for name, index in zip(op_names, indices):
                stride = 2 if index < 2 else 1
                # reduction cell only has 1 channel value
                op = OPS[name](C_set[0], C_set[0], stride, True)
                # print(name, index)
                self._ops += [op]
        else:

            node_index = 0
            sep_conv_set_index = 0
            edge_count = 0

            # These are the required output channel for each node
            node_to_input_size = [C_set[2], C_set[3], C_set[0], C_set[0]]

            for name, index in zip(op_names, indices):

                if "sep_conv" in name:
                    op = OPS[name](C_set[index], sep_conv_set[sep_conv_set_index], node_to_input_size[node_index], 1,
                                   True)
                    sep_conv_set_index = sep_conv_set_index + 1
                else:
                    op = OPS[name](C_set[index], node_to_input_size[node_index], 1, True)
                edge_count = edge_count + 1
                # Every node has 2 input edges
                if edge_count == 2:
                    node_index = node_index + 1
                    edge_count = 0
                # print(name, index)
                self._ops += [op]

        self._indices = indices

    def _compile_DARTS_PLUS(self, C_set, sep_conv_set, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        # This is the final feature map size after concatenation from all intermediate nodes
        # Please be cautious of this in the future
        if reduction:
            self.concat_size = 4 * C_set[0]  # Final concat size is 4x C0 channel size
        else:
            self.concat_size = C_set[0] + C_set[0] + C_set[2] + C_set[3]  # This need to be determined analytically! Draw the genotype on a piece of paper

        self._ops = nn.ModuleList()

        if reduction:
            for name, index in zip(op_names, indices):
                stride = 2 if index < 2 else 1
                # reduction cell only has 2 channel values now! update!!
                if index == 1 and name == 'skip_connect':
                    # This becomes a factorized reduce!
                    op = OPS[name](C_set[1], C_set[0], stride, True)
                else:
                    op = OPS[name](C_set[0], C_set[0], stride, True)
                # print(name, index)
                self._ops += [op]
        else:

            node_index = 0
            sep_conv_set_index = 0
            edge_count = 0

            # These are the required output channel for each node
            node_to_input_size = [C_set[0], C_set[0], C_set[2], C_set[3]]

            # These are the required input channel size for each edge
            edge_to_input_size = [C_set[0], C_set[1], C_set[0], C_set[0], C_set[1], C_set[0], C_set[0], C_set[2]]

            for name, index in zip(op_names, indices):

                if "sep_conv" in name:
                    op = OPS[name](edge_to_input_size[edge_count], sep_conv_set[sep_conv_set_index],
                                   node_to_input_size[node_index], 1, True)
                    sep_conv_set_index = sep_conv_set_index + 1
                else:
                    op = OPS[name](edge_to_input_size[edge_count], node_to_input_size[node_index], 1, True)
                edge_count = edge_count + 1
                # Every node has 2 input edges
                if edge_count % 2 == 0:
                    node_index = node_index + 1
                    # edge_count = 0
                # print(name, index)
                self._ops += [op]

        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)



class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      # Need to adjust dimension at this point based on number of extra layers

      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C_list, sep_conv_list, num_classes, layers, auxiliary, genotype, arch):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        # stem_multiplier = 3
        C_curr = C_list[0][0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        C_prev_prev, C_prev = C_curr, C_curr
        self.cells = nn.ModuleList()
        reduction_prev = False

        # Only increment this for normal cells. Reduction cells do not have sep convs
        sep_conv_list_index = 0
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(arch, genotype, C_prev_prev, C_prev, C_list[i + 1], sep_conv_list[sep_conv_list_index], reduction,
                        reduction_prev)
            if not reduction:
                sep_conv_list_index = sep_conv_list_index + 1

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.concat_size
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

class NetworkImageNet(nn.Module):

  def __init__(self, C_list, sep_conv_list, num_classes, layers, auxiliary, genotype, arch):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    # ImageNet DARTS actually uses 2 stems, compared to CIFAR which only had 1
    # Use the 1 stem value from CIFAR for both stems for now
    C = C_list[0][0]

    self.stem0 = nn.Sequential(

      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),


      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    #C_prev_prev, C_prev, C_curr = C, C, C
    C_prev_prev, C_prev = C, C

    sep_conv_list_index = 0
    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        #C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(arch, genotype, C_prev_prev, C_prev, C_list[i + 1], sep_conv_list[sep_conv_list_index], reduction,
                  reduction_prev)

      if not reduction:
        sep_conv_list_index = sep_conv_list_index + 1

      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.concat_size
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)

    #Update from 7 -> 2. Final feature map 2x2
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux