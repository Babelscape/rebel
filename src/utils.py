
import math
from typing import Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
from transformers.models.bart.modeling_bart import shift_tokens_right


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

class Trilinear(nn.Module):
    r"""Applies a bilinear transformation to the incoming data:
    :math:`y = x_1^T A x_2 + b`

    Args:
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        in3_features: size of each third input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}` and
          :math:`*` means any number of additional dimensions. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`.
        - Output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = Trilinear(20, 30, 40, 50)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> input3 = torch.randn(128, 40)
        >>> output = m(input1, input2, input3)
        >>> print(output.size())
        torch.Size([128, 50])
    """
    __constants__ = ['in1_features', 'in2_features', 'in3_features', 'out_features']
    in1_features: int
    in2_features: int
    in3_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in1_features: int, in2_features: int, in3_features: int, out_features: int, bias: bool = True) -> None:
        super(Trilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.in3_features = in3_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in1_features, in2_features, in3_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor, input3: Tensor) -> Tensor:
        if self.bias is not None:
            return torch.einsum('bn,bm,bo,anmo->ba', input1, input2, input3, self.weight) + self.bias
        return torch.einsum('bn,bm,bo,anmo->ba', input1, input2, input3, self.weight)
    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, in3_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.in2_features, self.out_features, self.bias is not None
        )

# class BartTripletHead(nn.Module):
#     """Head for sentence-level classification tasks."""

#     def __init__(
#         self,
#         input_dim: int,
#         inner_dim: int,
#         num_classes: int,
#         pooler_dropout: float,
#     ):
#         super().__init__()
#         self.dense = Trilinear(input_dim, input_dim, input_dim, inner_dim)
#         self.dropout = nn.Dropout(p=pooler_dropout)
#         self.out_proj = nn.Linear(inner_dim, num_classes)

#     def forward(self, head_states: torch.Tensor, tail_states: torch.Tensor, context_states: torch.Tensor):
#         head_states = self.dropout(head_states)
#         tail_states = self.dropout(tail_states)
#         context_states = self.dropout(context_states)
#         hidden_states = self.dense(head_states, tail_states, context_states)
#         hidden_states = torch.tanh(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.out_proj(hidden_states)
#         return hidden_states

# class BartTripletHead(nn.Module):
#     """Head for sentence-level classification tasks."""

#     def __init__(
#         self,
#         input_dim: int,
#         inner_dim: int,
#         num_classes: int,
#         pooler_dropout: float,
#     ):
#         super().__init__()
#         self.dense_head_tail = nn.Bilinear(input_dim, input_dim, inner_dim)
#         self.dense_rel_ctxt = nn.Bilinear(input_dim, inner_dim, inner_dim)

#         self.dropout = nn.Dropout(p=pooler_dropout)
#         self.out_proj = nn.Linear(inner_dim, num_classes)

#     def forward(self, head_states: torch.Tensor, tail_states: torch.Tensor, context_states: torch.Tensor):
#         head_states = self.dropout(head_states)
#         tail_states = self.dropout(tail_states)
#         context_states = self.dropout(context_states)
#         hidden_states = self.dense_head_tail(head_states, tail_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.dense_rel_ctxt(context_states, hidden_states)
#         hidden_states = torch.tanh(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.out_proj(hidden_states)
#         return hidden_states

class BartTripletHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense_head_tail_ctxt = nn.Linear(input_dim*3, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, head_states: torch.Tensor, tail_states: torch.Tensor, context_states: torch.Tensor):
        combined_state = torch.cat((head_states, tail_states, context_states), dim = 1)
        combined_state = self.dropout(combined_state)
        hidden_states = self.dense_head_tail_ctxt(combined_state)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

def shift_tokens_left(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    shifted_input_ids[:, -1] = pad_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."

    return shifted_input_ids

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

def extract_triplets_typed(text, mapping_types= {'<peop>': 'Peop', '<org>': 'Org', '<other>': 'Other', '<loc>': 'Loc'}):
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '','','','',''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                relation = ''
            subject = ''
        elif token in mapping_types:
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                object_ = ''
                subject_type = mapping_types[token]
            else:
                current = 'o'
                object_type = mapping_types[token]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
    return triplets