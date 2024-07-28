import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GRIP'))
#from main import my_load_model, compute_RMSE, display_result

import torch
import torch.optim as optim
import pickle
import time
import warnings
import itertools
import numpy as np
import logging
import copy
from datetime import datetime
import torch.nn as nn
from .dataloader import GRIPDataLoader
from prediction.model.base.interface import Interface


logger = logging.getLogger(__name__)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda=True):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.GRU(input_size, hidden_size * 30, num_layers, batch_first=True)

    def forward(self, input):
        output, hidden = self.lstm(input)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout=0.5, isCuda=True):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        # self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.lstm = nn.GRU(hidden_size, output_size * 30, num_layers, batch_first=True)

        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(output_size * 30, output_size)
        self.tanh = nn.Tanh()

    def forward(self, encoded_input, hidden):
        decoded_output, hidden = self.lstm(encoded_input, hidden)
        # decoded_output = self.tanh(decoded_output)
        # decoded_output = self.sigmoid(decoded_output)
        decoded_output = self.dropout(decoded_output)
        # decoded_output = self.tanh(self.linear(decoded_output))
        decoded_output = self.linear(decoded_output)
        # decoded_output = self.sigmoid(self.linear(decoded_output))
        return decoded_output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, isCuda=True):
        super(Seq2Seq, self).__init__()
        self.isCuda = isCuda
        # self.pred_length = pred_length
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, hidden_size, num_layers, dropout, isCuda)

    def forward(self, in_data, last_location, pred_length, teacher_forcing_ratio=0, teacher_location=None):
        batch_size = in_data.shape[0]
        out_dim = self.decoder.output_size
        self.pred_length = pred_length

        outputs = torch.zeros(batch_size, self.pred_length, out_dim)
        if self.isCuda:
            outputs = outputs.cuda()

        encoded_output, hidden = self.encoder(in_data)
        decoder_input = last_location
        for t in range(self.pred_length):
            # encoded_input = torch.cat((now_label, encoded_input), dim=-1) # merge class label into input feature
            now_out, hidden = self.decoder(decoder_input, hidden)
            now_out += decoder_input
            outputs[:, t:t + 1] = now_out
            teacher_force = np.random.random() < teacher_forcing_ratio
            decoder_input = (teacher_location[:, t:t + 1] if (type(teacher_location) is not type(
                None)) and teacher_force else now_out)
        # decoder_input = now_out
        return outputs
class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(1) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()

        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,nkvw->nctw', (x, A))

        return x.contiguous(), A


class Graph_Conv_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=False),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class Graph():
	""" The Graph Representation
	How to use:
		1. graph = Graph(max_hop=1)
		2. A = graph.get_adjacency()
		3. A = code to modify A
		4. normalized_A = graph.normalize_adjacency(A)
	"""
	def __init__(self,
				 num_node = 120,
				 max_hop = 1
				 ):
		self.max_hop = max_hop
		self.num_node = num_node

	def get_adjacency(self, A):
		# compute hop steps
		self.hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
		transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
		arrive_mat = (np.stack(transfer_mat) > 0)
		for d in range(self.max_hop, -1, -1):
			self.hop_dis[arrive_mat[d]] = d

		# compute adjacency
		valid_hop = range(0, self.max_hop + 1)
		adjacency = np.zeros((self.num_node, self.num_node))
		for hop in valid_hop:
			adjacency[self.hop_dis == hop] = 1
		return adjacency

	def normalize_adjacency(self, A):
		Dl = np.sum(A, 0)
		num_node = A.shape[0]
		Dn = np.zeros((num_node, num_node))
		for i in range(num_node):
			if Dl[i] > 0:
				Dn[i, i] = Dl[i]**(-1)
		AD = np.dot(A, Dn)

		valid_hop = range(0, self.max_hop + 1)
		A = np.zeros((len(valid_hop), self.num_node, self.num_node))
		for i, hop in enumerate(valid_hop):
			A[i][self.hop_dis == hop] = AD[self.hop_dis == hop]
		return A


class Model(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = np.ones((graph_args['max_hop'] + 1, graph_args['num_node'], graph_args['num_node']))

        # build networks
        spatial_kernel_size = np.shape(A)[0]
        temporal_kernel_size = 5  # 9 #5 # 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # best
        self.st_gcn_networks = nn.ModuleList((
            nn.BatchNorm2d(in_channels),
            Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
            Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
            Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(np.shape(A))) for i in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.num_node = num_node = self.graph.num_node
        self.out_dim_per_node = out_dim_per_node = 2  # (x, y) coordinate
        self.seq2seq_car = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5,
                                   isCuda=True)
        self.seq2seq_human = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5,
                                     isCuda=True)
        self.seq2seq_bike = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5,
                                    isCuda=True)

    def reshape_for_lstm(self, feature):
        # prepare for skeleton prediction model
        '''
        N: batch_size
        C: channel
        T: time_step
        V: nodes
        '''
        N, C, T, V = feature.size()
        now_feat = feature.permute(0, 3, 2, 1).contiguous()  # to (N, V, T, C)
        now_feat = now_feat.view(N * V, T, C)
        return now_feat

    def reshape_from_lstm(self, predicted):
        # predicted (N*V, T, C)
        NV, T, C = predicted.size()
        now_feat = predicted.view(-1, self.num_node, T,
                                  self.out_dim_per_node)  # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
        now_feat = now_feat.permute(0, 3, 2, 1).contiguous()  # (N, C, T, V)
        return now_feat

    def forward(self, pra_x, pra_A, pra_pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
        x = pra_x

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            if type(gcn) is nn.BatchNorm2d:
                x = gcn(x)
            else:
                x, _ = gcn(x, pra_A + importance)

        # prepare for seq2seq lstm model
        graph_conv_feature = self.reshape_for_lstm(x)
        last_position = self.reshape_for_lstm(pra_x[:, :2])  # (N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, C)]

        if pra_teacher_forcing_ratio > 0 and type(pra_teacher_location) is not type(None):
            pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

        # now_predict.shape = (N, T, V*C)
        now_predict_car = self.seq2seq_car(in_data=graph_conv_feature, last_location=last_position[:, -1:, :],
                                           pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio,
                                           teacher_location=pra_teacher_location)
        now_predict_car = self.reshape_from_lstm(now_predict_car)  # (N, C, T, V)

        now_predict_human = self.seq2seq_human(in_data=graph_conv_feature, last_location=last_position[:, -1:, :],
                                               pred_length=pra_pred_length,
                                               teacher_forcing_ratio=pra_teacher_forcing_ratio,
                                               teacher_location=pra_teacher_location)
        now_predict_human = self.reshape_from_lstm(now_predict_human)  # (N, C, T, V)

        now_predict_bike = self.seq2seq_bike(in_data=graph_conv_feature, last_location=last_position[:, -1:, :],
                                             pred_length=pra_pred_length,
                                             teacher_forcing_ratio=pra_teacher_forcing_ratio,
                                             teacher_location=pra_teacher_location)
        now_predict_bike = self.reshape_from_lstm(now_predict_bike)  # (N, C, T, V)

        now_predict = (now_predict_car + now_predict_human + now_predict_bike) / 3.

        return now_predict
class GRIPInterface(Interface):
    def __init__(self, obs_length, pred_length, pre_load_model=None, max_hop=2, num_node=120, in_channels=4, rescale=[1,1], smooth=0, dataset=None):
        super().__init__(obs_length, pred_length)
        self.graph_args = {'max_hop':max_hop, 'num_node':num_node}
        self.dataloader = GRIPDataLoader(
            self.obs_length, self.pred_length, graph_args=self.graph_args, dataset=dataset
        )

        self.dev = 'cuda:0'
        if pre_load_model is not None:
            self.model = self.load_model(self.default_model(in_channels=in_channels), pre_load_model)
        else:
            self.model = None

        self.rescale = rescale
        self.smooth = smooth
        self.dataset = dataset

    def default_model(self, in_channels=4):
        model = Model(in_channels=in_channels, graph_args=self.graph_args, edge_importance_weighting=True)
        model.to(self.dev)
        return model

    def load_model(self, model, model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
        logger.warn('Successfull loaded from {}'.format(model_path))
        return model

    def save_model(self, model, model_path):
        torch.save(
            {
                'xin_graph_seq2seq_model': model.state_dict(),
            }, 
            model_path)
        logger.warn("Model saved to {}".format(model_path))

    def run(self, input_data, perturbation=None, backward=False):
        assert(self.model is not None)

        # if not backward:
        #     self.model.eval()

        _input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, output_loc_GT, output_mask, obj_index = self.dataloader.preprocess(input_data, perturbation, smooth=self.smooth, rescale_x=self.rescale[0], rescale_y=self.rescale[1])
        predicted = self.model(pra_x=_input_data, pra_A=A, pra_pred_length=self.pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)
        output_data, loss = self.dataloader.postprocess(input_data, perturbation, predicted, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, obj_index)

        if loss is None:
            return output_data
        else:
            return output_data, loss
