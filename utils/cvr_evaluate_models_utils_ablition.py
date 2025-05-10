import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from models.EdgeBank import edge_bank_link_prediction
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data
from utils.load_configs import get_node_classification_args_abl_beta
# args = get_node_classification_args_abl_beta()



def cvr_evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, loss_func: nn.Module, sel_loss: nn.Module, coverage: float, alphaloss: float, cls_1_wgt: float,
                                       num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        # pred_prob, tsne_embedding, evaluate_selective_pred = np.zeros((evaluate_data.num_nodes, 1)), np.zeros((evaluate_data.num_nodes, 2)), np.zeros((evaluate_data.num_nodes, 1))
        pred_prob, tsne_embedding, evaluate_selective_pred = [], [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get predicted probabilities, shape (batch_size, )
            # predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()
            predicts, sel, aux = model[1](x=batch_src_node_embeddings)
            predicts = predicts.sigmoid()
            # pred_prob.append(predicts.cpu().numpy())
            tsne_embedding.append(batch_src_node_embeddings.cpu().numpy())
            # evaluate_selective_pred.append(sel.cpu().numpy())

            labels = torch.from_numpy(batch_labels).float().to(predicts.device)
            weights = torch.tensor([1.0, cls_1_wgt], device=predicts.device)  # weight for class 0 and class 1 respectively
            # sample_weights = weights[labels.data.view(-1).long()].view_as(labels)
            sample_weights = weights[labels.long()]
            # sample_weights = sample_weights.to(predicts.device)

            loss_func_ = nn.BCELoss(weight=sample_weights)
            # loss = loss_func_(input=predicts, target=labels)
            if coverage == 1.0:
                loss = loss_func_(input=predicts, target=labels)

            else:
                loss_1 = sel_loss(input=predicts, selective_prob=sel, labels=labels, coverage=coverage) # Selective loss L_(f,g)
                loss_2 = loss_func_(input=aux, target=labels) # Auxilary loss L_h
                loss =(alphaloss * loss_1) + ((1-alphaloss) * loss_2)

            evaluate_total_loss += loss.item()

            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(predicts)
            evaluate_selective_pred.append(sel)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= (batch_idx + 1)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)
        evaluate_selective_pred = torch.cat(evaluate_selective_pred, dim=0)

        t, _, _ = find_threshold(evaluate_selective_pred, coverage)
        mask = evaluate_selective_pred >= t

        selected_embeddings = evaluate_y_predicts[mask]
        selected_labels = evaluate_y_trues[mask]
        print("*"*70)
        print("t: ", t)
        print("selected_labels = 0: ", len(selected_labels[selected_labels==0.0]))
        print("selected_labels = 1: ", len(selected_labels[selected_labels==1.0]))
        print("*"*70)
        print("evaluate_actual_labels = 0: ", len(evaluate_y_trues[evaluate_y_trues==0.0]))
        print("evaluate_actual_labels = 1: ", len(evaluate_y_trues[evaluate_y_trues==1.0]))
        print("*"*70)
        labels_stat = {"t": t, "selected_labels": {"0": len(selected_labels[selected_labels==0.0]), "1": len(selected_labels[selected_labels==1.0])}, 
                       "evaluate_actual_labels": {"0": len(evaluate_y_trues[evaluate_y_trues==0.0]), "1": len(evaluate_y_trues[evaluate_y_trues==1.0])}}

        # evaluate_metrics = get_node_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)
        evaluate_metrics = get_node_classification_metrics(predicts=selected_embeddings, labels=selected_labels)

    return evaluate_total_loss, evaluate_metrics, labels_stat

def find_threshold(selective_prob, coverage):
    if coverage == 1.0:
       t = -0.1
       sel_min = 0.0
       sel_max = 1.0
       
    else:
      sel = selective_prob
      sel = sel.tolist()
      sel_min = min(sel)
      sel_max = max(sel)
      sel.sort(reverse=True)
      c = int(coverage * len(sel))       
      t = sel[c]
    return t, sel_min, sel_max
