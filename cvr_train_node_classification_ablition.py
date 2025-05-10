import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules_node_cover import MergeLayer, MLPClassifier
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler
from utils.cvr_evaluate_models_utils_ablition import cvr_evaluate_model_node_classification, find_threshold
from utils.metrics import get_node_classification_metrics
from utils.DataLoader import get_idx_data_loader, get_node_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_args_abl_beta
from utils.loss import SelectiveLoss, find_tres

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_classification_args_abl_beta()

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = \
        get_node_classification_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs, test_labels_stat_all_runs = [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.load_model_name = f'{args.model_name}_seed{args.seed}'
        args.save_model_name = f'node_classification_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/lr_{args.learning_rate}/cov_{args.coverage}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/lr_{args.learning_rate}/cov_{args.coverage}/ablition_lr_{args.learning_rate}_{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)

        # load the saved model in the link prediction task
        load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}"
        early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                       save_model_name=args.load_model_name, logger=logger, model_name=args.model_name)
        early_stopping.load_checkpoint(model, map_location='cpu')

        # create the model for the node classification task
        node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        model = nn.Sequential(model[0], node_classifier)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        # follow previous work, we freeze the dynamic_backbone and only optimize the node_classifier
        optimizer = create_optimizer(model=model[1], optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)
        # put the node raw messages of memory-based models on device
        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            for node_id, node_raw_messages in model[0].memory_bank.node_raw_messages.items():
                new_node_raw_messages = []
                for node_raw_message in node_raw_messages:
                    new_node_raw_messages.append((node_raw_message[0].to(args.device), node_raw_message[1]))
                model[0].memory_bank.node_raw_messages[node_id] = new_node_raw_messages

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/ablition/lr_{args.learning_rate}/cov_{args.coverage}/wgt_{args.cls_1_wgt}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        # loss_func = nn.BCELoss()
        # weights = torch.tensor([1.0, args.cls_1_wgt])  # weight for class 0 and class 1 respectively
        # loss_func = nn.BCEWithLogitsLoss(pos_weight=weights[1])

        sel_loss = SelectiveLoss(lambda_val=args.lambda_val)

        # set the dynamic_backbone in evaluation mode
        model[0].eval()

        for epoch in range(args.num_epochs):

            model[1].train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                # training process, set the neighbor sampler
                model[0].set_neighbor_sampler(full_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses, trues and predicts
            train_total_loss, train_y_trues, train_y_predicts, train_selective_pred = 0.0, [], [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], train_data.node_interact_times[train_data_indices], \
                    train_data.edge_ids[train_data_indices], train_data.labels[train_data_indices]

                with torch.no_grad():
                    if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=args.num_neighbors)
                    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              edge_ids=batch_edge_ids,
                                                                              edges_are_positive=True,
                                                                              num_neighbors=args.num_neighbors)
                    elif args.model_name in ['GraphMixer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=args.num_neighbors,
                                                                              time_gap=args.time_gap)
                    elif args.model_name in ['DyGFormer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times)
                    else:
                        raise ValueError(f"Wrong value for model_name {args.model_name}!")
                # get predicted probabilities, shape (batch_size, )
                # predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()
                predicts, sel, aux = model[1](x=batch_src_node_embeddings)
                predicts = predicts.sigmoid() ## commenting sigmoid and using loss as BCEWithLogitsLoss
                # print("predicts:", predicts[:20])
                # print("aux:", aux[:20])
                # print("*"*100)
                labels = torch.from_numpy(batch_labels).float().to(predicts.device)
                weights = torch.tensor([1.0, args.cls_1_wgt], device=predicts.device)  # weight for class 0 and class 1 respectively
                # sample_weights = weights[labels.data.view(-1).long()].view_as(labels)
                sample_weights = weights[labels.long()]
                # sample_weights = sample_weights.to(predicts.device)

                loss_func = nn.BCELoss(weight=sample_weights)
                # loss_func = nn.BCELoss()

                # loss = loss_func(input=predicts, target=labels)
                if args.coverage == 1.0:
                    loss = loss_func(input=predicts, target=labels)

                else:
                    loss_1 = sel_loss(input=predicts, selective_prob=sel, labels=labels, coverage=args.coverage) # Selective loss L_(f,g)
                    # loss_1 = selective_loss(predicts, sel, labels, args.coverage) # Selective loss L_(f,g)
                    loss_2 = loss_func(input=aux, target=labels) # Auxilary loss L_h
                
                    loss =(args.alphaloss * loss_1) + ((1-args.alphaloss) * loss_2)

                train_total_loss += loss.item()

                train_y_trues.append(labels)
                train_y_predicts.append(predicts)
                train_selective_pred.append(sel)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

            train_total_loss /= (batch_idx + 1)
            train_y_trues = torch.cat(train_y_trues, dim=0)
            train_y_predicts = torch.cat(train_y_predicts, dim=0)
            train_selective_pred = torch.cat(train_selective_pred, dim=0)

            t, _, _ = find_threshold(train_selective_pred, args.coverage)
            mask = train_selective_pred >= t
            train_selected_embeddings = train_y_predicts[mask]
            train_selected_labels = train_y_trues[mask]

            # train_metrics = get_node_classification_metrics(predicts=train_y_predicts, labels=train_y_trues)
            train_metrics = get_node_classification_metrics(predicts=train_selected_embeddings, labels=train_selected_labels)

            val_total_loss, val_metrics, _ = cvr_evaluate_model_node_classification(model_name=args.model_name,
                                                                             model=model,
                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                             evaluate_idx_data_loader=val_idx_data_loader,
                                                                             evaluate_data=val_data,
                                                                             loss_func=loss_func,
                                                                             sel_loss = sel_loss,
                                                                             coverage= args.coverage,
                                                                             alphaloss= args.alphaloss,
                                                                             cls_1_wgt = args.cls_1_wgt,
                                                                             num_neighbors=args.num_neighbors,
                                                                             time_gap=args.time_gap)

            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}')
            for metric_name in train_metrics.keys():
                logger.info(f'train {metric_name}, {train_metrics[metric_name]:.4f}')
            logger.info(f'validate loss: {val_total_loss:.4f}')
            for metric_name in val_metrics.keys():
                logger.info(f'validate {metric_name}, {val_metrics[metric_name]:.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                    val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                test_total_loss, test_metrics, test_labels_stat = cvr_evaluate_model_node_classification(model_name=args.model_name,
                                                                                   model=model,
                                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                                   evaluate_data=test_data,
                                                                                   loss_func=loss_func,
                                                                                   sel_loss = sel_loss,
                                                                                   coverage= args.coverage,
                                                                                   alphaloss= args.alphaloss,
                                                                                   cls_1_wgt=args.cls_1_wgt,
                                                                                   num_neighbors=args.num_neighbors,
                                                                                   time_gap=args.time_gap)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'test loss: {test_total_loss:.4f}')
                for metric_name in test_metrics.keys():
                    logger.info(f'test {metric_name}, {test_metrics[metric_name]:.4f}')
                for metric_name in test_labels_stat.keys():
                    logger.info(f'test {metric_name}, {test_labels_stat[metric_name]}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics.keys():
                val_metric_indicator.append((metric_name, val_metrics[metric_name], True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_total_loss, val_metrics, _ = cvr_evaluate_model_node_classification(model_name=args.model_name,
                                                                             model=model,
                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                             evaluate_idx_data_loader=val_idx_data_loader,
                                                                             evaluate_data=val_data,
                                                                             loss_func=loss_func,
                                                                             sel_loss = sel_loss,
                                                                             coverage= args.coverage,
                                                                             alphaloss= args.alphaloss,
                                                                             cls_1_wgt=args.cls_1_wgt,
                                                                             num_neighbors=args.num_neighbors,
                                                                             time_gap=args.time_gap)

        test_total_loss, test_metrics, test_labels_stat = cvr_evaluate_model_node_classification(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           sel_loss = sel_loss,
                                                                           coverage= args.coverage,
                                                                           alphaloss= args.alphaloss,
                                                                           cls_1_wgt=args.cls_1_wgt,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

        # store the evaluation metrics at the current run
    #     val_metric_dict, test_metric_dict = {}, {}

    #     if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
    #         logger.info(f'validate loss: {val_total_loss:.4f}')
    #         for metric_name in val_metrics.keys():
    #             val_metric = val_metrics[metric_name]
    #             logger.info(f'validate {metric_name}, {val_metric:.4f}')
    #             val_metric_dict[metric_name] = val_metric

    #     logger.info(f'test loss: {test_total_loss:.4f}')
    #     for metric_name in test_metrics.keys():
    #         test_metric = test_metrics[metric_name]
    #         logger.info(f'test {metric_name}, {test_metric:.4f}')
    #         test_metric_dict[metric_name] = test_metric

    #     single_run_time = time.time() - run_start_time
    #     logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

    #     if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
    #         val_metric_all_runs.append(val_metric_dict)
    #     test_metric_all_runs.append(test_metric_dict)

    #     # avoid the overlap of logs
    #     if run < args.num_runs - 1:
    #         logger.removeHandler(fh)
    #         logger.removeHandler(ch)

    #     # save model result
    #     if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
    #         result_json = {
    #             "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
    #             "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
    #             "test labels stat": test_labels_stat
    #         }
    #     else:
    #         result_json = {
    #             "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
    #             "test labels stat": test_labels_stat
    #         }
    #     result_json = json.dumps(result_json, indent=4)

    #     save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}/ablition/lr_{args.learning_rate}/cov_{args.coverage}"
    #     os.makedirs(save_result_folder, exist_ok=True)
    #     save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

    #     with open(save_result_path, 'w') as file:
    #         file.write(result_json)

    # # store the average metrics at the log of the last run
    # logger.info(f'metrics over {args.num_runs} runs:')

    # if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
    #     for metric_name in val_metric_all_runs[0].keys():
    #         logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
    #         logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
    #                     f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    # for metric_name in test_metric_all_runs[0].keys():
    #     logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
    #     logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
    #                 f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
    
    # # save the average metrics for all runs
    # if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
    #     val_metric_all_runs = {metric_name: list(val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs) for metric_name in val_metric_all_runs[0].keys()}
    # test_metric_all_runs = {metric_name: list(test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs) for metric_name in test_metric_all_runs[0].keys()}
    
    # result_json_all = {
    #     "validate metrics": val_metric_all_runs,
    #     "average validate metrics": {metric_name: f'{np.mean(val_metric_all_runs[metric_name]):.4f} ± {np.std(val_metric_all_runs[metric_name], ddof=1):.4f}' for metric_name in val_metric_all_runs},
    #     "test metrics": test_metric_all_runs,
    #     "average test metrics": {metric_name: f'{np.mean(test_metric_all_runs[metric_name]):.4f} ± {np.std(test_metric_all_runs[metric_name], ddof=1):.4f}' for metric_name in test_metric_all_runs},
    #     "test labels stat": test_labels_stat
    # }
    # result_json_all = json.dumps(result_json_all, indent=4)

    # save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}/ablition/lr_{args.learning_rate}/cov_{args.coverage}"
    # os.makedirs(save_result_folder, exist_ok=True)
    # save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}_all_{args.num_runs}_runs.json")

    # with open(save_result_path, 'w') as file:
    #     file.write(result_json_all)


    # sys.exit()
        val_metric_dict, test_metric_dict, test_labels_stat_dict = {}, {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {val_total_loss:.4f}')
            for metric_name in val_metrics.keys():
                val_metric = val_metrics[metric_name]
                logger.info(f'validate {metric_name}, {val_metric:.4f}')
                val_metric_dict[metric_name] = val_metric

        logger.info(f'test loss: {test_total_loss:.4f}')
        for metric_name in test_metrics.keys():
            test_metric = test_metrics[metric_name]
            logger.info(f'test {metric_name}, {test_metric:.4f}')
            test_metric_dict[metric_name] = test_metric

        for metric_name in test_labels_stat.keys():
            test_labels_stat_dict[metric_name] = test_labels_stat[metric_name]
            logger.info(f'test {metric_name}, {test_labels_stat[metric_name]}')



        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        test_labels_stat_all_runs.append(test_labels_stat_dict)


        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "test labels stat": test_labels_stat
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "test labels stat": test_labels_stat
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}/ablition/lr_{args.learning_rate}/cov_{args.coverage}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
        # logger.info(f'test labels stat {metric_name}, {[test_labels_stat_single_run[metric_name] for test_labels_stat_single_run in test_labels_stat_all_runs]}')
        # logger.info(f'average test labels stat {metric_name}, {np.mean([test_labels_stat_single_run[metric_name] for test_labels_stat_single_run in test_labels_stat_all_runs]):.4f} '
        #             f'± {np.std([test_labels_stat_single_run[metric_name] for test_labels_stat_single_run in test_labels_stat_all_runs], ddof=1):.4f}')
    
    # save the average metrics for all runs
    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        val_metric_all_runs = {metric_name: list(val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs) for metric_name in val_metric_all_runs[0].keys()}
    test_metric_all_runs = {metric_name: list(test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs) for metric_name in test_metric_all_runs[0].keys()}
    test_labels_stat_all_runs = {metric_name: list(test_labels_stat_single_run[metric_name] for test_labels_stat_single_run in test_labels_stat_all_runs) for metric_name in test_labels_stat_all_runs[0].keys()}

    
    result_json_all = {
        "validate metrics": val_metric_all_runs,
        "average validate metrics": {metric_name: f'{np.mean(val_metric_all_runs[metric_name]):.4f} ± {np.std(val_metric_all_runs[metric_name], ddof=1):.4f}' for metric_name in val_metric_all_runs},
        "test metrics": test_metric_all_runs,
        "average test metrics": {metric_name: f'{np.mean(test_metric_all_runs[metric_name]):.4f} ± {np.std(test_metric_all_runs[metric_name], ddof=1):.4f}' for metric_name in test_metric_all_runs},
        "test labels stat": test_labels_stat_all_runs
    }
    result_json_all = json.dumps(result_json_all, indent=4)

    save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}/ablition/lr_{args.learning_rate}/cov_{args.coverage}"
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}_all_{args.num_runs}_runs.json")

    with open(save_result_path, 'w') as file:
        file.write(result_json_all)


    sys.exit()