import os, sys
import random
import logging
import copy
import torch
import argparse
torch.backends.cudnn.enabled = False
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os
from prediction.dataset.generate import input_data_by_attack_step
import glob
import json
from count_distance import  *
from loss import attack_loss
from prediction.dataset.apolloscape import ApolloscapeDataset
from prediction.dataset.ngsim import NGSIMDataset
from prediction.dataset.nuscenes import NuScenesDataset
from prediction.dataset.generate import data_offline_generator
from prediction.attack.gradient import GradientAttacker
from prediction.attack.pso import PSOAttacker
from test_utils import *
from config import datasets, models

import warnings
warnings.filterwarnings("ignore")

for dataset_name in datasets:
    samples_file = os.path.join(datasets[dataset_name]["data_dir"], "samples.txt")
    with open(samples_file, 'r') as f:
        lines = f.readlines()
    datasets[dataset_name]["samples"] = [(int(line.split(' ')[0]), int(line.split(' ')[1])) for line in lines]

import numpy as np
import torch
import copy


def get_trace(obj, name):
    if "{}_trace".format(name) not in obj or "{}_mask".format(name) not in obj:
        return None
    trace = obj["{}_trace".format(name)]
    mask = obj["{}_mask".format(name)]
    indexes = np.argwhere(mask > 0)
    if indexes.shape[0] == 0:
        return None
    else:
        return trace[np.concatenate(indexes), :]


def get_unit_vector(vectors):
    scale = np.sum(vectors ** 2, axis=1) ** 0.5 + 0.001
    result = np.zeros(vectors.shape)
    result[:, 0] = vectors[:, 0] / scale
    result[:, 1] = vectors[:, 1] / scale
    return result


def get_metrics(trace_array):
    v = (trace_array[1:, :] - trace_array[:-1, :])*3
    a = v[1:, :] - v[:-1, :]
    aa = a[1:, :] - a[:-1, :]

    direction = get_unit_vector(v)
    direction_r = np.concatenate((direction[:, 1].reshape(direction.shape[0], 1),
                                  -direction[:, 0].reshape(direction.shape[0], 1)), axis=1)

    scalar_v = np.sum(v ** 2, axis=1) ** 0.5

    linear_a = np.absolute(np.sum(direction[:-1, :] * a, axis=1))
    rotate_a = np.absolute(np.sum(direction_r[:-1, :] * a, axis=1))

    linear_aa = np.absolute(np.sum(direction[:-2, :] * aa, axis=1))
    rotate_aa = np.absolute(np.sum(direction_r[:-2, :] * aa, axis=1))

    return scalar_v, linear_a, rotate_a, linear_aa, rotate_aa


def get_deviation(perturbation_array):
    return np.sum(perturbation_array ** 2, axis=1) ** 0.5


def hard_constraint(observe_trace_array, perturbation_tensor, hard_bound, physical_bounds):
    if not isinstance(perturbation_tensor, np.ndarray):
        perturbation_array = np.array(perturbation_tensor)
    else:
        perturbation_array = perturbation_tensor

    step = 0.01
    theta = 1 + step
    check_pass = False
    while not check_pass:
        theta -= 0.01
        if theta <= 0.01:
            break
        merged_trace_array = copy.deepcopy(observe_trace_array)
        merged_trace_array[:perturbation_array.shape[0], :] += theta * perturbation_array
        scalar_v, linear_a, rotate_a, linear_aa, rotate_aa = get_metrics(merged_trace_array)
        deviation = get_deviation(theta * perturbation_array)
        check_pass = (np.sum(scalar_v > physical_bounds["scalar_v"]) == 0 and np.sum(
            linear_a > physical_bounds["linear_a"]) == 0 and np.sum(
            rotate_a > physical_bounds["rotate_a"]) == 0 and np.sum(
            linear_aa > physical_bounds["linear_aa"]) == 0 and np.sum(
            rotate_aa > physical_bounds["rotate_aa"]) == 0 and np.sum(deviation > hard_bound) == 0)
    return np.array(perturbation_tensor) * theta


def get_physical_constraints(data_generator):
    max_scalar_v = 0
    max_rotate_a = 0
    max_linear_a = 0
    max_rotate_aa = 0
    max_linear_aa = 0

    for input_data in data_generator:
        for _, obj in input_data["objects"].items():
            if obj["type"] not in [1, 2]:
                continue

            observe_trace = get_trace(obj, "observe")
            future_trace = get_trace(obj, "future")
            predict_trace = get_trace(obj, "predict")

            # update boundaries
            trace_all = observe_trace
            if future_trace is not None:
                trace_all = np.vstack((trace_all, future_trace))

            if trace_all.shape[0] < 4:
                continue

            scalar_v, linear_a, rotate_a, linear_aa, rotate_aa = get_metrics(trace_all)

            max_scalar_v = max(max_scalar_v, np.max(scalar_v))
            max_linear_a = max(max_linear_a, np.max(linear_a))
            max_rotate_a = max(max_rotate_a, np.max(rotate_a))
            max_linear_aa = max(max_linear_aa, np.max(linear_aa))
            max_rotate_aa = max(max_rotate_aa, np.max(rotate_aa))

    return max_scalar_v, max_linear_a, max_rotate_a, max_linear_aa, max_rotate_aa


def load_model(model_name, dataset_name, augment=False, smooth=0, models=models):
    if model_name == "grip":
        from prediction.model.GRIP.interface import GRIPInterface
        api_class = GRIPInterface
    elif model_name == "fqa":
        from prediction.model.FQA.interface import FQAInterface
        api_class = FQAInterface
    elif model_name == "trajectron" or model_name == "trajectron_map":
        from prediction.model.Trajectron.interface import TrajectronInterface
        api_class = TrajectronInterface

    model_config = copy.deepcopy(models)
    model_config[model_name][dataset_name]["dataset"] = datasets[dataset_name]["instance"]
    if augment and not smooth:
        model_config[model_name][dataset_name]["pre_load_model"] = model_config[model_name][dataset_name]["pre_load_model"].replace("/original", "/augment")
    if smooth and not augment:
        if smooth == 1:
            model_config[model_name][dataset_name]["pre_load_model"] = model_config[model_name][dataset_name]["pre_load_model"].replace("/original", "/smooth")
        model_config[model_name][dataset_name]["smooth"] = smooth
    if smooth and augment:
        model_config[model_name][dataset_name]["pre_load_model"] = model_config[model_name][dataset_name]["pre_load_model"].replace("/original", "/augment_smooth")
        model_config[model_name][dataset_name]["smooth"] = True

    return api_class(
        datasets[dataset_name]["obs_length"],
        datasets[dataset_name]["pred_length"],
        **model_config[model_name][dataset_name]
    )

def get_tag(augment=False, smooth=0, blackbox=False):
    if augment and smooth:
        return "augment_smooth"
    elif augment:
        return "augment"
    elif smooth > 0:
        return "smooth" if smooth == 1 else "smooth"+str(smooth)
    elif blackbox:
        return "blackbox"
    else:
        return "original"
import torch

def ade(predict_trace, future_trace):
    predict_trace = torch.tensor(predict_trace)
    future_trace = torch.tensor(future_trace)
    return torch.sqrt(torch.sum(torch.square(predict_trace - future_trace)) / predict_trace.shape[0])
    #return (torch.sum(torch.square(predict_trace - future_trace)) / predict_trace.shape[0])


def fde(predict_trace, future_trace):
    predict_trace = torch.tensor(predict_trace)
    future_trace = torch.tensor(future_trace)
    return torch.sqrt(torch.sum(torch.square(predict_trace[-1,:] - future_trace[-1,:])))

def t_fde(predict_trace, target_trace):
    return torch.sqrt(torch.sum(torch.square(predict_trace[-1,:] - target_trace[-1,:])))

def perturbation_cost(perturbation):
    return torch.sum(torch.sqrt(torch.square(torch.absolute(perturbation)+1)))


def physical_constraint(observe_trace):
    v = observe_trace[1:,:] - observe_trace[:-1,:]
    dif = v[1:,:] - v[:-1,:]
    return torch.sum(torch.square(dif))


def perturbation_physical_constraint(observe_trace, perturbed_trace):
    return physical_constraint(perturbed_trace) - physical_constraint(observe_trace)


def interpolation(trace, inject_num=3):
    extended_trace = torch.zeros((trace.shape[0] -1) * inject_num + trace.shape[0], 2).cuda()
    for i in range(extended_trace.shape[0]):
        if i % (inject_num + 1) == 0:
            index = i // (inject_num + 1)
            extended_trace[i,:] = trace[index,:]
        else:
            start_index = i // (inject_num + 1)
            end_index = start_index + 1
            extended_trace[i,:] = (trace[end_index,:] - trace[start_index,:]) / (inject_num + 1) * (i - start_index * (inject_num + 1)) + trace[start_index,:]
    return extended_trace


def square_distance(point1, point2):
    return torch.sum(torch.square(point1 - point2))


def change_lane_attack_goal(predict_traces, future_traces, obj_id, **attack_opts):
    attacker_predict_trace = predict_traces[obj_id]
    # attacker_future_trace = future_traces[obj_id]
    victim_predict_trace = predict_traces[attack_opts["target_obj_id"]]
    extended_attacker_predict_trace = interpolation(attacker_predict_trace)
    extended_victim_predict_trace = interpolation(victim_predict_trace)
    # extended_attacker_future_trace = interpolation(attacker_future_trace)

    distance1 = torch.min(torch.sum(torch.square(extended_attacker_predict_trace - extended_victim_predict_trace), 1))
    distance2 = torch.min(torch.cdist(extended_attacker_predict_trace, extended_victim_predict_trace, p=2))
    return distance1 + distance2


def horizonal_distance(observe_trace, predict_trace, future_trace):
    predict_trace = torch.tensor(predict_trace)
    future_trace = torch.tensor(future_trace)
    observe_trace = torch.tensor(observe_trace)
    offset = predict_trace - future_trace
    direction = (future_trace -
                 torch.cat(
                   (torch.reshape(observe_trace[-1,:], (1,2)),
                    future_trace[:-1,:]), 0)).float()
    scale = torch.sqrt(torch.sum(torch.square(direction), 1)).float()
    right_direction = torch.matmul(
                        torch.tensor([[0., 1.], [-1., 0.]]).float(),
                        direction.t().float() / scale).t()
    average_distance = torch.sum(offset * right_direction) / predict_trace.shape[0]
    return average_distance


def vertical_distance(observe_trace, predict_trace, future_trace):
    predict_trace = torch.tensor(predict_trace)
    future_trace = torch.tensor(future_trace)
    observe_trace = torch.tensor(observe_trace)
    offset = predict_trace - future_trace
    direction = (future_trace -
                 torch.cat(
                   (torch.reshape(observe_trace[-1,:], (1,2)),
                    future_trace[:-1,:]), 0)).float()
    scale = torch.sqrt(torch.sum(torch.square(direction), 1)).float()
    average_distance = torch.sum(offset * (direction.t().float() / scale).t()) / predict_trace.shape[0]
    return average_distance


def attack_loss1(observe_traces, future_traces, predict_traces,perturbation, **attack_opts):
    if "perturbation_cost_c" not in attack_opts:
        attack_opts["perturbation_cost_c"] = 0.1
    if "physical_constraint_c" not in attack_opts:
        attack_opts["physical_constraint_c"] = 0.1
    if "attack_goal_c" not in attack_opts:
        attack_opts["attack_goal_c"] = 1


    # attacker_observe_trace = observe_traces[obj_id]
    # attacker_perturbed_trace = attacker_observe_trace + perturbation
    # loss = attack_opts["perturbation_cost_c"] * perturbation_cost(perturbation) + attack_opts["physical_constraint_c"] * perturbation_physical_constraint(attacker_observe_trace, attacker_perturbed_trace)
    loss = 0

    if "type" in attack_opts:
        attack_goal = attack_opts["type"]
        if attack_goal == "ade":
            loss -= attack_opts["attack_goal_c"] * ade(predict_traces, future_traces)
        elif attack_goal == "fde":
            loss -= attack_opts["attack_goal_c"] * fde(predict_traces, future_traces)
        elif attack_goal == "left":
            loss += attack_opts["attack_goal_c"] * horizonal_distance(observe_traces, predict_traces, future_traces)
        elif attack_goal == "right":
            loss -= attack_opts["attack_goal_c"] * horizonal_distance(observe_traces, predict_traces, future_traces)
        elif attack_goal == "front":
            loss -= attack_opts["attack_goal_c"] * vertical_distance(observe_traces, predict_traces, future_traces)
        elif attack_goal == "rear":
            loss += attack_opts["attack_goal_c"] * vertical_distance(observe_traces, predict_traces, future_traces)
        else:
            raise NotImplementedError()

    return loss


def attack(model_name, dataset_name, overwrite=0, mode="single_frame", augment=False, smooth=0, blackbox=False):
    api = load_model(model_name, dataset_name, augment=augment, smooth=smooth)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    samples = datasets[dataset_name]["samples"]
    attack_length = datasets[dataset_name]["attack_length"] if mode.endswith("multi_frame") else 1
    physical_bounds = datasets[dataset_name]["instance"].bounds
    tag = get_tag(augment=augment, smooth=smooth, blackbox=blackbox)

    if not blackbox:
        attacker = GradientAttacker(api.obs_length, api.pred_length, attack_length, api, seed_num=10, iter_num=100, physical_bounds=physical_bounds, bound=1, learn_rate=0.1)
    else:
        attacker = PSOAttacker(api.obs_length, api.pred_length, attack_length, api, physical_bounds=physical_bounds)

    datadir = "data/{}_{}/{}/attack/{}".format(model_name, dataset_name, mode, tag)
    adv_attack(attacker, "data/dataset/{}/multi_frame/raw".format(dataset_name, mode),
                        "{}/raw".format(datadir),
                        "{}/visualize".format(datadir), 
                        overwrite=overwrite, samples=samples)
# single_frame  multi_frame

def find_trace(dataset_dir,name,obj):
    input_data = load_data(os.path.join(dataset_dir, "{}.json".format(name)))
    return input_data["objects"][obj]["observe_trace"]


def normal(model_name, target,dataset_name, overwrite=0, mode="single_frame", augment=False, smooth=0):
    api = load_model(target, dataset_name, augment=augment, smooth=smooth)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    samples = datasets[dataset_name]["samples"]
    attack_length = datasets[dataset_name]["attack_length"] if mode.endswith("multi_frame") else 1
    tag = get_tag(augment=augment, smooth=smooth, blackbox=False)

    datadir = "data/{}_{}/{}/attack/{}/raw".format(model_name, dataset_name, mode, tag)
    dataset_dir = "data/dataset/apolloscape/multi_frame/raw"
    bounds = {
        "scalar_v": 10.539,
        "linear_a": 4.957,
        "rotate_a": 0.956,
        "linear_aa": 8.418,
        "rotate_aa": 1.577
    }
    ratio_list = [0.0,0.0,0.0,0.0,0.0,0.0]
    for name, obj_id in samples:
        #logging.warn("Log {} {}".format(name, obj_id))
        attack_goals = ["ade"]
        for attack_goal in attack_goals:
            file_paths = glob.glob(os.path.join(datadir, '*'))

            for file_path in file_paths:
                # 提取文件名
                filename = os.path.basename(file_path)
                if filename==f"{name}-{obj_id}-{attack_goal}.json":

                    json_path = datadir+r"/"+filename
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        objects = data["output_data"]["0"]
                        obj_id = str(obj_id)
                        future_trace = data["output_data"]["0"]["objects"][obj_id]["future_trace"]
                        observe_trace = data["output_data"]["0"]["objects"][obj_id]["observe_trace"]
                        attack_opts = data["attack_opts"]

                        perturbation = {"obj_id": obj_id, "loss": attack_loss, "value": {}, "ready_value": data["perturbation"],
                                        "attack_opts": attack_opts}
                        # 数据集扰动需要缩放  tra 0.6 grip 0.8 fqa 1，保证扰动大小在同一范围内
                        data["perturbation"][obj_id] = np.array(data["perturbation"][obj_id])
                        output_data = api.run(objects, perturbation=perturbation, backward=False)
                        perdicted_trace = output_data[0]["objects"][obj_id]["predict_trace"]

                        if attack_goal=="ade":
                            loss1 = ade( perdicted_trace,future_trace)
                            loss2 = fde( perdicted_trace,future_trace)
                            ratio_list[0] += float(loss1)
                            ratio_list[1] += float(loss2)

                        else:

                            loss1 = attack_loss1(observe_trace, future_trace, perdicted_trace, perturbation=perturbation,
                                             type=attack_goal)
                            if attack_goal == "left":
                                ratio_list[2] += float(loss1)
                            if attack_goal == "right":
                                ratio_list[3] += float(loss1)
                            if attack_goal == "front":
                                ratio_list[4] += float(loss1)
                            if attack_goal == "rear":
                                ratio_list[5] += float(loss1)




    with open("ratio.txt",mode="a") as f:
        for i in ratio_list:
            f.write(str(i/300)+" ")



def evaluate(model_name=None, dataset_name=None, overwrite=0, mode="single_frame", augment=False, smooth=0, blackbox=False):
    if model_name is None:
        model_list = list(models.keys())
    else:
        model_list = [model_name]

    if dataset_name is None:
        dataset_list = list(datasets.keys())
    else:
        dataset_list = [dataset_name]

    tag = get_tag(augment=augment, smooth=smooth, blackbox=blackbox)
    
    for model_name in model_list:
        for dataset_name in dataset_list:
            
            if model_name == "trajectron_map" and dataset_name in ["apolloscape", "ngsim"]:
                continue
            attack_length = datasets[dataset_name]["attack_length"] if mode.endswith("multi_frame") else 1
            samples = datasets[dataset_name]["samples"]
            if mode.startswith("normal"):
                datadir = "data/{}_{}/{}/normal/{}".format(model_name, dataset_name, mode[7:], tag)
                evaluate_loss("{}/raw".format(datadir), samples=samples, output_dir="{}/evaluate".format(datadir), normal_data=True, attack_length=attack_length)
            elif mode.startswith("transfer"):
                for other_model_name in models:
                    if other_model_name == model_name:
                        continue
                    datadir = "data/{}_{}/{}/transfer/{}".format(model_name, dataset_name, mode[9:], other_model_name)
                    evaluate_loss("{}/raw".format(datadir), samples=samples, output_dir="{}/evaluate".format(datadir), normal_data=False, attack_length=attack_length)
            else:
                datadir = "data/{}_{}/{}/attack/{}".format(model_name, dataset_name, mode, tag)
                evaluate_loss("{}/raw".format(datadir), samples=samples, output_dir="{}/evaluate".format(datadir), normal_data=False, attack_length=attack_length)



def main():

    parser = argparse.ArgumentParser(description='Testing script for prediction attacks.')
    parser.add_argument("--dataset", type=str, default="apolloscape", help="Name of dataset [apolloscape, ngsim, nuscenes]")
    parser.add_argument("--model", type=str, default="fqa", help="Name of model [grip, fqa, trajectron, trajectron_map]")
    parser.add_argument("--target", type=str, default="trajectron",
                        help="Name of model [grip, fqa, trajectron, trajectron_map]")
    parser.add_argument("--mode", type=str, default="single_frame", help="Prediction mode [single_frame, multi_frame]")
    parser.add_argument("--augment", action="store_true", default=False, help="Enable data augmentation")
    parser.add_argument("--smooth", type=int, default=0, help="Enable trajectory smoothing -- 0: no smoothing; 1: train-time smoothing; 2: test-time smoothing; 3: test-time smoothing with anomaly detection")
    parser.add_argument("--blackbox", action="store_true", default=False, help="Use blackbox attack instead of whitebox")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing data")
    args = parser.parse_args()


    normal(dataset_name=args.dataset,model_name=args.model, target=args.target,mode=args.mode, augment=args.augment, smooth=args.smooth, overwrite=args.overwrite)
    #evaluate(dataset_name=args.dataset, model_name=args.model, mode="normal_"+args.mode, augment=args.augment, smooth=args.smooth, overwrite=args.overwrite)
    #evaluate(dataset_name=args.dataset, model_name=args.model, mode=args.mode, augment=args.augment, smooth=args.smooth, blackbox=args.blackbox, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
