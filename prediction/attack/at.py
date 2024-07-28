
import torch
import logging
from torch.autograd import Variable

import torch.nn as nn
from .loss import attack_loss
from .constraint import hard_constraint
from prediction.dataset.generate import input_data_by_attack_step

logger = logging.getLogger(__name__)

import os
import json
import copy
import numpy as np

import random


def random_subset(lst, percentage):
    if percentage > 100 or percentage < 0:
        raise ValueError("Percentage should be between 0 and 100")
    k = int(len(lst) * percentage / 100)
    return random.sample(lst, k)
def get_dict_values(data):
    stack = [(data, [])]
    while len(stack) > 0:
        (d, k) = stack.pop()
        if isinstance(d, dict):
            for key in d:
                if not isinstance(d[key], dict):
                    yield d, key
                else:
                    stack.append((d[key], k + [key]))
        else:
            yield k
def json_to_data(json_data):
    data = copy.deepcopy(json_data)
    for d, k in get_dict_values(data):
        if isinstance(d[k], list):
            d[k] = np.array(d[k])
    return data
def load_data(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
    return json_data


import os
import numpy as np
import json
import copy


def add_flags(data):
    delete_obj_ids = []
    for obj_id, obj in data["objects"].items():
        obj["observe_mask"] = (obj["observe_trace"][:, 0] > 0).astype(np.int64)
        obj["future_mask"] = (obj["future_trace"][:, 0] > 0).astype(np.int64)

        if np.sum(obj["observe_mask"]) == 0:
            delete_obj_ids.append(obj_id)
            continue

        if np.min(np.concatenate((obj["observe_mask"], obj["future_mask"]), axis=0)) <= 0:
            obj["complete"] = False
        else:
            obj["complete"] = True

        if np.min(obj["observe_mask"][-1]) <= 0:
            obj["visible"] = False
        else:
            obj["visible"] = True

        obj["static"] = False
        trace = obj["observe_trace"][obj["observe_mask"] > 0, :]
        if trace.shape[0] > 1:
            v = trace[1:, :] - trace[:-1, :]
            v = np.sum(v ** 2, axis=1)
            if np.sum(v > 0) < v.shape[0]:
                obj["static"] = True

    for obj_id in delete_obj_ids:
        del data["objects"][obj_id]

    return data

def input_data_by_attack_step(data, obs_length, pred_length, attack_step):
    input_data = {"objects": {}}
    for key, value in data.items():
        if key != "objects":
            input_data[key] = value
    input_data["observe_length"] = obs_length
    input_data["predict_length"] = pred_length

    k = attack_step
    for _obj_id, obj in data["objects"].items():
        feature = np.array(obj["observe_feature"])
        observe_feature = copy.deepcopy(feature[k:k + obs_length, :])
        future_feature = copy.deepcopy(feature[k + obs_length:k + obs_length + pred_length, :])

        trace = np.array(obj["observe_trace"])
        observe_trace = copy.deepcopy(trace[k:k + obs_length, :])
        future_trace = copy.deepcopy(trace[k + obs_length:k + obs_length + pred_length, :])
        new_obj = {
            "type": int(obj["type"]),
            "observe_feature": observe_feature,
            "future_feature": future_feature,
            "observe_trace": observe_trace,
            "future_trace": future_trace,
            "predict_trace": np.zeros((pred_length, 2)),
        }
        input_data["objects"][_obj_id] = new_obj

    input_data = add_flags(input_data)
    return input_data

def save_model(model, model_path):
    torch.save(
        {
            'xin_graph_seq2seq_model': model.state_dict(),
        },
        model_path)
    logger.warn("Model saved to {}".format(model_path))

class GradientAt:
    def __init__(self, obs_length, pred_length,predictor, data_path,source_path,samples,learn_rate=0.0001):
        super(GradientAt,self).__init__()
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.learn_rate = learn_rate
        self.data_path = data_path
        self.loss = attack_loss
        self.predictor = predictor
        self.source_path = source_path
        self.samples = samples
    def run(self):
        loss_list = []
        global loss
        try:
            self.predictor.model.train()
            print("train")
        except:
            pass
        lr = self.learn_rate
        optimizer = torch.optim.Adam(self.predictor.model.parameters(), lr=lr)
        # perb_files
        current_directory = self.data_path  # 获取当前工作目录
        files_with_ade = []  # 存储包含'ade'字符的文件


        for filename in os.listdir(current_directory):
            if 'ade' in filename:
                files_with_ade.append(current_directory + "\\" + filename)
        #files_with_ade = random_subset(files_with_ade, 20)
        # origin_data

        epochs = 100
        max_total_loss = 1000000000
        for epoch in range(epochs):
            total_loss = 0
            i = int(0)
            for name, obj_id in self.samples:
                optimizer.zero_grad()
                origin_data = load_data(os.path.join(self.source_path, "{}.json".format(name)))
                input_data = input_data_by_attack_step(origin_data, self.obs_length, self.pred_length, 0)

                for filename in files_with_ade:
                    if "txt" in filename:
                        continue

                    data1 = load_data(filename)
                    real_file = filename.split("\\")[-1]
                    perb_name = real_file.split("-")[0]
                    perb_obj = real_file.split("-")[1]
                    attack_type = real_file.split("-")[-1]
                    attack_type = attack_type.split(".")[0]
                    if(obj_id==int(perb_obj) and name==int(perb_name)):
                        i = i + 1
                    else:
                        continue
                    obj_id = str(obj_id)
                    perturbed_length = len(data1["perturbation"][str(obj_id)])

                    attack_opts = data1["attack_opts"]
                    perturbation = {"obj_id": obj_id, "loss": self.loss, "value": {}, "ready_value": {},
                                    "attack_opts": attack_opts}
                    perturbation["value"][obj_id] = data1["perturbation"][obj_id]
                    perturbation["ready_value"][obj_id] =perturbation["value"][obj_id]

                    # 组合
                    ob_trace = np.array(input_data["objects"][obj_id]["observe_trace"])
                    #ob_trace = np.array(origin_data["objects"][str(obj_id)]["observe_trace"])
                    _perturbed_trace = ob_trace[:perturbed_length,:] + perturbation["value"][obj_id]

                    input_data["objects"][obj_id]["observe_trace"] = _perturbed_trace



                    # call predictor
                    output_data = self.predictor.run(input_data, perturbation=None, backward=True)

                    args = []

                    for trace_name in ["observe_trace", "future_trace", "predict_trace"]:
                        args.append({str(obj_id): torch.from_numpy(
                            output_data["objects"][str(obj_id)][trace_name]).cuda()})
                    loss = attack_loss(*args, str(obj_id), None, type=attack_type)


                    #loss.requires_grad_(True)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    if(attack_type=="ade" and i%100==0):
                        print(f"this is the {epoch}   ,{i}, loss = {loss.item()}")
            print(total_loss)
            if(max_total_loss>total_loss):
                loss_list.append(total_loss)
                max_total_loss = total_loss
                torch.save(self.predictor.model.state_dict(),"AT_model.pt")
                #save_model(self.predictor.model,"AT_model.pt")

                print("saves")
