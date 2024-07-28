import matplotlib.pyplot as plt
import copy
import json
import numpy as np
from matplotlib import rcParams

config = {
    "font.family": 'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False,
    "font.size": 12,
}
rcParams.update(config)
def get_trace(obj, name):
    trace = obj["{}_trace".format(name)]
    trace = np.array(trace)
    if name == "predict":
        return np.array(trace)
    mask = np.array(obj["{}_mask".format(name)])
    indexes = np.argwhere(mask > 0)
    if indexes.shape[0] == 0:
        return None
    else:
        return np.array(trace[np.concatenate(indexes), :])


def draw_multi_frame_attack(input_data, obj_id, perturbation, output_data_list, filename=None):
    fig, ax = plt.subplots(figsize=(6, 10),dpi=600)
    xlim, ylim = [0x7fffffff, -0x7fffffff], [0x7fffffff, -0x7fffffff]

    for _obj_id, obj in input_data["objects"].items():
        # update boundaries
        trace_all = get_trace(obj, "observe")
        # xlim[0] = min(xlim[0], trace_all[:, 0].min())
        # xlim[1] = max(xlim[1], trace_all[:, 0].max())
        # ylim[0] = min(ylim[0], trace_all[:, 1].min())
        # ylim[1] = max(ylim[1], trace_all[:, 1].max())
        xlim[0] = 40
        xlim[1] = 120
        ylim[0] = min(ylim[0], trace_all[:, 1].min())-1
        ylim[1] = max(ylim[1], trace_all[:, 1].max())+1
        # draw lines
        gt = trace_all
        ax.plot(gt[:, 0], gt[:, 1], "bo-")
        # print object id
        last_point = trace_all[0, :]
        ax.text(last_point[0], last_point[1], "{}:{}".format(_obj_id, obj["type"]),fontdict={"fontsize":10})
    perturbed_trace = np.array([[]])
    if perturbation is not None:
        perturbation[str(obj_id)] = np.array(perturbation[str(obj_id)])
        perturbed_length = perturbation[str(obj_id)].shape[0]
        for _obj_id, _perturb_value in perturbation.items():
            _perturbed_trace = input_data["objects"][str(_obj_id)]["observe_trace"][:perturbed_length,
                               :] + _perturb_value
            ax.plot(_perturbed_trace[:, 0], _perturbed_trace[:, 1], "ro-")
            if _obj_id == str(obj_id):
                perturbed_trace = _perturbed_trace
    else:
        perturbed_trace = input_data["objects"][str(obj_id)]["observe_trace"][:, :]

    for k, output_data in output_data_list.items():
        last_point = perturbed_trace[int(k) + output_data["observe_length"] - 1, :]
        predict_trace = np.concatenate((last_point.reshape(1, 2), output_data["objects"][str(obj_id)]["predict_trace"]),
                                       axis=0)
        ax.plot(predict_trace[:, 0], predict_trace[:, 1], "ro:")

    xlabel_font = {

        'fontsize': rcParams['axes.titlesize'], # 设置成和轴刻度标签一样的大小
        'fontweight': 'light',
    }
    # fontdict 设置字体的相关属性
    # labelpad 设置轴名称到轴的间距
    # loc 设置x轴是靠那边对其
    ax.set_xlabel('x', fontdict=xlabel_font, labelpad=5, loc='center')

    ylabel_font = {
        'fontsize': rcParams['axes.titlesize'], # 设置成和轴刻度标签一样的大小
        'fontweight': 'light',
    }
    ax.set_ylabel('y', fontdict=ylabel_font, labelpad=3)

    label_fontdict = {
        'fontsize': 20,
    }
    ax.set_title('ADE: 9.87', fontdict=label_fontdict, loc='center', pad=12)
    lim = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 1.1
    ax.set_xlim([sum(xlim) / 2 - lim / 2, sum(xlim) / 2 + lim / 2])
    ax.set_ylim([sum(ylim) / 2 - lim / 2, sum(ylim) / 2 + lim / 2])
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    if filename is None:
        fig.show()
        fig.savefig(r"C:\Users\Guoxn\Desktop\result\xiaorong\base.png",dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

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
    if isinstance(json_data, dict):
        data = json_to_data(json_data)
    elif isinstance(json_data, list):
        data = []
        for x in json_data:
            data.append(json_to_data(x))
    else:
        raise Exception("Wrong format!")
    return data






if __name__ == '__main__':
    path1=r"C:\Users\Guoxn\Desktop\导师任务\车道检测任务_对抗攻防\AdvTrajectoryPrediction-master\AdvTrajectoryPrediction-master\test\data\dataset\apolloscape\multi_frame\raw\12.json"
    input_data = load_data(path1)
    # rms : r"C:\Users\Guoxn\Desktop\导师任务\本科毕设\数据\论文算法数据\our\攻击数据\tra-al\raw"    11.93
    # base : r"C:\Users\Guoxn\Desktop\导师任务\本科毕设\数据\tro-al\single_frame\attack\original\raw" 9.87
    # de-rms C:\Users\Guoxn\Desktop\导师任务\本科毕设\数据\论文算法数据\our\防御数据\troj-al\augment\augment\raw 9.10
    # de-base C:\Users\Guoxn\Desktop\导师任务\本科毕设\数据\论文算法数据\Zhang\tra-al\augment\raw  9.01
    # clean : C:\Users\Guoxn\Desktop\导师任务\本科毕设\数据\tro-al\single_frame\normal\original\raw  3.52

    # wo-adapt: C:\Users\Guoxn\Desktop\导师任务\本科毕设\数据\论文算法数据\消融\是否需要不同的模块\无自适应学习率\tra\raw  11.86
    # wo-monm : C:\Users\Guoxn\Desktop\导师任务\本科毕设\数据\论文算法数据\消融\是否需要不同的模块\无动量\tra\raw      11.87
    # wo-balan : C:\Users\Guoxn\Desktop\导师任务\本科毕设\数据\论文算法数据\消融\是否需要不同的模块\无平衡参数\tra\raw  11.00
    path2 = r"C:\Users\Guoxn\Desktop\导师任务\本科毕设\数据\tro-al\single_frame\attack\original\raw"
    output_data = json.load(open(rf"{path2}\12-12-ade.json", 'r'))
    #draw_multi_frame(output_data)
    obj_id = 12
    # result["perturbation"], result["output_data"]
    keys = output_data.keys()
    print(keys)
    draw_multi_frame_attack(input_data, obj_id, output_data["perturbation"],(output_data["output_data"]))
