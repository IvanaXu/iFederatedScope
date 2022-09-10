# -*-coding:utf-8-*-
# @date 2022-09-10 03:32:09
# @auth Ivan
# @goal Feature To Xdata

import sys
import torch
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

if len(sys.argv) > 1:
    datp = "/root/proj/data/CIKM22Competition/"
    pklp = f"{datp}/X"
else:
    datp = "/Users/ivan/Desktop/Data/CIKM2022/CIKM22Competition/"
    pklp = f"{datp}/X"


def get_data(cid, data_type, top=0):
    _data = torch.load(f"{datp}/{cid}/{data_type}.pt")
    _lx, _ly = _data[0].x.shape[1], _data[0].y.shape[1]
    _ledge_attr = _data[0].edge_attr.shape[1] if "edge_attr" in _data[0].keys else 0

    #
    ydata, index_data, lxdata, ledata = [], [], [], []
    ei0data, ei1data, ei2data, ei3data = [], [], [], []
    xdatal = [x1data, x2data, x3data, x4data, x5data, x6data, x7data] = [[] for _ in range(7)]
    edatal = [e1data, e2data, e3data, e4data, e5data, e6data, e7data] = [[] for _ in range(7)]

    for idata in _data:
        x1data.append(np.max(np.array(idata.x), axis=0))
        x2data.append(np.mean(np.array(idata.x), axis=0))
        x3data.append(np.min(np.array(idata.x), axis=0))
        x4data.append(np.std(np.array(idata.x), axis=0))
        x5data.append(np.percentile(np.array(idata.x), q=25, axis=0))
        x6data.append(np.percentile(np.array(idata.x), q=50, axis=0))
        x7data.append(np.percentile(np.array(idata.x), q=75, axis=0))

        if _ledge_attr > 0:
            e1data.append(np.max(np.array(idata.edge_attr), axis=0))
            e2data.append(np.mean(np.array(idata.edge_attr), axis=0))
            e3data.append(np.min(np.array(idata.edge_attr), axis=0))
            e4data.append(np.std(np.array(idata.edge_attr), axis=0))
            e5data.append(np.percentile(np.array(idata.edge_attr), q=25, axis=0))
            e6data.append(np.percentile(np.array(idata.edge_attr), q=50, axis=0))
            e7data.append(np.percentile(np.array(idata.edge_attr), q=75, axis=0))

        ei0 = list(np.array(idata.edge_index)[0])
        ei0data.append([1 if i0 in ei0 else 0 for i0 in range(top+1)])
        ei1 = list(np.array(idata.edge_index)[1])
        ei1data.append([1 if i1 in ei1 else 0 for i1 in range(top+1)])
        ei2 = list(np.array(idata.edge_index)[0]) + list(np.array(idata.edge_index)[1])
        ei2data.append([1 if i2 in ei2 else 0 for i2 in range(top+1)])
        ei3 = list(np.array(idata.edge_index)[1])
        ei3data.append([ei3.count(i3) for i3 in range(top+1)])

        ydata.append(np.array(idata.y)[0])
        index_data.append(idata.data_index)
        lxdata.append(idata.x.shape[0])
        ledata.append(idata.edge_index.shape[1])

    _data = pd.DataFrame([])
    _data["y"] = ydata
    _data["data_index"] = index_data
    _data["e_x"] = lxdata
    _data["e_l"] = ledata

    _xcols = ["e_x", "e_l"]
    for i in range(_lx):
        for n, xdata in enumerate(xdatal):
            _data[f"x_{n}_{i}"] = np.array(xdata)[:, i]
            _xcols.append(f"x_{n}_{i}")

    for i in range(_ledge_attr):
        for n, edata in enumerate(edatal):
            _data[f"e_{n}_{i}"] = np.array(edata)[:, i]
            _xcols.append(f"e_{n}_{i}")

    for i in range(top+1):
        for n, eidata in enumerate([ei0data, ei1data, ei2data, ei3data]):
            _data[f"ei_{n}_{i}"] = np.array(eidata)[:, i]
            _xcols.append(f"ei_{n}_{i}")

    return _data, _xcols, _ly


train_data, xcols, ly = get_data(1, "train", 10)
print(train_data)
