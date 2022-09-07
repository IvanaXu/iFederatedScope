# -*-coding:utf-8-*-
# @date 2022-09-06 03:30:33
# @auth Ivan
# @goal Test In Client, No FL.

import time

time0 = time.time()

import sys

if len(sys.argv) > 1:
    datp = "/root/proj/data/CIKM22Competition/"
else:
    datp = "/Users/ivan/Library/CloudStorage/OneDrive-个人/Code/CIKM2022/data/CIKM22Competition/"

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


def get_data(cid, data_type):
    _data = torch.load(f"{datp}/{cid}/{data_type}.pt")
    _lx, _ly = _data[0].x.shape[1], _data[0].y.shape[1]
    _ledge_attr = _data[0].edge_attr.shape[1] if "edge_attr" in _data[0].keys else 0

    ydata, index_data, lxdata, ledata = [], [], [], []
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

    return _data, _xcols, _ly


def get_model1():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="regression",
        bagging_fraction=0.80,
        feature_fraction=0.80,
        max_depth=10,
        n_estimators=100,
        verbose=-1,
        n_jobs=-1,
    )


def get_score1(_yt, _yp):
    from sklearn.metrics import accuracy_score
    return 1 - accuracy_score(_yt, _yp)


def get_predict1(x, mL):
    return [int(_r) for _r in np.mean([
        i_model.predict(x) for i_model in mL
    ], axis=0)]


def get_model2():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        objective="regression",
        bagging_fraction=0.80,
        feature_fraction=0.80,
        max_depth=10,
        n_estimators=100,
        verbose=-1,
        n_jobs=-1,
    )


def get_score2(_yt, _yp):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(_yt, _yp)


def get_predict2(x, mL):
    return [float(_r) for _r in np.mean([
        i_model.predict(x) for i_model in mL
    ], axis=0)]


# cid, task_type, metric, K, model, score, predict
ids = [
    [1, ["cls", "Error rate", 4, get_model1, get_score1, get_predict1]],
    [2, ["cls", "Error rate", 2, get_model1, get_score1, get_predict1]],
    # 3, no edge_attr
    [3, ["cls", "Error rate", 4, get_model1, get_score1, get_predict1]],
    [4, ["cls", "Error rate", 3, get_model1, get_score1, get_predict1]],
    [5, ["cls", "Error rate", 11, get_model1, get_score1, get_predict1]],
    [6, ["cls", "Error rate", 7, get_model1, get_score1, get_predict1]],
    # 7, no edge_attr
    [7, ["cls", "Error rate", 2, get_model1, get_score1, get_predict1]],
    [8, ["cls", "Error rate", 9, get_model1, get_score1, get_predict1]],

    # 10/13, more Y
    [9, ["reg", "MSE", 6, get_model2, get_score2, get_predict2]],
    [10, ["reg", "MSE", 4, get_model2, get_score2, get_predict2]],
    [11, ["reg", "MSE", 2, get_model2, get_score2, get_predict2]],
    [12, ["reg", "MSE", 2, get_model2, get_score2, get_predict2]],
    [13, ["reg", "MSE", 7, get_model2, get_score2, get_predict2]],
]
# For Test
_ids = [
    # [1, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
    # [2, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
    # [3, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
    # [4, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
    # [5, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
    # [6, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
    # [7, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
    # [8, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
    # [9, ["reg", "MSE", k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
    # [10, ["reg", "MSE", k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
    # [11, ["reg", "MSE", k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
    # [12, ["reg", "MSE", k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
    # [13, ["reg", "MSE", k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
]

result, record = [], []
for [cid, paras] in ids:
    print(f"\nID {cid}:")
    [task_type, metric, K, model, score, predict] = paras

    train_data, xcols, ly = get_data(cid, "train")
    valis_data, _1, _2 = get_data(cid, "val")
    tests_data, _3, _4 = get_data(cid, "test")

    i_result = pd.DataFrame([cid for i in tests_data["data_index"]], columns=["client_id"])
    i_result["sample_id"] = tests_data["data_index"]

    iy, train_scoreL, valis_scoreL = 0, [], []
    for iy in tqdm(range(ly)):
        train_X, train_Y = train_data[xcols], train_data["y"].apply(lambda x: x[iy])
        # print(pd.value_counts(train_Y), train_X.shape, "\n")
        valis_X, valis_Y = valis_data[xcols], valis_data["y"].apply(lambda x: x[iy])
        tests_X = tests_data[xcols]

        modelL = []
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=K, shuffle=True, random_state=930721)
        for k, (i_train, i_tests) in enumerate(kf.split(train_X)):
            train_dataX1 = train_X.loc[i_train]
            train_dataX2 = train_X.loc[i_tests]
            train_dataY1 = train_Y.loc[i_train]
            train_dataY2 = train_Y.loc[i_tests]

            i_model = model()
            i_model.fit(train_dataX1, train_dataY1)
            modelL.append(i_model)

        train_score, valis_score = score(train_Y, predict(train_X, modelL)), score(valis_Y, predict(valis_X, modelL))
        # print(f""">>> {cid} Y{iy} /{K} Train {metric}: {train_score:.6f}""")
        # print(f""">>> {cid} Y{iy} /{K} Valis {metric}: {valis_score:.6f}""")

        i_result[f"Y{iy}"] = predict(tests_X, modelL)
        train_scoreL.append(train_score)
        valis_scoreL.append(valis_score)

    if True:
        train_score, valis_score = np.mean(train_scoreL), np.mean(valis_scoreL)
        std_train_valis = np.std([train_score, valis_score])
        print(
            f""">>> {cid} Y-AVG /{K} {metric}"""
            f""" Train: {train_score:.6f}"""
            f""" Valis: {valis_score:.6f}"""
            f""" STD: {std_train_valis:.6f}"""
        )
    record.extend([train_score, valis_score])
    result.append(i_result)

result = pd.concat(result)
result.to_csv(f"{datp}/result0.csv", index=False, header=False)
print(result.head(), result.shape)

with open(f"{datp}/result1.csv", "w") as f1:
    with open(f"{datp}/result0.csv", "r") as f0:
        for i in f0:
            i = i.strip("\n")
            i = ",".join([j[0] if j in ["0.0", "1.0"] else j for j in i.split(",") if j])
            f1.write(f"{i}\n")

for i in record:
    print(f"{i:.6f}")

print(f"\nUSE {time.time() - time0:.6f}s")
