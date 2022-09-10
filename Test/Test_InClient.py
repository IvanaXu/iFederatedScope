# -*-coding:utf-8-*-
# @date 2022-09-06 03:30:33
# @auth Ivan
# @goal Test In Client, No FL.

import os
import time
time0 = time.time()

import sys

if len(sys.argv) > 1:
    datp = "/root/proj/data/CIKM22Competition/"
else:
    datp = "/Users/ivan/Desktop/Data/CIKM2022/CIKM22Competition/"

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

lej0 = ['11-12', '4-7', '6-3', '14-3', '13-5', '7-0', '3-24', '11-14', '11-7', '19-1', '5-18', '26-1', '3-21', '8-16', '43-2', '28-4', '37-2', '41-3', '34-2', '42-1', '11-1', '17-0', '10-14', '12-13', '7-18', '5-4', '2-15', '8-20', '19-15', '6-19', '2-8', '5-11', '17-13', '15-10', '10-2', '18-16', '37-29', '31-19', '29-31', '34-22', '38-3', '8-34', '20-4', '3-27', '7-29', '4-33', '14-26', '43-20', '18-5', '8-46', '5-31', '19-20', '40-2', '8-40', '2-34', '4-32', '32-4', '0-51', '28-40', '5-43', '42-39', '1-38', '27-5', '41-32', '0-43', '43-24', '40-43', '2-43', '2-39', '40-4', '8-3', '2-26', '4-28', '2-25', '23-21', '14-20', '9-25', '13-14', '7-20', '20-3', '7-22', '15-13', '16-3', '5-19', '15-3', '1-12', '12-8', '0-14', '0-16', '20-2', '4-11', '5-13', '6-8', '8-14', '7-14', '17-2', '3-9', '1-13', '0-2', '4-1']


def get_data(cid, data_type, top=0):
    _data = torch.load(f"{datp}/{cid}/{data_type}.pt")
    _lx, _ly = _data[0].x.shape[1], _data[0].y.shape[1]
    _ledge_attr = _data[0].edge_attr.shape[1] if "edge_attr" in _data[0].keys else 0

    #
    ydata, index_data, lxdata, ledata = [], [], [], []
    ei0data, ei1data = [], []
    ej0data = []
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
        # ei1 = list(np.array(idata.edge_index)[1])
        # ei1data.append([1 if i1 in ei1 else 0 for i1 in range(top+1)])

        ej0 = str(list(np.array(idata.edge_index[1])))
        ej0data.append([1 if i1 in ej0 else 0 for i1 in lej0])

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
        for n, eidata in enumerate([ei0data]):
            _data[f"ei_{n}_{i}"] = np.array(eidata)[:, i]
            _xcols.append(f"ei_{n}_{i}")

    for n, data in enumerate(lej0):
        _data[f"ej_{data}"] = np.array(ej0data)[:, n]
        _xcols.append(f"ej_{data}")

    return _data, _xcols, _ly


def get_model1():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="regression",
        bagging_fraction=0.80,
        feature_fraction=0.80,
        max_depth=9,
        n_estimators=100,
        verbose=-1,
        n_jobs=-1,
        learning_rate=0.1,
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
        max_depth=9,
        n_estimators=100,
        verbose=-1,
        n_jobs=-1,
        learning_rate=0.1,
    )


def get_score2(_yt, _yp):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(_yt, _yp)


def get_predict2(x, mL):
    return [float(_r) for _r in np.mean([
        i_model.predict(x) for i_model in mL
    ], axis=0)]


# cid, task_type, metric, top, K, model, score, predict
TEST = True
# TEST = False
if TEST:
    # For Test
    ids = [
        # [1, ["cls", "Error rate", 111, k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        [2, ["cls", "Error rate", 29, k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [3, ["cls", "Error rate", 105, k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [4, ["cls", "Error rate", 22, k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [5, ["cls", "Error rate", 29, k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [6, ["cls", "Error rate", 99, k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [7, ["cls", "Error rate", 91, k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [8, ["cls", "Error rate", 63, k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [9, ["reg", "MSE", 36, k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
        # [10, ["reg", "MSE", 11, k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
        # [11, ["reg", "MSE", 48, k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
        # [12, ["reg", "MSE", 34, k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
        # [13, ["reg", "MSE", 28, k, get_model2, get_score2, get_predict2]] for k in range(2, 21)

        # [2, ["cls", "Error rate", 29, 9, get_model1, get_score1, get_predict1]],
        # [4, ["cls", "Error rate", 22, 4, get_model1, get_score1, get_predict1]],
        # [11, ["reg", "MSE", 48, 2, get_model2, get_score2, get_predict2]],
    ]
else:
    ids = [
        [1, ["cls", "Error rate", 111, 4, get_model1, get_score1, get_predict1]],
        [2, ["cls", "Error rate", 29, 9, get_model1, get_score1, get_predict1]],
        # 3, no edge_attr
        [3, ["cls", "Error rate", 105, 3, get_model1, get_score1, get_predict1]],
        [4, ["cls", "Error rate", 22, 4, get_model1, get_score1, get_predict1]],
        [5, ["cls", "Error rate", 29, 2, get_model1, get_score1, get_predict1]],
        [6, ["cls", "Error rate", 99, 2, get_model1, get_score1, get_predict1]],
        # 7, no edge_attr
        [7, ["cls", "Error rate", 91, 3, get_model1, get_score1, get_predict1]],
        [8, ["cls", "Error rate", 63, 18, get_model1, get_score1, get_predict1]],

        # 10/13, more Y
        [9, ["reg", "MSE", 36, 2, get_model2, get_score2, get_predict2]],
        [10, ["reg", "MSE", 11, 3, get_model2, get_score2, get_predict2]],
        [11, ["reg", "MSE", 48, 2, get_model2, get_score2, get_predict2]],
        [12, ["reg", "MSE", 34, 2, get_model2, get_score2, get_predict2]],
        [13, ["reg", "MSE", 28, 2, get_model2, get_score2, get_predict2]],
    ]


min_train_valis = np.inf
result, record = [], []
for [cid, paras] in ids:
    print(f"\nID {cid}:")
    [task_type, metric, top, K, model, score, predict] = paras

    train_data, xcols, ly = get_data(cid, "train", top)
    valis_data, _1, _2 = get_data(cid, "val", top)
    tests_data, _3, _4 = get_data(cid, "test", top)

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
            f""" {"✅" if TEST and min_train_valis > std_train_valis else ""}"""
        )
        if min_train_valis > std_train_valis:
            min_train_valis = std_train_valis

    record.extend([train_score, valis_score])
    result.append(i_result)

result = pd.concat(result)
result.to_csv(f"{datp}/result0.csv", index=False, header=False)
print(result.head(), result.shape)

with open(f"{datp}/result1.csv", "w") as f1:
    with open(f"{datp}/result0.csv", "r") as f0:
        for i in f0:
            i = i.strip("\n")
            # when 0.0,1.0 always in 0 for Test/ j[0]
            i = ",".join([j[0] if j in ["0.0", "1.0"] else j for j in i.split(",") if j])
            f1.write(f"{i}\n")

with open(".record", "w") as f:
    for i in record:
        f.write(f"{i:.6f}\n")

print(f"\nUSE {time.time() - time0:.6f}s")
os.system('say "i finish the job"')
