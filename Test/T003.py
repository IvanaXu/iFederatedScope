# Test submission
import sys
if len(sys.argv) > 1:
    datp = "/root/proj/data/CIKM22Competition/"
else:
    datp = "/Users/ivan/Library/CloudStorage/OneDrive-个人/Code/CIKM2022/data/CIKM22Competition/"

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd


def get_data(cid, data_type):
    _data = torch.load(f"{datp}/{cid}/{data_type}.pt")
    _lx, _ly = _data[0].x.shape[1], _data[0].y.shape[1]
    _xcols = [f"x{i}" for i in range(_lx)]

    xdata, ydata, index_data = [], [], []
    for idata in _data:
        xdata.append(np.mean(np.array(idata.x), axis=0))
        ydata.append(np.array(idata.y)[0])
        index_data.append(idata.data_index)

    _data = pd.DataFrame(xdata, columns=_xcols)
    _data["y"] = ydata
    _data["data_index"] = index_data
    return _data, _xcols, _ly


def get_model1():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=10086)


def get_score1(_yt, _yp):
    from sklearn.metrics import accuracy_score
    return 1 - accuracy_score(_yt, _yp)


def get_predict1(x, mL):
    return [int(_r) for _r in np.mean([
        i_model.predict(x) for i_model in mL
    ], axis=0)]


def get_model2():
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(random_state=10086)


def get_score2(_yt, _yp):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(_yt, _yp)


def get_predict2(x, mL):
    return [float(_r) for _r in np.mean([
        i_model.predict(x) for i_model in mL
    ], axis=0)]


# cid, task_type, metric, model, score, predict
ids = {
    1: ["cls", "Error rate", get_model1, get_score1, get_predict1],
    2: ["cls", "Error rate", get_model1, get_score1, get_predict1],
    3: ["cls", "Error rate", get_model1, get_score1, get_predict1],
    4: ["cls", "Error rate", get_model1, get_score1, get_predict1],
    5: ["cls", "Error rate", get_model1, get_score1, get_predict1],
    6: ["cls", "Error rate", get_model1, get_score1, get_predict1],
    7: ["cls", "Error rate", get_model1, get_score1, get_predict1],
    8: ["cls", "Error rate", get_model1, get_score1, get_predict1],

    9: ["reg", "MSE", get_model2, get_score2, get_predict2],
    10: ["reg", "MSE", get_model2, get_score2, get_predict2],
    11: ["reg", "MSE", get_model2, get_score2, get_predict2],
    12: ["reg", "MSE", get_model2, get_score2, get_predict2],
    13: ["reg", "MSE", get_model2, get_score2, get_predict2],
}


result = []
for cid, paras in tqdm(ids.items()):
    print(f"\nID {cid}:")
    [task_type, metric, model, score, predict] = paras

    train_data, xcols, ly = get_data(cid, "train")
    valis_data, _1, _2 = get_data(cid, "val")
    tests_data, _3, _4 = get_data(cid, "test")

    i_result = pd.DataFrame([cid for i in tests_data["data_index"]], columns=["client_id"])
    i_result["sample_id"] = tests_data["data_index"]

    print("\n")
    for iy in range(ly):
        train_X, train_Y = train_data[xcols], train_data["y"].apply(lambda x: x[iy])
        valis_X, valis_Y = valis_data[xcols], valis_data["y"].apply(lambda x: x[iy])
        tests_X = tests_data[xcols]

        modelL = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=8, shuffle=True, random_state=930721)
        for k, (i_train, i_tests) in enumerate(kf.split(train_X)):
            train_dataX1 = train_X.loc[i_train]
            train_dataX2 = train_X.loc[i_tests]
            train_dataY1 = train_Y.loc[i_train]
            train_dataY2 = train_Y.loc[i_tests]

            i_model = model()
            i_model.fit(train_dataX1, train_dataY1)
            # print(f""">>> {cid} K{k}-T {metric}: {score(train_dataY1, i_model.predict(train_dataX1)):.6f}""")
            # print(f""">>> {cid} K{k}-V {metric}: {score(train_dataY2, i_model.predict(train_dataX2)):.6f}""")
            modelL.append(i_model)

        print(f""">>> {cid} Y{iy} Train {metric}: {score(train_Y, predict(train_X, modelL)):.6f}""")
        print(f""">>> {cid} Y{iy} Valis {metric}: {score(valis_Y, predict(valis_X, modelL)):.6f}""")

        i_result[f"Y{iy}"] = predict(tests_X, modelL)
    result.append(i_result)

result = pd.concat(result)
result.to_csv(f"{datp}/result0.csv", index=False, header=False)
print(result)

with open(f"{datp}/result1.csv", "w") as f1:
    with open(f"{datp}/result0.csv", "r") as f0:
        for i in f0:
            i = i.strip("\n")
            i = ",".join([j[0] if j in ["0.0", "1.0"] else j for j in i.split(",") if j])
            f1.write(f"{i}\n")
