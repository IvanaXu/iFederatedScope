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

ids = range(1, 13+1)
# ids = [1, 13]
# ids = [1]


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


def get_score1(_yt, _yp):
    from sklearn.metrics import accuracy_score
    return 1 - accuracy_score(_yt, _yp)


def get_score2(_yt, _yp):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(_yt, _yp)


result = []
for cid in tqdm(ids):
    print(f"\nID {cid}:")
    train_data, xcols, ly = get_data(cid, "train")
    valis_data, _1, _2 = get_data(cid, "val")
    tests_data, _3, _4 = get_data(cid, "test")

    iresult = pd.DataFrame([cid for i in tests_data["data_index"]], columns=["client_id"])
    iresult["sample_id"] = tests_data["data_index"]

    print()
    if cid < 9:
        for iy in range(ly):
            train_X, train_Y = train_data[xcols], train_data["y"].apply(lambda x: x[iy])
            valis_X, valis_Y = valis_data[xcols], valis_data["y"].apply(lambda x: x[iy])
            tests_X = tests_data[xcols]

            modelL = []
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=6, shuffle=True, random_state=930721)
            for k, (i_train, i_tests) in enumerate(kf.split(train_X)):
                train_dataX1 = train_X.loc[i_train]
                train_dataX2 = train_X.loc[i_tests]
                train_dataY1 = train_Y.loc[i_train]
                train_dataY2 = train_Y.loc[i_tests]

                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier()
                model.fit(train_dataX1, train_dataY1)

                print(f""">>> {cid} K{k}-T Error rate: {get_score1(train_dataY1, model.predict(train_dataX1)):.6f}""")
                print(f""">>> {cid} K{k}-V Error rate: {get_score1(train_dataY2, model.predict(train_dataX2)):.6f}""")
                modelL.append(model)


            def predict1(x):
                return [int(_r) for _r in np.mean([
                    i_model.predict(x) for i_model in tqdm(modelL)
                ], axis=0)]

            print(f""">>> {cid} Y{iy} Train Error rate: {get_score1(train_Y, predict1(train_X)):.6f}""")
            print(f""">>> {cid} Y{iy} Valis Error rate: {get_score1(valis_Y, predict1(valis_X)):.6f}""")

            iresult[f"Y{iy}"] = predict1(tests_X)
    else:
        for iy in range(ly):
            train_X, train_Y = train_data[xcols], train_data["y"].apply(lambda x: x[iy])
            valis_X, valis_Y = valis_data[xcols], valis_data["y"].apply(lambda x: x[iy])
            tests_X = tests_data[xcols]

            modelL = []
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=6, shuffle=True, random_state=930721)
            for k, (i_train, i_tests) in enumerate(kf.split(train_X)):
                train_dataX1 = train_X.loc[i_train]
                train_dataX2 = train_X.loc[i_tests]
                train_dataY1 = train_Y.loc[i_train]
                train_dataY2 = train_Y.loc[i_tests]

                from sklearn.tree import DecisionTreeRegressor
                model = DecisionTreeRegressor()
                model.fit(train_dataX1, train_dataY1)

                print(f""">>> {cid} Y{iy} K{k}-T MSE: {get_score2(train_dataY1, model.predict(train_dataX1)):.6f}""")
                print(f""">>> {cid} Y{iy} K{k}-V MSE: {get_score2(train_dataY2, model.predict(train_dataX2)):.6f}""")
                modelL.append(model)


            def predict2(x):
                return [float(_r) for _r in np.mean([
                    i_model.predict(x) for i_model in tqdm(modelL)
                ], axis=0)]


            print(f""">>> {cid} Y{iy} Train MSE: {get_score2(train_Y, predict2(train_X)):.6f}""")
            print(f""">>> {cid} Y{iy} Valis MSE: {get_score2(valis_Y, predict2(valis_X)):.6f}""")

            iresult[f"Y{iy}"] = predict2(tests_X)
    result.append(iresult)

result = pd.concat(result)
result.to_csv(f"{datp}/result0.csv", index=False, header=False)
print(result)

with open(f"{datp}/result1.csv", "w") as f1:
    with open(f"{datp}/result0.csv", "r") as f0:
        for i in f0:
            i = i.strip("\n")
            i = ",".join([j[0] if j in ["0.0", "1.0"] else j for j in i.split(",") if j])
            f1.write(f"{i}\n")
