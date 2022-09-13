#!/usr/bin/env python
import pandas as pd
import pickle

# from core.configuration import CONFIGS


machine_idx = {
    1: range(1, 8 + 1),
    2: range(1, 9 + 1),
    3: range(1, 11 + 1),
}
# config = CONFIGS["smd"]()
n_var = 37

####################
# train set
####################
print("preprocessing train set... ")

dfs = []
times_by_machine = [0]
machine_num = 0
times = 0
for i, v in machine_idx.items():
    for j in v:
        df = pd.read_csv(f"templates/ServerMachineDataset/train/machine-{i}-{j}.txt", header=None)

        df = df.drop(columns=[7])

        df.columns = [f"var_{i}" for i in range(n_var)]

        times += len(df.index)
        times_by_machine.append(times)

        df["machine_num"] = [machine_num for _ in range(len(df.index))]
        df["time"] = [i for i in range(len(df.index))]
        dfs.append(df)

        machine_num += 1

data = pd.concat(dfs, ignore_index=True)

mean = data.iloc[:, :n_var].mean(axis=0)
std = data.iloc[:, :n_var].std(axis=0)

data.iloc[:, :n_var] = (data.iloc[:, :n_var] - mean) / std

mean_std = pd.DataFrame({"mean": mean, "std": std,})
column_names = list(data.columns)

with open("data/column_names.bin", "wb") as f:
    pickle.dump(column_names, f)
with open("data/times_by_machine_train.bin", "wb") as f:
    pickle.dump(times_by_machine, f)
data.to_csv("data/SMD_train.csv")
mean_std.to_csv("data/SMD_mean_std.csv")
print("Done")


####################
# test set
####################
print("preprocessing test set... ")

dfs = []
times_by_machine = [0]
machine_num = 0
times = 0
for i, v in machine_idx.items():
    for j in v:
        df = pd.read_csv(f"templates/ServerMachineDataset/test/machine-{i}-{j}.txt", header=None)
        label = pd.read_csv(f"templates/ServerMachineDataset/test_label/machine-{i}-{j}.txt", header=None)

        df = df.drop(columns=[7])

        df.columns = [f"var_{i}" for i in range(n_var)]

        times += len(df.index)
        times_by_machine.append(times)

        df["machine_num"] = [machine_num for _ in range(len(df.index))]
        df["time"] = [i for i in range(len(df.index))]
        df["label"] = label
        dfs.append(df)

        machine_num += 1

data = pd.concat(dfs, ignore_index=True)

# mean_std = pd.read_csv("data/mean_std.csv", index_col=0)
# mean = mean_std["mean"]
# std = mean_std["std"]

data.iloc[:, :n_var] = (data.iloc[:, :n_var] - mean) / std

with open("data/times_by_machine_test.bin", "wb") as f:
    pickle.dump(times_by_machine, f)
data.to_csv("data/SMD_test.csv")
print("Done")
