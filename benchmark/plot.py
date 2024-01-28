import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark import Result

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, help="parameter dim", default=16)
parser.add_argument("-m", type=int, help="sliced space dim", default=3)
args = parser.parse_args()

result_dir = Path("./result")

pattern_proposed = r"proposed_n(?P<n>\d+)_m(?P<m>\d+)_(?P<uuid_str>[a-f0-9\-]+)\.json"
pattern_bo = r"bo_n(?P<n>\d+)_m(?P<m>\d+)_(?P<uuid_str>[a-f0-9\-]+)\.json"
pattern_cmaes = r"cmaes_n(?P<n>\d+)_m(?P<m>\d+)_(?P<uuid_str>[a-f0-9\-]+)\.json"


data = {}
for file in result_dir.iterdir():
    file_name = Path(file).name
    match1 = re.search(pattern_proposed, file_name)
    if match1:
        extracted_data = match1.groupdict()
        key = (
            "proposed",
            int(extracted_data["n"]),
            int(extracted_data["m"]),
        )
        if key not in data:
            data[key] = []
        result = Result.load(file)
        data[key].append(result)
    else:
        match2 = re.search(pattern_bo, file_name)
        if match2:
            extracted_data = match2.groupdict()
            key = (
                "saasbo",
                int(extracted_data["n"]),
                int(extracted_data["m"]),
            )
            if key not in data:
                data[key] = []
            result = Result.load(file)
            data[key].append(result)
        else:
            match3 = re.search(pattern_cmaes, file_name)
            if match3:
                extracted_data = match3.groupdict()
                key = (
                    "cmaes",
                    int(extracted_data["n"]),
                    int(extracted_data["m"]),
                )
                if key not in data:
                    data[key] = []
                result = Result.load(file)
                data[key].append(result)


fig, ax = plt.subplots(figsize=(12, 8))
for key, results in data.items():
    if key[2] != args.m or key[1] != args.n:
        continue
    # if key[0] != "cmaes":
    #     continue
    print(key)
    # if key[2] != 6:
    #     continue
    # if key[3] != 0.5:
    #     continue
    size_hist_list = []
    for result in results:
        size_hist = np.array(result.size_hist) / result.size_opt_gt
        if key[0] in ("saasbo", "cmaes"):
            size_est_hist = np.array(result.size_est_hist) / result.size_opt_gt
            size_est_max = -1
            size_hist_max_list = []
            for i in range(len(size_hist)):
                size = size_hist[i]
                size_est = size_est_hist[i]
                if size_est > size_est_max:
                    size_est_max = size_est
                    size_hist_max_list.append(size)
                else:
                    size_hist_max_list.append(size_hist_max_list[-1])
            size_hist = np.array(size_hist_max_list)
        size_hist_list.append(size_hist)
    size_hist_average = np.mean(size_hist_list, axis=0)
    size_hist_std = np.std(size_hist_list, axis=0)
    if key[0] == "cmaes":
        result.n_eval_hist = result.n_eval_hist[1:]
        size_hist_average = size_hist_average[1:]
        size_hist_std = size_hist_std[1:]
    color = None
    marker = None
    if key[0] == "proposed":
        color = "blue"
        marker = "o"
    elif key[0] == "saasbo":
        color = "red"
        marker = "x"
    elif key[0] == "cmaes":
        color = "green"
        marker = "s"
    else:
        assert False

    ax.plot(
        result.n_eval_hist,
        size_hist_average,
        label=f"{key[0]}: n={key[1]}, m={key[2]}",
        marker=marker,
        markersize=3,
        color=color,
    )
    ax.fill_between(
        result.n_eval_hist,
        size_hist_average - size_hist_std,
        size_hist_average + size_hist_std,
        alpha=0.2,
        color=color,
    )
# ax.set_xlim(0, 2000)
ax.set_ylim(0, 1.1)

# if args.n == 16 and args.m == 3:
#     ax.legend()
ax.set_xscale("log", basex=10)
size = fig.get_size_inches()
size = size * 0.4
fig.set_size_inches(*size)
plt.tight_layout()
file_path = Path(f"./fig/plot_n{args.n}_m{args.m}.png")
plt.savefig(file_path, dpi=300)

# plt.figure(figsize=(12, 8))
# for (n, m, l), array in data.items():
#     if m == 2:
#         plt.plot(array, label=f"n={n}, m={m}, l={l}")
#
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.title("Plots of Data from .npy Files")
# plt.legend()
# plt.show()
