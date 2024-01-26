import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark import Result

result_dir = Path("./result")

pattern_proposed = (
    r"proposed_n(?P<n>\d+)_m(?P<m>\d+)_r(?P<r>[0-9.]+)_(?P<uuid_str>[a-f0-9\-]+)\.pkl"
)
pattern_bo = (
    r"bo_n(?P<n>\d+)_m(?P<m>\d+)_(?P<uuid_str>[a-f0-9\-]+)\.pkl"
)


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
            float(extracted_data["r"]),
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
                "bo",
                int(extracted_data["n"]),
                int(extracted_data["m"]),
            )
            if key not in data:
                data[key] = []
            result = Result.load(file)
            data[key].append(result)


fig, ax = plt.subplots(figsize=(12, 8))
for key, results in data.items():
    print(key)
    # if key[2] != 6:
    #     continue
    # if key[3] != 0.5:
    #     continue
    size_hist_list = []
    for result in results:
        size_hist = np.array(result.size_hist) / result.size_opt_gt
        if key[0] == "bo":
            size_hist = np.maximum.accumulate(size_hist)
        size_hist_list.append(size_hist)
    size_hist_average = np.mean(size_hist_list, axis=0)
    size_hist_std = np.std(size_hist_list, axis=0)
    ax.plot(
        result.n_eval_hist, size_hist_average, label=f"{key[0]}: n={key[1]}, m={key[2]}"
    )
    ax.fill_between(
        result.n_eval_hist,
        size_hist_average - size_hist_std,
        size_hist_average + size_hist_std,
        alpha=0.2,
    )
ax.legend()
plt.show()

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
