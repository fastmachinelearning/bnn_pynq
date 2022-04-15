import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from determined.experimental import client

plt.style.use(hep.style.ROOT)


errs = 0

# Gets a list of all the trials for a given experiment
experiment = 37
os.system(
    "det experiment list-trials {experiment} | grep COMPLETED > trials.txt".format(
        experiment=experiment
    )
)
with open("trials.txt") as f:
    trial_ids = [int(line.split()[0]) for line in f.readlines()]

metrics = {}
for t in trial_ids:
    trial = client.get_trial(t)
    try:
        # Recieves the top checkpoint from each trial
        checks = trial.top_checkpoint()

        # Gets the validation metrics from the checkpoint
        for key in checks.validation["metrics"]["validationMetrics"].keys():
            if key not in metrics.keys():
                metrics[key] = []
            metrics[key].append(checks.validation["metrics"]["validationMetrics"][key])

        # Gets the hyperparameters from the checkpoint
        for key in checks.hparams.keys():
            if key not in metrics.keys():
                metrics[key] = []
            metrics[key].append(checks.hparams[key])
    except AssertionError:
        # Increments the error count. If needed, one can print this number out
        errs = errs + 1

# Creates a scatter plot for each trial,
# x-axis representing the hardware cost
# and the y-axis representing the accuracy
axis_label = {"total_bops": "BOPs", "total_mem_w_bits": "Total weight bits","inference_cost": "Inference cost"}


# CNV-W1A1 (val. acc. after 100 epochs: exp #60)
reference = {
    "total_bops": 70347776.0,
    "total_mem_w_bits": 1542848.0,
    "validation_accuracy": 0.7929,
    "inference_cost": 1
}

# Creates two lists to divide the total bops list and total_mem_w_bits by a reference
list1 = [x / reference["total_bops"] for x in metrics["total_bops"]]
list2 = [x / reference["total_mem_w_bits"] for x in metrics["total_mem_w_bits"]]

metrics["inference_cost"] = []
# Combines both lists together 
for (item1,item2) in zip(list1, list2):
    # This formula is arbitrary and can be changed in the future. However the idea
    # for this is that cnv-1w1a has an inference score of 1
    metrics["inference_cost"].append(0.5*item1 + 0.5*item2)

for key in ["total_bops", "total_mem_w_bits","inference_cost"]:
    cost = np.array(metrics[key])
    accuracy = np.array(metrics["validation_accuracy"])
    act_bits = np.array(metrics["act_bit_width"])
    weight_bits = np.array(metrics["weight_bit_width"])
    mask_w1a1 = np.logical_and(weight_bits == 1, act_bits == 1)
    mask_w1a2 = np.logical_and(weight_bits == 1, act_bits == 2)
    mask_w2a1 = np.logical_and(weight_bits == 2, act_bits == 1)
    mask_w2a2 = np.logical_and(weight_bits == 2, act_bits == 2)
    cmap = np.array(["blue", "orange", "green", "red"])
    plt.figure()
    plt.scatter(
        cost[mask_w1a1],
        accuracy[mask_w1a1],
        c=cmap[0],
        label="1-bit weights, 1-bit act.",
    )
    plt.scatter(
        cost[mask_w1a2],
        accuracy[mask_w1a2],
        c=cmap[1],
        label="1-bit weights, 2-bit act.",
    )
    plt.scatter(
        cost[mask_w2a1],
        accuracy[mask_w2a1],
        c=cmap[2],
        label="2-bit weights, 1-bit act.",
    )
    plt.scatter(
        cost[mask_w2a2],
        accuracy[mask_w2a2],
        c=cmap[3],
        label="2-bit weights, 2-bit act.",
    )
    plt.scatter(
        reference[key],
        reference["validation_accuracy"],
        label="CNV-W1A1",
        marker="*",
        c="black",
        s=160,
    )
    plt.xlabel(axis_label[key])
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.xscale("log")
    # Saves the plot
    plt.savefig("{metric}_accuracy.png".format(metric=key))
    plt.savefig("{metric}_accuracy.pdf".format(metric=key))
