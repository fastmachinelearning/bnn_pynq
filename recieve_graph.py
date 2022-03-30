import matplotlib.pyplot as plt
from determined.common.experimental import trial
from determined.experimental import client

errs = 0

# Gets a list of the top 100 trials and sorting by
# the BEST_VALIDATION_METRIC which is currently
# defined as the validation error
trial_list = client.get_experiment(37).get_trials(
    sort_by=trial.TrialSortBy.BEST_VALIDATION_METRIC
)
list_of_pairs = []
for t in trial_list:
    try:
        # Recieved the latest checkpoint from each trial
        checks = t.select_checkpoint(latest=True)

        # Gets the current binary operation count from the checkpoint
        bop_number = checks.validation["metrics"]["validationMetrics"]["bops"]

        # Gets the current validation accuracy from the checkpoint
        accuracy = checks.validation["metrics"]["validationMetrics"][
            "validation_accuracy"
        ]

        # Creates a tuple containing both the binary operation count
        # and the accuracy
        pair = (t, bop_number, accuracy)

        # Appends this tuple to a list of tuples containing each
        # trial's binary operation count and validation accuracy
        list_of_pairs.append(pair)
    except AssertionError:
        # Increments the error count. If needed, one can print this number out
        errs = errs + 1

# Make plot
plt.xlabel("BOPS")
plt.ylabel("Accuracy")
plt.title("BOPS vs Accuracy")

# Creates a scatter plot for each trial,
# x-axis representing the binary operations
# and the y-axis representing the accuracy
for thing in list_of_pairs:
    plt.scatter(thing[1], thing[2])

plt.xscale("log")
# Saves the plot in a file called BOPSvAccuracy.png. If needed, this can be modified
plt.savefig("BOPSvAccuracy.png")
