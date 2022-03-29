from determined.experimental import client
import matplotlib.pyplot as plt
errs = 0 
trial_list =  client.get_experiment(37).get_trials()
list_of_pairs = []
for t in trial_list:
    try:
        checks = t.select_checkpoint(latest=True)


        bop_number = checks.validation['metrics']['validationMetrics']['bops']
        accuracy = checks.validation['metrics']['validationMetrics']['validation_accuracy']
        pair = (t,bop_number,accuracy)
        print(pair)
        list_of_pairs.append(pair)

    except:

        errs = errs + 1 
print(list_of_pairs)


fig, ax = plt.subplots()
ax.set(xlabel='BOPS', ylabel='accuracy',
       title='BOPS vs accuracy')
for thing in list_of_pairs:
    ax.scatter(thing[1],thing[2])
    

fig.savefig("best.png")