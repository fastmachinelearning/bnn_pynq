from determined.experimental import client
errs = 0 
trial_list =  client.get_experiment(37).get_trials()

for t in trial_list:
    try:
       
        checks = t.top_checkpoint()
        print(checks.validation['metrics']['validationMetrics']['bops'])
    except:
        errs = errs + 1 
print(errs)