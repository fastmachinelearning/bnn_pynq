#!/usr/bin/env python3

import matplotlib.pyplot as plt
from determined.experimental import client

bops = []
validation_accuracy = []

for t in client.get_experiment(37).get_trials():
	try:
	        # create a list of pairs to graph
		# list = something
		bops = t.top_checkpoint().validation['metrics']['validationMetrics']['bops'])
		print(t.top_checkpoint().validation['metrics']['validationMetrics']['validation_accuracy'])
	except:
	        print("Error")
