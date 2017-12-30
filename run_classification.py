from pat_to_numpy import PatToNumpy
from starclassifier import StarClassifier
#from starclassifier import StarClassifier

parser = PatToNumpy()
training = parser.perform_parsing('tra.pat')
testing = parser.perform_parsing('val.pat')

if training is not None:
    training_data, training_labels = training
    testing_data, testing_labels = testing
    starclassifier = StarClassifier(training_data, training_labels, testing_data, testing_labels)
    starclassifier.train()