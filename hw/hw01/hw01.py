

# py38


def poisson_MLE(data):
    """
    Poisson distribution is ca
    """

    sum = 0.0

    for i in data:
        sum =+ data[i]

    mean = sum / data.__len__()

    return(mean)







# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

