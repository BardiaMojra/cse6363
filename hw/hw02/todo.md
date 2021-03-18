# ToDo:

- Make latex for answers.
- Confirm Q1 answer and training. Double check the input bias handing.
- Write Q1 discussion.
- Write Q2 discussion.

Perceptron (Question 1)
Remember that a perceptron here is a single unit that uses the sign function as
an activation function. You do not have to implement a network of Perceptrons,
only a single Perceptron.

Classification using Neural Network (Question 2)
You only need to implement the given network architecture and only for the
given datasets. You will need to implement your own error back propagation
equations (but only for this specific architecture).
To use the datasets with a neural network you will have to do one of the
following two changes if you are using Sigmoid function units, your network can only output values between 0 and 1. As a result a target value Y=-1 is not really possible. As a result you should replace the target values (Y) for the two classes in the data sets with 1 and 0 (i.e. just replace the -1 Y entries in the dataset with 0.
You could build your neural network using tanh() activation functions which can produce outputs between -1 and 1 and can thus be trained with the original class labels of 1 and -1.
To determine the class in the Neural Network, use the same way as we used in Logistic Regression
For sigmoid function networks, if the output is > 0.5 it corresponds to class 1, otherwise it corresponds to class 0 (-1)
For tanh() activation function, if the output is >0 it corresponds to predicting class 1, otherwise it corresponds to class -1
Reporting misclassified items in Questions 1 and 2 to investigate convergence:
For this, I want you, after each set of n learning iterations, to compute the predicted class for every one of the data items by computing the output of the network for this data item and determine how many were incorrectly classified. The gives you data that shows how the number of misclassified items changes as the number of training iterations increases. (you can plot this as a curve of misclassifications over time to make it easier to look at). The way this relate to convergence is that if your network converges you would expect this number to eventually stabilize and no longer change while if it does not converge, the number will continue to change (and oscillate).

Doing iterations of SMO in Question 3:
This is intended to be on paper. You do not have to implement an algorithm.
The basic operation of an iteration fo the algorithm is
Pick two alpha parameters for optimization, treating all others as constants (usually a sophisticated heuristic is used here to determine what the most promising parameters are - i.e. which ones are most likely to correspond to support vectors; for this assignment you can pick them any way you want)
Rewrite one of them as a function of the other and put them into the Lagrangian formulation under equilibrium conditions
Optimize the Lagrangian while ignoring the constraints
Clip the resulting value along the line so that none of the alphas is less than 0
Compute  w and b that correspond to the new set of alphas
Determine whether the resulting values fulfill all the KKT conditions (i.e. if they correspond to an extremum of the function and all the constraints are fulfilled)
If any of the conditions are violated, go to the next iteration
Running SVM implementations in Question 3:
Existing implementations of SVM (sklearn, Matlab, etc.) will contain a parameter, usually called C, which is a regularization term to address non-linearly separable datasets (we will talk about this in detail on Tuesday). For the assignment, the dataset is linearly separable so you should set this value to a large number (C=1000 or so) to force it to find a solution that will linearly separate the two classes.
