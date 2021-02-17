"""CSE 6363 - HWO1 - Part 2
"""
import math
import numpy as np

W = 'W'
M = 'M'

trainData_noAge = {((170, 57), W), ((190, 95), M), ((150, 45), W),
((168, 65), M), ((175, 78), M), ((185, 90), M), ((171, 65), W),
((155, 48), W), ((165, 60), W), ((182, 80), M), ((175, 69), W),
((178, 80), M), ((160, 50), W), ((170, 72), M)}

trainData = {((170, 57, 32), W), ((190, 95, 28), M), ((150, 45, 35), W),
((168, 65, 29), M), ((175, 78, 26), M), ((185, 90, 32), M), ((171, 65, 28), W),
((155, 48, 31), W), ((165, 60, 27), W), ((182, 80, 30), M), ((175, 69, 28), W),
((178, 80, 27), M), ((160, 50, 31), W), ((170, 72, 30), M)}

testX = {(162, 53, 28), (168, 75, 32), (175, 70, 30), (180, 85, 29)}

testX_noAge = {(162, 53), (168, 75), (175, 70), (180, 85)}

class KNN:
    def __init__(self, trainXY, testX, k, precision=4):
        print("--- K: ", k, "  | test datum: ", testX, " ---")
        print()
        print()
        print("trainXY:")
        print(trainXY)
        print()
        print()
        print('testX:')
        print(testX)

        self.precision = precision
        self.predictions = list()
        for datum in testX:
            prediction = self.predict_class(trainXY, datum, k)
            self.predictions.append(prediction)
        print()
        print("Summary:")
        print('k: ', k)
        print('Test set:', testX)
        print('Predictions:  ', self.predictions)
        print('Precision: ', precision, " sigfig")
        print("--- end of process ---")
        print()
        print()
        print()
        print()
        return

    def euclidean_distance(self, row_A, row_B):
        dist = 0.0
        diffList = list()
        for i in range(len(row_A[0])):
            diff = 0.0
            diff = row_A[0][i]-row_B[i]
            diffList.append(diff)
            dist += (diff)**2
        dist = math.sqrt(dist)
        return round(dist, self.precision)

    # Calculate nearest neighbors
    def get_neighbors(self, trainX, test_row, k): # trainX, trainX_i, # of nearest neighbors
        distances = list()
        neighbors = list()
        print()
        print()
        print('Calculate distances to datum: ', test_row)
        print('Training data XY  |   Distance |')
        print('__________________|____________|')
        for X_i in trainX:
            dist = self.euclidean_distance(X_i, test_row)
            print(X_i, ' |  ', dist)
            distances.append((X_i, dist))
        distances.sort(key=lambda tup: tup[1])
        for i in range(k): neighbors.append(distances[i][:])
        #self.print_distances(distances)
        #self.print_neighbors(neighbors)
        print()
        print(k, ' Nearest Neighbors:')
        for i in neighbors:
            print(i)
        return neighbors

    def predict_class(self, trainXY, testX, k):
        #print(trainXY.dtype)
        #print(testX.dtype)
        self.trainXY = np.asarray([sublist for sublist in trainXY], dtype=object)
        neighbors = self.get_neighbors(trainXY, testX, k)
        output_values = [row[-2][1] for row in neighbors]
        print()
        print('KNN classes: ', output_values)
        self.prediction = max(set(output_values), key=output_values.count)
        print('Datum prediction: ', self.prediction)
        return (self.prediction)



if __name__ == '__main__':

    """ Complete Dataset --- Part a and b """
    KNN(trainData, testX, 1)
    KNN(trainData, testX, 3)
    KNN(trainData, testX, 5)

    """ No Age Dataset --- Part c """
    KNN(trainData_noAge, testX_noAge, 1)
    KNN(trainData_noAge, testX_noAge, 3)
    KNN(trainData_noAge, testX_noAge, 5)
""" end of simple_knn.py """
