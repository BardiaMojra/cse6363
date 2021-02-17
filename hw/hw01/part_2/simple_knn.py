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

class KNN:
    def __init__(self, trainXY, testX, k):
        #trainXY = np.asarray([sublist for sublist in trainXY], dtype=object)
        #testX = np.asarray([sublist for sublist in testX], dtype=object)


        self.predictions = list()
        for datum in testX:
            prediction = self.predict_class(trainXY, datum, k)
            self.predictions.append(prediction)

        print()
        print('prediction:')
        print(self.predictions)
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
        return (dist)

    # Calculate nearest neighbors
    def get_neighbors(self, trainX, test_row, k): # trainX, trainX_i, # of nearest neighbors
        distances = list()
        neighbors = list()
        for X_i in trainX:
            print('X_i: ', X_i)
            dist = self.euclidean_distance(X_i, test_row)
            distances.append((X_i, dist))
        distances.sort(key=lambda tup: tup[1])
        for i in range(k): neighbors.append(distances[i][:])
        self.print_distances(distances)
        self.print_neighbors(neighbors)
        print('Distances:')
        print(distances)
        print()
        print('Neighbors')
        print(neighbors)
        return neighbors

    def predict_class(self, trainXY, testX, k):
        #print(trainXY.dtype)
        #print(testX.dtype)
        self.trainXY = np.asarray([sublist for sublist in trainXY], dtype=object)
        neighbors = self.get_neighbors(trainXY, testX, k)
        output_values = [row[-1] for row in neighbors]
        print()
        #print('output val')
        #print(output_values)
        self.prediction = max(set(output_values), key=output_values.count)
        #print(self.prediction)
        return (self.prediction)

    # get min and max of every feature column (trainX)
    def get_trainX_minmax_list(self, trainX):
        self.minmax = list()
        for datum in range(len(trainX[0])):
            col_list = [row[datum] for row in trainX]
            self.minmax.append([min(col_list), max(col_list)])
        return self.minmax
    # normalize to 0-1
    def normalize_trainX(self, trainX):
        minmax = self.get_trainX_minmax_list(trainX)
        for row in trainX:
            for i in range(len(row)):
                row[i] = (row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])


testX = {(162, 53, 28), (168, 75, 32), (175, 70, 30), (180, 85, 29)}
testX_noAge = {(162, 53), (168, 75), (175, 70), (180, 85)}

if __name__ == '__main__':

    """ Complete Dataset --- Part a and b """

    print()
    print()
    print("trainXY")
    print(trainData)

    print()
    print()
    print('testX')
    print(testX)
    print()
    print()

    k = 1
    KNN(trainData, testX, k)
    print('for test set:')
    print(testX)
    print("k = ", k)

    k = 3
    KNN(trainData, testX, k)
    print('for test set:')
    print(testX)
    print("k = ", k)

    k = 5
    KNN(trainData, testX, k)
    print('for test set:')
    print(testX)
    print("k = ", k)


    """ No Age Dataset --- Part c """
    k = 1
    KNN(trainData_noAge, testX_noAge, k)
    print('for test set:')
    print(testX_noAge)
    print("k = ", k)

    k = 3
    KNN(trainData_noAge, testX_noAge, k)
    print('for test set:')
    print(testX_noAge)
    print("k = ", k)

    k = 5
    KNN(trainData_noAge, testX_noAge, k)
    print('for test set:')
    print(testX_noAge)
    print("k = ", k)
""" end of simple_knn.py """
