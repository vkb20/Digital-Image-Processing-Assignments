import cv2
import math
import numpy as np


print("Solution to question 3 goes here")
input_matrix = cv2.imread("./assign1.jpg", 0)
M1 = len(input_matrix)
N1 = len(input_matrix[0])


'''
#user has to input the number of rows and columns
M1 = int(input("enter number of rows for input matrix : "))
N1 = int(input("enter number of cols for input matrix : "))

#creating matrix of M1xN1 initialized with zeros
input_matrix = np.zeros((M1, N1))

#asking user to input each cell value
for i in range(M1):
	for j in range(N1):
		input_matrix[i][j] = int(input("enter row " + str(i) + " column " + str(j) + " value : "))

#print(input_matrix)
'''

isInputCorrect = False
while(isInputCorrect==False):
	#asking user to input the interpolating factor
	C = float(input("enter interpolating factor : "))

	#number of rows and columns of interpolated matrix
	M2 = M1*C
	N2 = N1*C
	if(M2==int(M2) and N2==int(N2)):
		isInputCorrect = True
		M2 = int(M1*C)
		N2 = int(N1*C)
	else:
		print("ERROR : Enter valid value of C!!!")

#creating matrix of M2xN2 initialized with zeros
interpolated_matrix = np.zeros((M2, N2), dtype = np.uint8)

for i in range(M2):
	for j in range(N2):
		x = i/C
		y = j/C

		#this case is when both the "x" and "y" are directly mapping to input matrix.
		if(x==math.floor(x) and y==math.floor(y)):
			if(x<M1 and y<N1):
				interpolated_matrix[i][j] = input_matrix[int(x)][int(y)]
		else:
			#when the "x" value is somewhere between two discrete values.
			if(x!=math.floor(x)):
				x1 = math.floor(x)
				x2 = math.ceil(x)
			else:
				#when the "x" value is directly mapping to input matrix value.
				#case 1 : when x is zero then take x1 = 0 and x2 = 1 as these will be the closest to 0.
				#case 2 : when x is not zero then take x1 = x-1 and x2 = x.
				if(x==0):
					x1 = 0
					x2 = x+1
				else:
					x1 = x-1
					x2 = x

			#when the "y" value is somewhere between two discrete values.
			if(y!=math.floor(y)):
				y1 = math.floor(y)
				y2 = math.ceil(y)
			else:
				#when the "y" value is directly mapping to input matrix value.
				#case 1 : when y is zero then take y1 = 0 and y2 = 1 as these will be the closest to 0.
				#case 2 : when y is not zero then take y1 = y-1 and y2 = y.
				if(y==0):
					y1 = 0
					y2 = y+1
				else:
					y1 = y-1
					y2 = y

			#converting the coordinates to int
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

			#only considering cases when x2 is not exceeding the maximum row value of input matrix
			#and y2 is not exceeding the maximum column value of input matrix.
			if(x2<M1 and y2<N1):
				X = [
				[x1, y1, x1*y1, 1], 
				[x1, y2, x1*y2, 1],
				[x2, y2, x2*y2, 1],
				[x2, y1, x2*y1, 1],
				]

				V = [
					[input_matrix[x1][y1]], 
					[input_matrix[x1][y2]],
					[input_matrix[x2][y2]],
					[input_matrix[x2][y1]],
					]

				#A = (X^-1)*V
				A = np.dot(np.linalg.inv(X), V)

				interpolated_matrix[i][j] = np.dot(np.array([x, y, x*y, 1]), A)

#till rem_x the rows are already filled and till rem_y the columns are already filled
rem_x = int((M1-1)*C)
rem_y = int((N1-1)*C)

#fill the remaining row till rem_y
for i in range(rem_x+1, M2):
	for j in range(0, rem_y+1):
		interpolated_matrix[i][j] = interpolated_matrix[i-1][j]

#filling the full remaining column
for j in range(rem_y+1, N2):
	for i in range(0, M2):
		interpolated_matrix[i][j] = interpolated_matrix[i][j-1]

#print(interpolated_matrix)


cv2.imshow("output image", interpolated_matrix)
cv2.imshow("input image", input_matrix)
cv2.waitKey(0)
cv2.destroyAllWindows()

