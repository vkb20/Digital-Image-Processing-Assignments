import cv2
from math import * 
import numpy as np
import matplotlib.pyplot as plt


#Here I am passing the image matrix matrix and then storing the count of every pixel in the pixel_count array.
def count_pixels(img, pixel_count, M, N):
	for i in range(M):
		for j in range(N):
			pixel_val = img[i][j]
			pixel_count[pixel_val] = int(pixel_count[pixel_val]) + 1

#Here for every pixel, I am calculating the PDF and CDF. PDF is calculate by PDF[pixel] = pixel_count[pixel]/(M*N) and 
#then CDF[pixel] = CDF[pixel-1] + PDF[pixel]. I am storing PDF in array normalized_histogram.
def histogram_normalization(pixel_count, M, N, normalized_histogram):
	CDF = np.zeros(256)
	for i in range(256):
		normalized_histogram[i] = pixel_count[i]/(M*N)
		if(i==0):
			CDF[i] = normalized_histogram[i]
		else:
			CDF[i] = CDF[i-1] + normalized_histogram[i]
	return CDF

#Here I am calculating s[pixel] which is basically where the "ith" pixel will be mapped to int(round(255*CDF[i])).
#Then I am basically for every input pixel, assigning the output[i][j] = s[pixel].
def histogram_equalization(CDF, M, N, img, output, s):
	for i in range(256):
		s[i] = int(round(255*CDF[i]))

	for i in range(M):
		for j in range(N):
			pixel_val = img[i][j]
			output[i][j] = s[pixel_val]

#Here for every input pixel value I am mapping it to c*(pixel)^gamma.
def gamma_transformation(img, gamma, M, N):
	target = np.zeros((M, N), dtype = np.uint8)
	max_in_matrix = np.amax(img)
	c = 255/((max_in_matrix)**gamma)
	for i in range(M):
		for j in range(N):
			target[i][j] = c*(img[i][j])**gamma
	return target

#Here I have two CDFs and I have to create a mapping where for every pixel I have to map it to other pixel value and 
#that pixel value will be pixel value corresponding to target CDF such that abs(target_CDF-input_CDF) is minimum.
def histogram_matching(input_CDF, target_CDF, img, M , N):
	output = np.zeros((M, N), dtype = np.uint8)
	mapping = np.zeros(256)
	for i in range(256):
		cdf = input_CDF[i]
		closest_pixel = -1
		min_diff = 2
		for j in range(256):
			temp = abs(target_CDF[j]-cdf)
			if(temp<=min_diff):
				min_diff = temp
				closest_pixel = j				
		mapping[i] = closest_pixel

	for i in range(M):
		for j in range(N):
			pixel_val = img[i][j]
			output[i][j] = mapping[pixel_val]
	return output

choose = int(input("enter 3 for solution 3, enter 4 for solution 4, enter 5 for solution 5 : "))

if(choose==3):
	print("Solution 3 goes here")
	img = cv2.imread("./A2.jpg", 0)
	M1 = len(img)
	N1 = len(img[0])

	#store count of pixels for input image
	input_pixel_count = np.zeros(256)

	#calling function count_pixels to populate the input_pixel_count array
	count_pixels(img, input_pixel_count, M1, N1)

	#normalized histogram initialisation
	input_normalized_histogram = np.zeros(256)

	#store the CDF corresponding to the pixels
	input_CDF = histogram_normalization(input_pixel_count, M1, N1, input_normalized_histogram)

	#this "input_s" will contain all the mappings from input pixels to transformed pixels after equalization
	input_s = np.zeros(256, dtype = np.uint8)

	#output image matrix
	output = np.zeros((M1,N1), dtype = np.uint8)

	#calling function histogram_equalization to populate output matrix
	histogram_equalization(input_CDF, M1, N1, img, output, input_s)

	#output pixel count
	output_pixel_count = np.zeros(256)

	#calling function count_pixels to populate the output_pixel_count array
	count_pixels(output, output_pixel_count, M1, N1)

	#normalized histogram for output
	output_normalized_histogram = np.zeros(256)

	#Here there is no need to store CDF. Just normalized_histogram needed
	histogram_normalization(output_pixel_count, M1, N1, output_normalized_histogram)

	#displaying input and output image
	cv2.imshow("input", img)
	cv2.imshow("output", output)

	#x_axis array for plotting purpose
	x_axis = np.zeros(256, dtype = np.uint8)
	for i in range(256):
		x_axis[i] = i

	#creating window for plotting bar graphs
	fig = plt.figure(figsize = (15, 5))
	f1 = fig.add_subplot(1,2,1)
	f2 = fig.add_subplot(1,2,2)

	#first plot will have input_normalized_histogram which is basically probability corresponding to every pixel
	f1.bar(x_axis, input_normalized_histogram)
	f1.set_xlabel("pixel values")
	f1.set_ylabel("PDF for input image pixels")
	f1.set_title("input image normalized histogram")

	#second plot will have output_normalized_histogram which is basically probability corresponding to every pixel
	f2.bar(x_axis, output_normalized_histogram)
	f2.set_xlabel("pixel values")
	f2.set_ylabel("PDF for output image pixels")
	f2.set_title("output image normalized histogram")

	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

elif(choose==4):
	print("Solution 4 goes here")
	img = cv2.imread("./A2.jpg", 0)
	M1 = len(img)
	N1 = len(img[0])

	#store count of pixels for input image
	input_pixel_count = np.zeros(256)

	#calling function count_pixels to populate the input_pixel_count array
	count_pixels(img, input_pixel_count, M1, N1)

	#normalized histogram initialisation
	input_normalized_histogram = np.zeros(256)

	#store the CDF corresponding to the pixels
	input_CDF = histogram_normalization(input_pixel_count, M1, N1, input_normalized_histogram)

	#gamma value
	gamma = 0.5

	#obtaining target image by calling gamma_transformation function
	target = gamma_transformation(img, gamma, M1, N1)

	#store count of pixels for target image
	target_pixel_count = np.zeros(256)

	#calling function count_pixels to populate the target_pixel_count array
	count_pixels(target, target_pixel_count, M1, N1)

	#normalized histogram initialisation
	target_normalized_histogram = np.zeros(256)

	#store the CDF corresponding to the pixels
	target_CDF = histogram_normalization(target_pixel_count, M1, N1, target_normalized_histogram)

	#output after calling histogram_matching function
	output = histogram_matching(input_CDF, target_CDF, img, M1 , N1)

	#output pixel count
	output_pixel_count = np.zeros(256)

	#calling function count_pixels to populate the output_pixel_count array
	count_pixels(output, output_pixel_count, M1, N1)

	#normalized histogram initialisation
	output_normalized_histogram = np.zeros(256)

	#store the CDF corresponding to the pixels
	histogram_normalization(output_pixel_count, M1, N1, output_normalized_histogram)

	cv2.imshow("input", img)
	cv2.imshow("target", target)
	cv2.imshow("output", output)

	#x_axis array for plotting purpose
	x_axis = np.zeros(256, dtype = np.uint8)
	for i in range(256):
		x_axis[i] = i

	#creating window for plotting bar graphs
	fig = plt.figure(figsize = (15, 12))
	f1 = fig.add_subplot(2,2,1)
	f2 = fig.add_subplot(2,2,2)
	f3 = fig.add_subplot(2,2,3)

	#first plot will have input_normalized_histogram which is basically probability corresponding to every pixel
	f1.bar(x_axis, input_normalized_histogram)
	f1.set_xlabel("pixel values")
	f1.set_ylabel("PDF for input image pixels")
	f1.set_title("input image normalized histogram")

	#second plot will have target_normalized_histogram which is basically probability corresponding to every pixel
	f2.bar(x_axis, target_normalized_histogram)
	f2.set_xlabel("pixel values")
	f2.set_ylabel("PDF for target image pixels")
	f2.set_title("target image normalized histogram")

	#third plot will have output_normalized_histogram which is basically probability corresponding to every pixel
	f3.bar(x_axis, output_normalized_histogram)
	f3.set_xlabel("pixel values")
	f3.set_ylabel("PDF for output image pixels")
	f3.set_title("output image normalized histogram")

	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

elif(choose==5):
	print("solution 5 goes here")

	#creating random int matrix of dimensions 3x3 and values ranging from 0 to 7
	img = np.random.randint(0,8,(3, 3))
	original_filter = np.random.randint(0,8,(3, 3))

	print("image matrix below")
	print(img)
	print("filter matrix below")
	print(original_filter)

	#rotating the 3x3 matrix by 180 degree
	rotated_filter = np.zeros((3,3), dtype = np.uint8)
	for i in range(2, -1, -1):
		for j in range(2, -1, -1):
			rotated_filter[2-i][2-j] = original_filter[i][j]

	print("rotated matrix below")
	print(rotated_filter)

	#padded matrix with all same values as "img" matrix from (2,2) till (4,4) and rest positions are 0.
	padded_matrix = np.zeros((7,7), dtype = np.uint8)
	for i in range(2, 5):
		for j in range(2, 5):
			padded_matrix[i][j] = img[i-2][j-2]

	#creating output matrix of dimenstions 5x5
	output = np.zeros((5,5), dtype = np.uint8)

	#window_matrix will contain all 3x3 windows while traversing the "padded_matrix"
	window_matrix = np.zeros((3,3), dtype = np.uint8)

	#looping from (0,0) till (5,5) and creating a window for each cell. For example if i=0 and j=0 then
	#window_matrix will be [[pm[0][0], pm[0][1], pm[0][2]], [pm[1][0], pm[1][1], pm[1][2]], [pm[2][0], pm[2][1], pm[2][2]]].
	#Then basically multiplying value present at filter[i][j] with window[i][j] and calculating sum. Then storing that sum
	#int the output matrix.
	for i in range(5):
		for j in range(5):
			for k in range(i, i+3):
				for l in range(j, j+3):
					val = padded_matrix[k][l]
					window_matrix[k-i][l-j] = val
			out_value = 0
			for k in range(3):
				for l in range(3):
					out_value = out_value + window_matrix[k][l]*rotated_filter[k][l]
			output[i][j] = out_value
			

	print("output matrix below")
	print(output)



