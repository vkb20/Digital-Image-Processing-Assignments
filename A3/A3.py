import cv2
from math import * 
import numpy as np

#function to obtain magnitude spectrum
def obtain_magnitude_spectrum(input_dft, gamma_value):
	M = len(input_dft)
	N = len(input_dft[0])

	input_dft_spectrum_temp = np.zeros((M, N))
	max_val = -1
	for i in range(M):
		for j in range(N):
			mag = log(abs(input_dft[i][j])+1)
			input_dft_spectrum_temp[i][j] = mag
			max_val = max(max_val, mag)

	input_dft_spectrum = np.zeros((M, N), dtype = np.uint8)
	c = 255/(max_val)**gamma_value
	for i in range(M):
		for j in range(N):
			input_dft_spectrum[i][j] = c*((input_dft_spectrum_temp[i][j])**gamma_value)

	return input_dft_spectrum


n = int(input("choose 1 for ans 1, choose 3 for ans 3 and choose 4 for ans 4 : "))

if(n==1):
	#input image
	img = cv2.imread("./image_cameraman.jpg", 0)
	M1 = len(img)
	N1 = len(img[0])

	#zero padding the input image
	zero_padded_img = np.zeros((2*M1, 2*N1), dtype = np.uint8)
	for i in range(M1):
		for j in range(N1):
			zero_padded_img[i][j] = img[i][j]

	#calculating fft of the zero padded image
	zero_padded_dft = np.fft.fft2(zero_padded_img)
	#obtaining magnitude spectrum of the zero padded image
	zero_padded_spectrum = obtain_magnitude_spectrum(zero_padded_dft, 1.2)

	#calculating centered_spatial_img by multiplying with (-1)^(i+j)
	centered_spatial_img = np.zeros((2*M1, 2*N1))
	for i in range(2*M1):
		for j in range(2*N1):
			s = i+j
			val = zero_padded_img[i][j]*((-1)**s)
			centered_spatial_img[i][j] = val

	#calculating fft of the centered_spatial_img
	centered_dft_img = np.fft.fft2(centered_spatial_img)
	#obtaining magnitude spectrum of the centered_spatial_img
	centered_magnitude_spectrum = obtain_magnitude_spectrum(centered_dft_img, 1.2)

	#initialising the filters of size 2M1x2N1
	butterworth_filter_1 = np.zeros((2*M1, 2*N1), dtype = np.uint8)
	butterworth_filter_2 = np.zeros((2*M1, 2*N1), dtype = np.uint8)
	butterworth_filter_3 = np.zeros((2*M1, 2*N1), dtype = np.uint8)

	#corresponding cutoff frequencies of filters
	cutoff1 = 10
	cutoff2 = 30
	cutoff3 = 60

	#order of the filter
	order = 2

	#obtaining filters h(i, j) = 255/(1+(D(i,j)/cutoff))^2*n
	for i in range(2*M1):
		for j in range(2*N1):
			D = ((i-(M1))**2 + (j-(N1))**2)**0.5
			#these lie in the range [0, 1] so multiply by 255 to scale
			h1 = 1 + (D/cutoff1)**(2*order)
			h2 = 1 + (D/cutoff2)**(2*order)
			h3 = 1 + (D/cutoff3)**(2*order)
			butterworth_filter_1[i][j] = 255/h1
			butterworth_filter_2[i][j] = 255/h2
			butterworth_filter_3[i][j] = 255/h3

	spectrum_filter_1 = obtain_magnitude_spectrum(butterworth_filter_1, 0.5)
	spectrum_filter_2 = obtain_magnitude_spectrum(butterworth_filter_2, 0.5)
	spectrum_filter_3 = obtain_magnitude_spectrum(butterworth_filter_3, 0.5)


	#elementwise_mult is basically multiplication of each cell (i,j) of centered_dft with each cell (i,j) of butterworth_filter
	elementwise_mult_1 = np.zeros((2*M1, 2*N1), dtype = complex)
	elementwise_mult_2 = np.zeros((2*M1, 2*N1), dtype = complex)
	elementwise_mult_3 = np.zeros((2*M1, 2*N1), dtype = complex)

	#doing the elementwise multiplication
	for i in range(2*M1):
		for j in range(2*N1):
			elementwise_mult_1[i][j] = centered_dft_img[i][j]*butterworth_filter_1[i][j]
			elementwise_mult_2[i][j] = centered_dft_img[i][j]*butterworth_filter_2[i][j]
			elementwise_mult_3[i][j] = centered_dft_img[i][j]*butterworth_filter_3[i][j]

	#calculating the inverse fft
	filtered_img_1 = np.fft.ifft2(elementwise_mult_1)
	filtered_img_2 = np.fft.ifft2(elementwise_mult_2)
	filtered_img_3 = np.fft.ifft2(elementwise_mult_3)

	#initialising the matrices that will store real part of filtered_img
	real_filtered_img_1 = np.zeros((2*M1, 2*N1))
	real_filtered_img_2 = np.zeros((2*M1, 2*N1))
	real_filtered_img_3 = np.zeros((2*M1, 2*N1))

	#assigning the real part
	for i in range(2*M1):
		for j in range(2*N1):
			real_filtered_img_1[i][j] = np.real(filtered_img_1[i][j])
			real_filtered_img_2[i][j] = np.real(filtered_img_2[i][j])
			real_filtered_img_3[i][j] = np.real(filtered_img_3[i][j])

	#min_values will contain minimum values corresponding to each matrix 
	min_val1 = 100000000
	min_val2 = 100000000
	min_val3 = 100000000

	for i in range(2*M1):
		for j in range(2*N1):
			s = i+j
			real_filtered_img_1[i][j] = real_filtered_img_1[i][j]*((-1)**s)
			real_filtered_img_2[i][j] = real_filtered_img_2[i][j]*((-1)**s)
			real_filtered_img_3[i][j] = real_filtered_img_3[i][j]*((-1)**s)
			min_val1 = min(min_val1, real_filtered_img_1[i][j])
			min_val2 = min(min_val2, real_filtered_img_2[i][j])
			min_val3 = min(min_val3, real_filtered_img_3[i][j])

	#max_values will contain max values corresponding to each matrix
	max_val1 = -1
	max_val2 = -1
	max_val3 = -1

	for i in range(2*M1):
		for j in range(2*N1):
			real_filtered_img_1[i][j] = real_filtered_img_1[i][j] - min_val1
			real_filtered_img_2[i][j] = real_filtered_img_2[i][j] - min_val2
			real_filtered_img_3[i][j] = real_filtered_img_3[i][j] - min_val3
			max_val1 = max(max_val1, real_filtered_img_1[i][j])
			max_val2 = max(max_val2, real_filtered_img_2[i][j])
			max_val3 = max(max_val3, real_filtered_img_3[i][j])

	#dividing by max_val so that each pixel value comes in range [0,1]
	for i in range(2*M1):
		for j in range(2*N1):
			real_filtered_img_1[i][j] = real_filtered_img_1[i][j]/max_val1
			real_filtered_img_2[i][j] = real_filtered_img_2[i][j]/max_val2
			real_filtered_img_3[i][j] = real_filtered_img_3[i][j]/max_val3

	#multiplying those values in range [0,1] by 255 and then converting to integer
	real_filtered_img_1_s = np.zeros((2*M1, 2*N1), dtype = np.uint8)
	real_filtered_img_2_s = np.zeros((2*M1, 2*N1), dtype = np.uint8)
	real_filtered_img_3_s = np.zeros((2*M1, 2*N1), dtype = np.uint8)
	for i in range(2*M1):
		for j in range(2*N1):
			real_filtered_img_1_s[i][j] = 255*real_filtered_img_1[i][j]
			real_filtered_img_2_s[i][j] = 255*real_filtered_img_2[i][j]
			real_filtered_img_3_s[i][j] = 255*real_filtered_img_3[i][j]


	#cropping the final output to M1xN1
	final_filtered_img_1 = np.zeros((M1, N1), dtype = np.uint8)
	final_filtered_img_2 = np.zeros((M1, N1), dtype = np.uint8)
	final_filtered_img_3 = np.zeros((M1, N1), dtype = np.uint8)
	for i in range(M1):
		for j in range(N1):
			final_filtered_img_1[i][j] = real_filtered_img_1_s[i][j]
			final_filtered_img_2[i][j] = real_filtered_img_2_s[i][j]
			final_filtered_img_3[i][j] = real_filtered_img_3_s[i][j]


	cv2.imshow("input_image", img)
	cv2.imshow("zero_padded_image", zero_padded_img)
	cv2.imshow("zero_padded_spectrum", zero_padded_spectrum)
	cv2.imshow("centered_magnitude_spectrum", centered_magnitude_spectrum)
	cv2.imshow("butterworth_filter_1", butterworth_filter_1)
	cv2.imshow("butterworth_filter_2", butterworth_filter_2)
	cv2.imshow("butterworth_filter_3", butterworth_filter_3)
	cv2.imshow("spectrum_filter_1", spectrum_filter_1)
	cv2.imshow("spectrum_filter_2", spectrum_filter_2)
	cv2.imshow("spectrum_filter_3", spectrum_filter_3)
	cv2.imshow("first_output", final_filtered_img_1)
	cv2.imshow("second_output", final_filtered_img_2)
	cv2.imshow("third_output", final_filtered_img_3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

elif(n==3):
	#input image
	img = cv2.imread("./image_cameraman.jpg", 0)
	M1 = len(img)
	N1 = len(img[0])

	#nxn box filter
	n = 9

	#assigning 1/(n*n) values to each box filter
	box_filter = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			box_filter[i][j] = 1/(n*n)

	#zero padding the filter as well as the image
	zero_padded_img = np.zeros((M1+n-1, N1+n-1), dtype = np.uint8)
	zero_padded_filter = np.zeros((M1+n-1, N1+n-1))

	#assigning values till M1xN1 and then all zeroes
	for i in range(M1):
		for j in range(N1):
			zero_padded_img[i][j] = img[i][j]

	#assigning values till n and then all zeroes
	for i in range(n):
		for j in range(n):
			zero_padded_filter[i][j] = box_filter[i][j]

	#calculating dft of the zero padded image and zero padded filter
	zero_padded_img_dft = np.fft.fft2(zero_padded_img)
	zero_padded_filter_dft = np.fft.fft2(zero_padded_filter)

	#initialising matrix to store elementwise multiplication of dft(img) and dft(filter)
	elementwise_mult = np.zeros((M1+n-1, N1+n-1), dtype = complex)
	#assigning values to matrix by multiplying (i,j) of dft(img) with (i,j) of dft(filter)
	for i in range(M1+n-1):
		for j in range(N1+n-1):
			elementwise_mult[i][j] = zero_padded_img_dft[i][j]*zero_padded_filter_dft[i][j]

	#calculating the idft
	elementwise_mult_idft = np.fft.ifft2(elementwise_mult)

	#taking the real part of idft
	real_idft = np.zeros((M1+n-1, N1+n-1), dtype = np.uint8)
	for i in range(M1+n-1):
		for j in range(N1+n-1):
			real_idft[i][j] = np.real(elementwise_mult_idft[i][j])

	#obtaining spatial convolution using inbuilt function to compare answer
	spatial_conv = cv2.filter2D(img, -1, box_filter)

	cv2.imshow("zero_padded_img", zero_padded_img)
	cv2.imshow("real_idft", real_idft)
	cv2.imshow("spatial_conv", spatial_conv)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

elif(n==4):
	#input image
	img = cv2.imread("./image_noise.jpg", 0)
	M1 = len(img)
	N1 = len(img[0])

	#zero padding the input image
	zero_padded_img = np.zeros((2*M1, 2*N1), dtype = np.uint8)
	for i in range(M1):
		for j in range(N1):
			zero_padded_img[i][j] = img[i][j]

	#calculating centered_spatial_img by multiplying with (-1)^(i+j)
	centered_spatial_img = np.zeros((2*M1, 2*N1))
	for i in range(2*M1):
		for j in range(2*N1):
			s = i+j
			val = zero_padded_img[i][j]*((-1)**s)
			centered_spatial_img[i][j] = val

	#calculating fft of the centered_spatial_img
	centered_dft_img = np.fft.fft2(centered_spatial_img)
	#obtaining magnitude spectrum of the centered_spatial_img
	centered_magnitude_spectrum = obtain_magnitude_spectrum(centered_dft_img, 2)

	filter_1 = np.zeros((2*M1, 2*N1), dtype = np.uint8)

	cutoff = 30
	order = 3

	#observed the coordinates where noise is present
	noise1_coord_x = 192
	noise1_coord_y = 192

	noise2_coord_x = 320
	noise2_coord_y = 320

	#assigning white pixels near (192, 192) with cutoff = 30
	for i in range(M1):
		for j in range(N1):
			D = ((i-(noise1_coord_x))**2 + (j-(noise1_coord_y))**2)**0.5
			h1 = 1 + (D/cutoff)**(2*order)
			filter_1[i][j] = 255/h1

	#assigning white pixels near (320, 320) with cutoff = 30
	for i in range(M1, 2*M1):
		for j in range(N1, 2*N1):
			D = ((i-(noise2_coord_x))**2 + (j-(noise2_coord_y))**2)**0.5
			h1 = 1 + (D/cutoff)**(2*order)
			filter_1[i][j] = 255/h1

	#inverting the filter values. White pixels connverted to black and vice versa
	for i in range(2*M1):
		for j in range(2*N1):
			filter_1[i][j] = 255-filter_1[i][j]

	#elementwise_mult is basically multiplication of each cell (i,j) of centered_dft with each cell (i,j) of butterworth_filter
	elementwise_mult = np.zeros((2*M1, 2*N1), dtype = complex)

	#doing the elementwise multiplication
	for i in range(2*M1):
		for j in range(2*N1):
			elementwise_mult[i][j] = centered_dft_img[i][j]*filter_1[i][j]

	#obtaining magnitude spectrum of elmentwise_mult
	mult_spectrum = obtain_magnitude_spectrum(elementwise_mult, 2)

	#doing fft of the elementwise_mult
	filtered_img = np.fft.ifft2(elementwise_mult)

	#taking only the real part of the filtered_img
	real_filtered_img = np.zeros((2*M1, 2*N1))
	for i in range(2*M1):
		for j in range(2*N1):
			real_filtered_img[i][j] = np.real(filtered_img[i][j])

	#min_values will contain minimum values corresponding to each matrix 
	min_val = 100000000

	for i in range(2*M1):
		for j in range(2*N1):
			s = i+j
			real_filtered_img[i][j] = real_filtered_img[i][j]*((-1)**s)
			min_val = min(min_val, real_filtered_img[i][j])

	#max_values will contain max values corresponding to each matrix
	max_val = -1

	for i in range(2*M1):
		for j in range(2*N1):
			real_filtered_img[i][j] = real_filtered_img[i][j] - min_val
			max_val = max(max_val, real_filtered_img[i][j])

	#dividing by max_val so that each pixel value comes in range [0,1]
	for i in range(2*M1):
		for j in range(2*N1):
			real_filtered_img[i][j] = real_filtered_img[i][j]/max_val

	#multiplying those values in range [0,1] by 255 and then converting to integer
	real_filtered_img_s = np.zeros((2*M1, 2*N1), dtype = np.uint8)

	for i in range(2*M1):
		for j in range(2*N1):
			real_filtered_img_s[i][j] = 255*real_filtered_img[i][j]

	#cropping the final output to M1xN1
	final_filtered_img = np.zeros((M1, N1), dtype = np.uint8)
	for i in range(M1):
		for j in range(N1):
			final_filtered_img[i][j] = real_filtered_img_s[i][j]

	cv2.imshow("input_image", img)
	cv2.imshow("centered_magnitude_spectrum", centered_magnitude_spectrum)
	cv2.imshow("filter_1", filter_1)
	cv2.imshow("mult_spectrum", mult_spectrum)
	cv2.imshow("final_filtered_img", final_filtered_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

