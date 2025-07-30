import cv2
from math import * 
import numpy as np
import matplotlib.pyplot as plt

#calculating F_cap in frequency domain by using F_cap = (box_filter*)x(noisy_image)/((box_filter_mag)^2 + lambda*(laplacian_filter_mag)^2)
def constrainedLeastSquare(zero_padded_h_DFT_conj, zero_padded_img_DFT, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambdas):
	F_cap = np.zeros((M1+10, N1+10), dtype = complex)
	for i in range(M1+10):
		for j in range(N1+10):
			numerator = zero_padded_h_DFT_conj[i][j]*zero_padded_img_DFT[i][j]
			denominator = zero_padded_h_DFT_abs[i][j] + lambdas*zero_padded_l_DFT_abs[i][j]
			F_cap[i][j] = numerator/denominator
	return F_cap

def constrainedLeastSquareFilter(zero_padded_h_DFT_conj, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambdas):
	clsf = np.zeros((M1+10, N1+10), dtype = complex)
	for i in range(M1+10):
		for j in range(N1+10):
			numerator = zero_padded_h_DFT_conj[i][j]
			denominator = zero_padded_h_DFT_abs[i][j] + lambdas*zero_padded_l_DFT_abs[i][j]
			clsf[i][j] = numerator/denominator
	return clsf

#here I am converting F_cap in frequency domain to spatial domain
#For that I am doing inverse fourier transform. The pixel values are in the range negative to more than 255 after IFFT.
#So I am scaling those range to [0,255].
def restoredImage(F_cap):
	f_cap_temp = np.fft.ifft2(F_cap)
	f_cap_temp_2 = np.zeros((M1+10, N1+10))
	min_val = 10000000000000
	max_val = -10000000000000
	for i in range(M1+10):
		for j in range(N1+10):
			f_cap_temp_2[i][j] = np.real(f_cap_temp[i][j])
			min_val = min(min_val, f_cap_temp_2[i][j])
			max_val = max(max_val, f_cap_temp_2[i][j])
	
	for i in range(M1+10):
		for j in range(N1+10):
			f_cap_temp_2[i][j] = f_cap_temp_2[i][j] - min_val

	max_val = max_val - min_val
	for i in range(M1+10):
		for j in range(N1+10):
			f_cap_temp_2[i][j] = f_cap_temp_2[i][j]/max_val

	f_cap = np.zeros((M1, N1), dtype = np.uint8)
	for i in range(M1):
		for j in range(N1):
			f_cap[i][j] = int(255*f_cap_temp_2[i][j])

	return f_cap


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


#For mse calculations, I am using original image of cameraman and filtered image.
#then I am calculating square of difference of each cell value and summing it.
#Then dividing it by M*N
def mse_calculations(original_img, f_cap, M1, N1):
	mse = 0
	for i in range(M1):
		for j in range(N1):
			mse = mse + (int(f_cap[i][j]) - int(original_img[i][j]))**2
	mse = mse/(M1*N1)
	return mse

#Here I am calculating psnr using formula psnr = 10*log10((((2^b)-1)^2)/MSE)
def psnr_calculations(mse):
	b = 8
	num = ((2**b)-1)**2
	psnr = 10*(log10(num/mse))
	return psnr

#Here I am just finding max_psnr out of 5 psnr's and displaying image which has psnr maximum.
def max_psnr(psnr_arr):
	max_psnr = -100000000
	max_idx = 0
	for i in range(len(psnr_arr)):
		if(psnr_arr[i]>max_psnr):
			max_psnr = psnr_arr[i]
			max_idx = i

	return [max_psnr, max_idx]

#Here I am passing the image matrix matrix and then storing the count of every pixel in the pixel_count array
def count_pixels(img, pixel_count, M, N):
	for i in range(M):
		for j in range(N):
			pixel_val = img[i][j]
			pixel_count[pixel_val] = int(pixel_count[pixel_val]) + 1

#Here for every pixel, I am calculating the PDF and CDF. PDF is calculate by PDF[pixel] = pixel_count[pixel]/(M*N) and 
#then CDF[pixel] = CDF[pixel-1] + PDF[pixel]. I am storing PDF in array normalized_histogram
def histogram_normalization(pixel_count, M, N, normalized_histogram):
	CDF = np.zeros(256)
	for i in range(256):
		normalized_histogram[i] = pixel_count[i]/(M*N)
		if(i==0):
			CDF[i] = normalized_histogram[i]
		else:
			CDF[i] = CDF[i-1] + normalized_histogram[i]
	return CDF

#Here I am calculating s[pixel] which is basically where the "ith" pixel will be mapped to int(round(255*CDF[i]))
#Then I am basically for every input pixel, assigning the output[i][j] = s[pixel]
def histogram_equalization(CDF, M, N, img, output, s):
	for i in range(256):
		s[i] = int(round(255*CDF[i]))

	for i in range(M):
		for j in range(N):
			pixel_val = img[i][j]
			output[i][j] = s[pixel_val]

#calculating intensity matrix by doing summation of R,G,B values of the cell and then dividing by 3
def intensityCalc(img, M1, N1):
	intensity = np.zeros((M1, N1), dtype = np.uint8)
	for i in range(M1):
		for j in range(N1):
			R = img[i][j][0]
			G = img[i][j][1]
			B = img[i][j][2]
			val = (int(R)+int(G)+int(B))/3
			intensity[i][j] = val
	return intensity

#Here After doing histogram equalization of intensity matrix. I am converting HSI to RGB
#Here I am taking hue and saturation values of original image and taking intensity values after equalization
#Calculated Saturation as S = 1-(3*min(R,G,B)/(R+G+B+0.00001)) and added 0.00001 to avoid zero division error
#calculated theta and then assigning H = theta or H = 360-theta depending upon if B<=G or B>G respectively
#then depending upon Hue values, calculated the new R,G,B values with the help of saturation, hue and intensity
def HSItoRGB(img, M1, N1, intensity):
	RGB_img = np.zeros((M1, N1, 3), dtype = np.uint8)
	for i in range(M1):
		for j in range(N1):
			R = int(img[i][j][0])
			G = int(img[i][j][1])
			B = int(img[i][j][2])
			I = intensity[i][j]/255
			L = [R, G, B]
			L = sorted(L)
			S = 1 - ((3*L[0])/(R+G+B+0.00001))
			numerator = (R-G) + (R-B)
			denominator = 2*((((R-G)**2) + (R-B)*(G-B))**0.5)
			theta = degrees(acos(numerator/(denominator+0.00001)))
			if(B<=G):
				H = theta
			else:
				H = 360-theta
			if(H>=0 and H<120):
				new_B = I*(1-S)
				num = S*cos(radians(H))
				den = cos(radians(60-H))
				new_R = I*(1 + (num/den))
				new_G = 3*I - (new_B+new_R)
			elif(H>=120 and H<240):
				H = H - 120
				new_R = I*(1-S)
				num = S*cos(radians(H))
				den = cos(radians(60-H))
				new_G = I*(1 + (num/den))
				new_B = 3*I - (new_R+new_G)
			else:
				H = H - 240
				new_G = I*(1-S)
				num = S*cos(radians(H))
				den = cos(radians(60-H))
				new_B = I*(1 + (num/den))
				new_R = 3*I - (new_G+new_B)
				
			new_R = new_R*255
			new_G = new_G*255
			new_B = new_B*255
			if(new_R>255):
				new_R = 255

			if(new_G>255):
				new_G = 255

			if(new_B>255):
				new_B = 255


			RGB_img[i][j][0] = int(round(new_R))
			RGB_img[i][j][1] = int(round(new_G))
			RGB_img[i][j][2] = int(round(new_B))
	return RGB_img


n = int(input("enter 1 for ans 1 or enter 3 for ans 3 : "))

if(n==1):
	img = cv2.imread("./Noisy_image.jpg", 0)
	M1 = len(img)
	N1 = len(img[0])

	original_img = cv2.imread("./original_image.jpg", 0)

	#zero padding the noisy image.
	zero_padded_img = np.zeros((M1+10, N1+10), dtype = np.uint8)
	for i in range(M1):
		for j in range(N1):
			zero_padded_img[i][j] = img[i][j]

	#box filter of 11x11 dimension and then zero padded.
	zero_padded_h = np.zeros((M1+10, N1+10))
	for i in range(11):
		for j in range(11):
			zero_padded_h[i][j] = 1/121

	#laplacian filter of 3x3 dimensiona and then zero padded.
	lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
	zero_padded_l = np.zeros((M1+10, N1+10))
	for i in range(3):
		for j in range(3):
			zero_padded_l[i][j] = lap[i][j]

	#getting DFT of zero padded noisy image
	zero_padded_img_DFT = np.fft.fft2(zero_padded_img)

	#getting DFT of zero padded box filter 
	zero_padded_h_DFT = np.fft.fft2(zero_padded_h)

	#getting DFT of zero padded laplacian filter
	zero_padded_l_DFT = np.fft.fft2(zero_padded_l)

	#for box_filter_DFT, getting conjugate of each cell
	zero_padded_h_DFT_conj = np.zeros((M1+10, N1+10), dtype = complex)
	for i in range(M1+10):
		for j in range(N1+10):
			zero_padded_h_DFT_conj[i][j] = np.conj(zero_padded_h_DFT[i][j])

	#for box_filter_DFT, getting absolute of each cell.
	zero_padded_h_DFT_abs = np.zeros((M1+10, N1+10))
	for i in range(M1+10):
		for j in range(N1+10):
			zero_padded_h_DFT_abs[i][j] = abs(zero_padded_h_DFT[i][j])

	#getting square of absolute of each cell of box_filter_DFT.
	zero_padded_h_DFT_abs = np.multiply(zero_padded_h_DFT_abs, zero_padded_h_DFT_abs)

	#for laplacian_filter_DFT, getting absolute of each cell.
	zero_padded_l_DFT_abs = np.zeros((M1+10, N1+10))
	for i in range(M1+10):
		for j in range(N1+10):
			zero_padded_l_DFT_abs[i][j] = abs(zero_padded_l_DFT[i][j])

	#getting square of absolute of each cell of laplacian_filter_DFT.
	zero_padded_l_DFT_abs = np.multiply(zero_padded_l_DFT_abs, zero_padded_l_DFT_abs)

	lambda1 = 0
	lambda2 = 0.25
	lambda3 = 0.5
	lambda4 = 0.75
	lambda5 = 1

	#getting output in frequency domain, spatial domain, calculating mse and calculating psnr for lambda = 0
	F_cap1 = constrainedLeastSquare(zero_padded_h_DFT_conj, zero_padded_img_DFT, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambda1)
	f_cap1 = restoredImage(F_cap1)
	mse1 = mse_calculations(original_img, f_cap1, M1, N1)
	psnr1 = psnr_calculations(mse1)
	clsf1 = constrainedLeastSquareFilter(zero_padded_h_DFT_conj, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambda1)
	clsf1_spectrum = obtain_magnitude_spectrum(clsf1, 2)

	#getting output in frequency domain, spatial domain, calculating mse and calculating psnr for lambda = 0.25
	F_cap2 = constrainedLeastSquare(zero_padded_h_DFT_conj, zero_padded_img_DFT, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambda2)
	f_cap2 = restoredImage(F_cap2)
	mse2 = mse_calculations(original_img, f_cap2, M1, N1)
	psnr2 = psnr_calculations(mse2)
	clsf2 = constrainedLeastSquareFilter(zero_padded_h_DFT_conj, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambda2)
	clsf2_spectrum = obtain_magnitude_spectrum(clsf2, 2)

	#getting output in frequency domain, spatial domain, calculating mse and calculating psnr for lambda = 0.5
	F_cap3 = constrainedLeastSquare(zero_padded_h_DFT_conj, zero_padded_img_DFT, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambda3)
	f_cap3 = restoredImage(F_cap3)
	mse3 = mse_calculations(original_img, f_cap3, M1, N1)
	psnr3 = psnr_calculations(mse3)
	clsf3 = constrainedLeastSquareFilter(zero_padded_h_DFT_conj, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambda3)
	clsf3_spectrum = obtain_magnitude_spectrum(clsf3, 2)

	#getting output in frequency domain, spatial domain, calculating mse and calculating psnr for lambda = 0.75
	F_cap4 = constrainedLeastSquare(zero_padded_h_DFT_conj, zero_padded_img_DFT, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambda4)
	f_cap4 = restoredImage(F_cap4)
	mse4 = mse_calculations(original_img, f_cap4, M1, N1)
	psnr4 = psnr_calculations(mse4)
	clsf4 = constrainedLeastSquareFilter(zero_padded_h_DFT_conj, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambda4)
	clsf4_spectrum = obtain_magnitude_spectrum(clsf4, 2)

	#getting output in frequency domain, spatial domain, calculating mse and calculating psnr for lambda = 1
	F_cap5 = constrainedLeastSquare(zero_padded_h_DFT_conj, zero_padded_img_DFT, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambda5)
	f_cap5 = restoredImage(F_cap5)
	mse5 = mse_calculations(original_img, f_cap5, M1, N1)
	psnr5 = psnr_calculations(mse5)
	clsf5 = constrainedLeastSquareFilter(zero_padded_h_DFT_conj, zero_padded_h_DFT_abs, zero_padded_l_DFT_abs, lambda5)
	clsf5_spectrum = obtain_magnitude_spectrum(clsf5, 2)

	#getting max_psnr and printing it.
	psnr_arr = [psnr1, psnr2, psnr3, psnr4, psnr5]
	max_psnr_idx = max_psnr(psnr_arr)
	print("max psnr is :",max_psnr_idx[0])

	cv2.imshow("input", img)

	#based one max_psnr. If max_psnr index = 0, then display output1, if max_psnr index = 1, then display output2 and so on.
	if(max_psnr_idx[1]==0):
		cv2.imshow("output1", f_cap1)
		cv2.imshow("filter1 spectrum", clsf1_spectrum)

	elif(max_psnr_idx[1]==1):
		cv2.imshow("output2", f_cap2)
		cv2.imshow("filter2 spectrum", clsf2_spectrum)

	elif(max_psnr_idx[1]==2):
		cv2.imshow("output3", f_cap3)
		cv2.imshow("filter3 spectrum", clsf3_spectrum)

	elif(max_psnr_idx[1]==3):
		cv2.imshow("output4", f_cap4)
		cv2.imshow("filter4 spectrum", clsf4_spectrum)
	else:
		cv2.imshow("output5", f_cap5)
		cv2.imshow("filter5 spectrum", clsf5_spectrum)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

elif(n==3):
	img = cv2.imread("./image.tif")
	M1 = len(img)
	N1 = len(img[0])

	#intensity matrix contains the intensity of each cell of original image
	intensity = intensityCalc(img, M1, N1)

	#store count of pixels for input image
	input_pixel_count = np.zeros(256)

	#calling function count_pixels to populate the input_pixel_count array
	count_pixels(intensity, input_pixel_count, M1, N1)

	#normalized histogram initialisation
	input_normalized_histogram = np.zeros(256)

	#store the CDF corresponding to the pixels
	input_CDF = histogram_normalization(input_pixel_count, M1, N1, input_normalized_histogram)

	#this "input_s" will contain all the mappings from input pixels to transformed pixels after equalization
	input_s = np.zeros(256, dtype = np.uint8)

	#output image matrix
	output = np.zeros((M1,N1), dtype = np.uint8)

	#calling function histogram_equalization to populate output matrix
	histogram_equalization(input_CDF, M1, N1, intensity, output, input_s)

	#output pixel count
	output_pixel_count = np.zeros(256)

	#calling function count_pixels to populate the output_pixel_count array
	count_pixels(output, output_pixel_count, M1, N1)

	#normalized histogram for output
	output_normalized_histogram = np.zeros(256)

	#Here there is no need to store CDF. Just normalized_histogram needed
	histogram_normalization(output_pixel_count, M1, N1, output_normalized_histogram)

	#obtained the new RGB image after intensity equalization
	RGB_img = HSItoRGB(img, M1, N1, output)

	#displaying input and output image
	cv2.imshow("input", img)
	cv2.imshow("output", RGB_img)

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
	f1.set_ylabel("P value for each pixels")
	f1.set_title("before equalization")

	#second plot will have output_normalized_histogram which is basically probability corresponding to every pixel
	f2.bar(x_axis, output_normalized_histogram)
	f2.set_xlabel("pixel values")
	f2.set_ylabel("PDF for each pixels")
	f2.set_title("after equalization")

	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()


