import cv2
import numpy as np
import math

def histogram_equalization(img):  # 히스토그램 평활화
    height, width = img.shape
    result = np.zeros((height, width), np.uint8)  # result image
    histogram = np.zeros((256))
    CDF = np.zeros((256))

    for x in range(width):
        for y in range(height):
            i = img[y, x]
            histogram[i] +=1 

    for x in range(0, 255, 1):
        if x == 0:
            CDF[x] = histogram[x] / img.size
        else:
            CDF[x] = CDF[x-1] + (histogram[x] / img.size)    

    for x in range(0, 255, 1):
        CDF[x] = round(CDF[x] * 255) 

    for x in range(width):
        for y in range(height):
            j = img[y, x]
            result[y, x] = CDF[j]

    return result

def color_resoration_R(inputY, outputY, inputR, s):
    height, width = inputY.shape
    result = np.zeros((height, width), np.uint8)    # outputR

    for x in range(width):
        for y in range(height):
            result[y, x] = (outputY[y, x] * ((inputR[y, x] / outputY[y, x])**s)).astype('int')

    return result

def color_resoration_G(inputY, outputY, inputG, s):
    height, width = inputY.shape
    result = np.zeros((height, width), np.uint8)    # outputG

    for x in range(width):
        for y in range(height):
            result[y, x] = (outputY[y, x] * ((inputG[y, x] / outputY[y, x])**s)).astype('int')

    return result

def color_resoration_B(inputY, outputY, inputB, s):
    height, width = inputY.shape
    result = np.zeros((height, width), np.uint8)    # outputB

    for x in range(width):
        for y in range(height):
            result[y, x] = (outputY[y, x] * ((inputB[y, x] / outputY[y, x])**s)).astype('int')

    return result



in_image = cv2.imread('dgu_night_color.png', cv2.IMREAD_COLOR)  # 이미지 불러오기
InputB, InputG, InputR = cv2.split(in_image)    # r, g, b로 컬러 영상을 분리

yCrCb = cv2.cvtColor(in_image, cv2.COLOR_BGR2YCrCb)    # rgb to ycbcr
InputY, Cr, Cb = cv2.split(yCrCb)    # y, Cr, Cb로 컬러 영상을 분리 

OutputY = histogram_equalization(InputY)    # Y값을 히스토그램 평활화

OutputB = color_resoration_B(InputY, OutputY, InputB, 0.2)
OutputG = color_resoration_G(InputY, OutputY, InputG, 0.2)
OutputR = color_resoration_R(InputY, OutputY, InputR, 0.2)

Out_image = cv2.merge([OutputB, OutputG, OutputR])

OutputB2 = color_resoration_B(InputY, OutputY, InputB, 0.5)
OutputG2 = color_resoration_G(InputY, OutputY, InputG, 0.5)
OutputR2 = color_resoration_R(InputY, OutputY, InputR, 0.5)

Out_image2 = cv2.merge([OutputB2, OutputG2, OutputR2])

OutputB3 = color_resoration_B(InputY, OutputY, InputB, 0.8)
OutputG3 = color_resoration_G(InputY, OutputY, InputG, 0.8)
OutputR3 = color_resoration_R(InputY, OutputY, InputR, 0.8)

Out_image3 = cv2.merge([OutputB3, OutputG3, OutputR3])

cv2.imshow('Input Image', in_image)
cv2.imshow('Output Image 0.2', Out_image)
cv2.imshow('Output Image 0.5', Out_image2)
cv2.imshow('Output Image 0.8', Out_image3)

cv2.imwrite('Output Image 0.2.png', Out_image)  # save result img
cv2.imwrite('Output Image 0.5.png', Out_image2)  
cv2.imwrite('Output Image 0.8.png', Out_image3)  

cv2.waitKey()
