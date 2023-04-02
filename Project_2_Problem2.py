#Brendan Neal
#ENPM673 Project 1 Question 1
#bneal12

#Stitching Images Together using Homography and Feature Selection

#------------------Importing Libraries--------------------##
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

##---------Defining my Calculate Homography Function-------##
'''This function will calculate the homography between corresponding points'''

def calculateHomography(Feature_Corr_List):
    #Forming my A Matrix
    A_Matrix_as_List = []

    for correspondance in Feature_Corr_List:
         
         Point1 = np.array([correspondance.item(0), correspondance.item(1)])

         Point2 = np.array([correspondance.item(2), correspondance.item(3)])

         A1 = [Point2[0], Point2[1], 1, 0, 0, 0, -Point1[0]*Point2[0], -Point1[0]*Point2[1], -Point1[0]]

         A2 = [0,0,0, Point2[0], Point2[1],1,-Point1[1]*Point2[0], -Point1[1]*Point2[1], -Point1[1]]

         A_Matrix_as_List.append(A1)

         A_Matrix_as_List.append(A2)

    #Use SVD to get the eigenvector associated with the smallest eigenvalue.
    A = np.matrix(A_Matrix_as_List)
    _, _, v = np.linalg.svd(A)
    H_Est = np.reshape(v[8], (3,3))
    H_Est = H_Est/H_Est.item(8)
    return H_Est

##-------Defining my Error Calculation for RANSAC Homography-----##
'''This function calculates the error in order to test the accuracy of my randomly sampled homography'''
def ErrorCalculation(Feature_Corr_List, Iteration_H):
    Applied_Homography_Error = [] #Initialize the Error Array
    for i in range(len(Feature_Corr_List)): #For Every matched feature
        First_Point = np.transpose(np.matrix([Feature_Corr_List[i].item(2), Feature_Corr_List[i].item(3), 1])) #Grab a Point from Source Image
        Point2_Est = np.dot(Iteration_H, First_Point) #Project a Point on the 2nd Image
        Point2_Est = (1/Point2_Est.item(2))*Point2_Est #Normalize Point 2 Estimate
        Point2_Truth = np.transpose(np.matrix([Feature_Corr_List[i].item(0), Feature_Corr_List[i].item(1), 1])) #Grab the True Point
        error = Point2_Truth - Point2_Est #Compute Error
        error = np.linalg.norm(error) #Get Cartesian Distance
        Applied_Homography_Error.append(error) #Append to Error Matrix for Inlier Computation
    return Applied_Homography_Error

##-------Defining my RANSAC Homography Function----------##
''' This function is adapted from last projects function to randomly sample data.'''

def RANSAC_Homography(Feature_Corr_List, sample_size, threshold):

    iter_max = math.inf #Generate Temporary Max Iteration. This will change later
    iteration = 0 #Init First Iterationj
    max_inliers = 0 #Init max_inliers
    best_model = None #create best model variable
    prob_outlier = 0 #I want 0 outlier probability
    prob_des = 0.95 #I wanta  95% Accuracy Rate
    n = len(Feature_Corr_List) #Init N
    inlier_count = 0 #Init Inlier Count

    while iteration < iter_max: #While iteration number is less than calculated max
        np.random.shuffle(Feature_Corr_List) #Shuffle the data randomly
        samples = Feature_Corr_List[:sample_size,:] #Take out random data points
        iteration_model = calculateHomography(samples)
        error = ErrorCalculation(Feature_Corr_List ,iteration_model) #Calculate the Error compared to the zdata
        error = np.array(error)
        inlier_count = np.count_nonzero(error < threshold)
        if inlier_count > max_inliers: #If the number of inliers is greater than the current max:
            max_inliers = inlier_count #Update the Max Inliers
            print("Max Inliers:", max_inliers)
            best_model = iteration_model #Update the current best model
            prob_outlier = 1-(inlier_count/n) #Calculate the probability of an outlier
        if prob_outlier > 0: #If the probability of an outlier is greater than 0:
            iter_max = math.log(1-prob_des)/math.log(1-(1-prob_outlier)**sample_size) #Recalculate the new number of max iteration number
        print("Max Iterations:", iter_max, "Current Iteration:", iteration, "Current Max Inlier Count:", max_inliers)
        iteration+=1 #Increase Iteration Number
        if iteration > 10000: #Hard Code Iteration Stopper Since if it doesnt find an ideal model in 10000 iterations, it most likely wont.
             break

    return best_model

##---------------------Loading Images----------------------##
Image1 = cv.imread('image_1.jpg')
Image2 = cv.imread('image_2.jpg')
Image3 = cv.imread('image_3.jpg')
Image4 = cv.imread('image_4.jpg')

##--------------Converting Images to Gray------------------##

G_Image1 = cv.cvtColor(Image1, cv.COLOR_BGR2GRAY)
G_Image2 = cv.cvtColor(Image2, cv.COLOR_BGR2GRAY)
G_Image3 = cv.cvtColor(Image3, cv.COLOR_BGR2GRAY)
G_Image4 = cv.cvtColor(Image4, cv.COLOR_BGR2GRAY)

##--------------Create ORB Feature Extractor---------------##

ORB = cv.ORB_create()

##------------Using ORB to Extract Key Points---------------##
keypoints1 = ORB.detect(G_Image1, None)
keypoints2 = ORB.detect(G_Image2, None)
keypoints3 = ORB.detect(G_Image3, None)
keypoints4 = ORB.detect(G_Image4, None)

##----------------Attach Descriptors to Key Points---------##
keypoints1, descriptors1 = ORB.compute(G_Image1, keypoints1)
keypoints2, descriptors2 = ORB.compute(G_Image2, keypoints2)
keypoints3, descriptors3 = ORB.compute(G_Image3, keypoints3)
keypoints4, descriptors4 = ORB.compute(G_Image4, keypoints4)

##------------Displaying Key Features----------------------##
ORB_Image_1 = cv.drawKeypoints(G_Image1, keypoints1, None, color = (0,255,0), flags = 0)
plt.imshow(ORB_Image_1)
plt.show()
ORB_Image_2 = cv.drawKeypoints(G_Image2, keypoints2, None, color = (0,255,0), flags = 0)
plt.imshow(ORB_Image_2)
plt.show()
ORB_Image_3 = cv.drawKeypoints(G_Image3, keypoints3, None, color = (0,255,0), flags = 0)
plt.imshow(ORB_Image_3)
plt.show()
ORB_Image_4 = cv.drawKeypoints(G_Image4, keypoints4, None, color = (0,255,0), flags = 0)
plt.imshow(ORB_Image_4)
plt.show()


##-------------Matching Features for Homography Calculation---------------##

BF = cv.BFMatcher()

IM1_IM2_Matches = BF.match(descriptors1,descriptors2)
IM2_IM3_Matches = BF.match(descriptors2,descriptors3)
IM3_IM4_Matches = BF.match(descriptors3,descriptors4)

##-------Arranging Matches into Useful Data Structures for Homography------##

Corr_List_IM1_IM2 = []
for match in IM1_IM2_Matches:
        (x1_IM12, y1_IM12) = keypoints1[match.queryIdx].pt
        (x2_IM12, y2_IM12) = keypoints2[match.trainIdx].pt
        Corr_List_IM1_IM2.append([x1_IM12, y1_IM12, x2_IM12, y2_IM12])
Corr_Matrix_IM1_IM2 = np.matrix(Corr_List_IM1_IM2)

Corr_List_IM2_IM3 = []
for match in IM2_IM3_Matches:
        (x1_IM23, y1_IM23) = keypoints2[match.queryIdx].pt
        (x2_IM23, y2_IM23) = keypoints3[match.trainIdx].pt
        Corr_List_IM2_IM3.append([x1_IM23, y1_IM23, x2_IM23, y2_IM23])
Corr_Matrix_IM2_IM3 = np.matrix(Corr_List_IM2_IM3)

Corr_List_IM3_IM4 = []
for match in IM3_IM4_Matches:
        (x1_IM34, y1_IM34) = keypoints3[match.queryIdx].pt
        (x2_IM34, y2_IM34) = keypoints4[match.trainIdx].pt
        Corr_List_IM3_IM4.append([x1_IM34, y1_IM34, x2_IM34, y2_IM34])
Corr_Matrix_IM3_IM4 = np.matrix(Corr_List_IM3_IM4)

##--------------Matching Features for Visualization-----------------##

IM1_IM2_Matches_Vis = BF.knnMatch(descriptors1,descriptors2, k=2)
IM2_IM3_Matches_Vis = BF.knnMatch(descriptors2,descriptors3, k=2)
IM3_IM4_Matches_Vis = BF.knnMatch(descriptors3,descriptors4, k=2)


##---------Apply Ratio Test for Each Pair of Images----------##
Good_IM1_IM2 = []
for m,n in IM1_IM2_Matches_Vis:
    if m.distance < 0.75*n.distance:
        Good_IM1_IM2.append([m])

Good_IM2_IM3 = []
for m,n in IM2_IM3_Matches_Vis:
    if m.distance < 0.75*n.distance:
        Good_IM2_IM3.append([m])

Good_IM3_IM4 = []
for m,n in IM3_IM4_Matches_Vis:
    if m.distance < 0.75*n.distance:
        Good_IM3_IM4.append([m])

##--------------------Visualize Matches---------------------##

Match1_2_IMG = cv.drawMatchesKnn(G_Image1,keypoints1,G_Image2,keypoints2,Good_IM1_IM2,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
Match2_3_IMG = cv.drawMatchesKnn(G_Image2,keypoints2,G_Image3,keypoints3,Good_IM2_IM3,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
Match3_4_IMG = cv.drawMatchesKnn(G_Image3,keypoints3,G_Image4,keypoints4,Good_IM3_IM4,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(Match1_2_IMG)
plt.show()
plt.imshow(Match2_3_IMG)
plt.show()
plt.imshow(Match3_4_IMG)
plt.show()

##----------------------Calculate Homography Before Stitching-------------##

H_Est_1_2 = RANSAC_Homography(Corr_Matrix_IM1_IM2, 4, 25)
print("Finished For Images 1 and 2!")
H_Est_2_3 = RANSAC_Homography(Corr_Matrix_IM2_IM3, 4, 25)
print("Finished For Images 2 and 3!")
H_Est_3_4 = RANSAC_Homography(Corr_Matrix_IM3_IM4, 4, 25)
print("Finished For Images 3 and 4!")

##------------------------Image Stitching----------------------##


#Stitching Images 1 and 2 Together
result_IM1_IM2 = cv.warpPerspective(Image2 ,H_Est_1_2,(Image1.shape[1]+Image2.shape[1], Image2.shape[0]))
result_IM1_IM2[0:Image1.shape[0],0:Image1.shape[1]] = Image1
plt.imshow(cv.cvtColor(result_IM1_IM2, cv.COLOR_BGR2RGB))
plt.show()
 
#Stitching Images 2 and 3 Together
result_IM2_IM3 = cv.warpPerspective(Image3, H_Est_2_3, (Image2.shape[1] + Image3.shape[1], Image3.shape[0]))
result_IM2_IM3[0:Image2.shape[0], 0:Image2.shape[1]] = Image2
plt.imshow(cv.cvtColor(result_IM2_IM3, cv.COLOR_BGR2RGB))
plt.show()

#Stitching Images 3 and 4 Together
result_IM3_IM4 = cv.warpPerspective(Image4, H_Est_3_4, (Image3.shape[1]+Image4.shape[1], Image4.shape[0]))
result_IM3_IM4[0:Image3.shape[0], 0:Image3.shape[1]] = Image3
plt.imshow(cv.cvtColor(result_IM3_IM4, cv.COLOR_BGR2RGB))
plt.show()


#Layering Stitched Images Together to Form Panorama
PanoW = Image1.shape[1] + result_IM1_IM2.shape[1]+result_IM2_IM3.shape[1]+result_IM3_IM4.shape[1] #Panorama Width is the Resulting Image Width Combined
PanoH = Image1.shape[0] #Keep Height Consistent
Pano = np.zeros((PanoH, PanoW,3), dtype = np.uint8) #Initialize the Panorama
#Layer The Last Stitched Image First
Pano[:Image1.shape[0], Image1.shape[1] + result_IM1_IM2.shape[1] + result_IM2_IM3.shape[1]: Image1.shape[1] + result_IM2_IM3.shape[1]+result_IM1_IM2.shape[1]+result_IM3_IM4.shape[1], :] = result_IM3_IM4
#Layer Stitched 2 and 3
Pano[:Image1.shape[0], Image1.shape[1]: Image1.shape[1] + result_IM2_IM3.shape[1], :] = result_IM2_IM3
#Layer Stitched 1 and 2
Pano[:Image1.shape[0], :result_IM1_IM2.shape[1], :] = result_IM1_IM2
#Layer Image 1 for Good Measure
Pano[:Image1.shape[0], :Image1.shape[1], :] = Image1
plt.imshow(cv.cvtColor(Pano, cv.COLOR_BGR2RGB))
plt.show()










