#Brendan Neal
#ENPM673 Project 1 Question 1
#bneal12

#Extracting Camera Transformations using homography and the Hough Transform.

#------------------Importing Libraries--------------------##
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

#-----------Importing Camera Intrinsic Matrix-------------##

Intrinsics = np.array([[1382.58398, 0.0, 945.743164], [0.0, 1382.58398, 527.04834], [0, 0, 1]])


##-------------Importing the Paper Dimensions--------------##
PaperWidth = 216.0 #mm
PaperHeight = 270.9 #mm

##----------Setting up Real-World Paper Corner Points----##

PC1X = 0 #Top Left Corner as (0,0)
PC1Y = 0
PC2X = 0 #Bottom Left (0,21.6)
PC2Y = PaperWidth
PC3X = PaperHeight #Top Right (27.9,0)
PC3Y = 0
PC4X = PaperHeight #Bottom Right (27.9,21.6)
PC4Y = PaperWidth


#--------Loading the Video File into my Workspace---------## 

original_video = cv.VideoCapture('project2.avi')

#---------Setting Up Important Variables to be used in Video------------##

count = 0

#----------------------Defining my Hough Transform Function-------------##

def HoughTrans(BWImage): 

    hgt, wdt = BWImage.shape # we need heigth and width to calculate the diagonal length


    MaxD = np.ceil(np.sqrt(hgt**2 + wdt**2)) #Calculate the Maximum possible D for an image (which is the diagonal)
    Tested_Ds = np.arange(-MaxD, MaxD + 1, 1) #Initialize All Possible Ds (as integers)
    Tested_Thetas = np.deg2rad(np.arange(0, 180, 1)) #Initialize All Possible Thetas in rad because that's what python works with.

    # create the empty Hough Accumulator with dimensions equal to the size of possible rhos and thetas
    H = np.zeros((len(Tested_Ds), len(Tested_Thetas)))
    
    Y_Edge_Points, X_Edge_Points = np.nonzero(BWImage) #Get the Indices of the edges detected in image

    for i in range(len(X_Edge_Points)):
        XTest = X_Edge_Points[i] #Grab an edge point
        YTest = Y_Edge_Points[i] #Grab an edge point

        #Test all possible thetas on that point and give votes to acclimator
        for theta in range(len(Tested_Thetas)): 
            D = int(MaxD + XTest*np.cos(Tested_Thetas[theta]) + YTest*np.sin(Tested_Thetas[theta]))
            H[D, theta] += 1

    return H, Tested_Thetas, Tested_Ds

##---------------------Defining my Find Hough Peaks Function-----------------------##

def Find_Hough_Peaks(H, num_peaks, neighborhood_size):
    Peak_indicies = []
    H_Copy = np.copy(H)

    for i in range(num_peaks):
        idx = np.argmax(H_Copy) # Find the indices of H in flattened array
        H_Copy_idx = np.unravel_index(idx, H_Copy.shape) # Change shape to match H
        Peak_indicies.append(H_Copy_idx)

        # This next section of code prevents the peak finder from finding peaks too close to one another. I want 4 distinct lines.

        Y_Index, X_Index = H_Copy_idx # Separate x, y indices from Peak Indices

        # If the found X index is too close to an existing edge choose different relevant
        if (X_Index - (neighborhood_size/2)) < 0: 
            Minimum_X = 0
        else: 
            Minimum_X = X_Index - (neighborhood_size/2)
        if ((X_Index + (neighborhood_size/2) + 1) > H.shape[1]): 
            Maximum_X = H.shape[1]
        else: 
            Maximum_X = X_Index + (neighborhood_size/2) + 1

        #If the found Y index is too close to an existing line, choose relevant values
        if (Y_Index - (neighborhood_size/2)) < 0: 
            Minimum_Y = 0
        else: 
            Minimum_Y = Y_Index - (neighborhood_size/2)
        if ((Y_Index + (neighborhood_size/2) + 1) > H.shape[0]): 
            Maximum_Y = H.shape[0]
        else: 
            Maximum_Y = Y_Index + (neighborhood_size/2) + 1

        # Set the neighborhood size as king.

        for x in range(int(Minimum_X), int(Maximum_X)):
            for y in range(int(Minimum_Y), int(Maximum_Y)):
                H_Copy[y, x] = 0 # remove neighborhoods in H_Copy in order to start the loop again

                # highlight peaks in original H to make sure they are better found.
                if (x == Minimum_X or x == (Maximum_X - 1)):
                    H[y, x] = 255
                if (y == Minimum_Y or y == (Maximum_Y - 1)):
                    H[y, x] = 255

    # return the peak indices
    return Peak_indicies

##--------------Defining Function that will draw lines for visual confirmation----------##
def Create_Hough_Line(img, indicies, Ds, Thetas): 
    StartX = [] #Initialize the Start X Array (Need for cv.line)
    StartY = [] #Initialize the Start Y Array (Need for cv.line)
    EndX = [] #Initialize the End X Array (Need for cv.line)
    EndY = [] #Initialize the End Y Array (Need for cv.line)

    Line_Info = np.zeros((4,6)) #Initialize Hough Line info Array. Need to Capture 6 Pieces of Information for each of the 4 lines.
    for i in range(len(indicies)):

        TestD = Ds[indicies[i][0]] #Get the D associated with a peak
        TestTheta = Thetas[indicies[i][1]] #Get the Theta associated with a peak


        Aeq = np.cos(TestTheta) #Convert X Equation to Cartesian
        Beq = np.sin(TestTheta) #Convert Y Equation to Cartesian

        X_0 = Aeq*TestD #Get "Core Point X" for the Line. Line will appear off page for starting.
        Y_0 = Beq*TestD #Get "Core Point Y" for the line. Line will appear of page for starting.

        X1 = int(X_0 + 2500*(-Beq)) #Initialize Start X of Line off Page, Long enough that it will cover whole screen
        StartX.append(X1) #Append to Starting Points
        Y1 = int(Y_0 + 2500*Aeq) #Initialize Start Y of Line off Page, Long enough that it will cover whole screen
        StartY.append(Y1) #Append to Starting Points


        X2 = int(X_0 - 2500*(-Beq)) #Initialize End X of Line off page, Long enough that it will cover whole screen
        EndX.append(X2) #Append to Ending Points
        Y2 = int(Y_0 - 2500*Aeq) #Initialize End Y of Line off page, Long enough that it will cover whole screen
        EndY.append(Y2) #Append to Ending Points

        Slope_Line = (Y2-Y1)/(X2-X1) #Calculate the Slope of the Line
        B_Line = Y2-Slope_Line*X2 #Calculate the Y Intercept of the Line


        # Line Info Structure: [Line Slope, Y Intercept, StartingX, StartingY, EndingX, Ending Y]
        Line_Info[i][0] = Slope_Line #Append to Line info.
        Line_Info[i][1] = B_Line
        Line_Info[i][2] = X1
        Line_Info[i][3] = Y1
        Line_Info[i][4] = X2
        Line_Info[i][5] = Y2
        

    #Draw Lines on Screen
    cv.line(img, (StartX[0], StartY[0]), (EndX[0], EndY[0]), (0,255,0), 3)
    cv.line(img, (StartX[1], StartY[1]), (EndX[1], EndY[1]), (0,255,0), 3)
    cv.line(img, (StartX[2], StartY[2]), (EndX[2], EndY[2]), (0,255,0), 3)
    cv.line(img, (StartX[3], StartY[3]), (EndX[3], EndY[3]), (0,255,0), 3)
    return Line_Info


##-------Defining my Sort Lines Function To Have Consistent Solve Corners-----##
'''The theory behind this is that I can sort and maintain consistency of lines by each of the lines' Y intercepts
In this case, since the paper isn't rotating or changing too drastically on frame, I can ID the lines based on how large the Y intercept is'''

def Sort_Lines(Line_Info):
    Temp_Array = Line_Info.copy()
    B_Info = Temp_Array[:,1]
    Sorted_Indices = B_Info.argsort() #Sort by Y Intercept
    Sorted_Line_Info = Temp_Array[Sorted_Indices]
    return Sorted_Line_Info

##----------------Seperate Sorted Lines into Usable Data---------------------##
''' Similar to Before, This is how I sort them.'''
def InitializeLines(Sorted_Lines):
    Line1_Info = Sorted_Lines[0,:] #Line 1 is the "Top" Line because it has the smallest Y Intercept
    Line2_Info = Sorted_Lines[1,:] #Line 2 is the "Bottom" Line because it has the 2nd smallest Y Intercept
    Line3_Info = Sorted_Lines[2,:] #Line 3 is the leftmost Line because it has the 2nd largets Y Intercept
    Line4_Info = Sorted_Lines[3,:] #Line 4 is the rightmost Line Because it has the largest Y Intercept

    return Line1_Info, Line2_Info, Line3_Info, Line4_Info


##-----------Defining my Solve Intersections Function---------------------##
'''This function solves for the intersection using CRAMERS RULE'''
def Solve_Intersections(Line1Info, Line2Info):
    Line1 = Line1Info[2:6]
    Line2 = Line2Info[2:6]
    XDiff = (Line1Info[2]-Line1Info[4], Line2Info[2]-Line2Info[4])
    YDiff = (Line1Info[3]-Line1Info[5], Line2Info[3]-Line2Info[5])

    def determinant(a,b):
        Det = a[0] * b[1] - a[1] * b[0]
        return Det
    
    Divider = determinant(XDiff, YDiff)
    if Divider == 0:
       print('lines do not intersect')

    d = (determinant((Line1[0],Line1[1]), (Line1[2], Line1[3])), determinant((Line2[0],Line2[1]), (Line2[2], Line2[3])))
    X_Intersect = int(round(determinant(d, XDiff) / Divider))
    Y_Intersect = int(round(determinant(d, YDiff) / Divider))
   
    return X_Intersect, Y_Intersect

##-----------------Defining my Paper Coordinate Frame Function---------##

def SetupCoordFrame(Xc1, Yc1, Xc2, Yc2, Xc3, Yc3, Xc4, Yc4):
    XC1_New = Xc1-Xc1 #Using top left corner as origin.
    YC1_New = Yc1-Yc1
    XC2_New = Xc2-Xc1 #Bottom Left
    YC2_New = Yc2-Yc1
    XC3_New = Xc3-Xc1 #Top Right
    YC3_New = Yc3-Yc1
    XC4_New = Xc4-Xc1 #Bottom Right
    YC4_New = Yc4-Yc1
    return XC1_New, YC1_New, XC2_New, YC2_New, XC3_New, YC3_New, XC4_New, YC4_New


##----------------Defining my Solve Homography Function------------------##
def SolveHomographyMatrix(XC1, YC1, XC2, YC2, XC3, YC3, XC4, YC4, PC1X, PC1Y, PC2X, PC2Y, PC3X, PC3Y, PC4X, PC4Y):

    #Forming my A Matrix
    A = np.array([[PC1X, PC1Y, 1, 0, 0, 0, -XC1*PC1X, -XC1*PC1Y, -XC1], 
                 [0,0,0, PC1X, PC1Y,1,-YC1*PC1X, -YC1*PC1Y, -YC1],

                 [PC2X, PC2Y, 1, 0, 0, 0, -XC2*PC2X, -XC2*PC2Y, -XC2], 
                 [0,0,0, PC2X, PC2Y,1,-YC2*PC2X, -YC2*PC2Y, -YC2],

                 [PC3X, PC3Y, 1, 0, 0, 0, -XC3*PC3X, -XC3*PC3Y, -XC3], 
                 [0,0,0, PC3X, PC3Y,1,-YC3*PC1X, -YC3*PC3Y, -YC3],

                 [PC4X, PC4Y, 1, 0, 0, 0, -XC4*PC4X, -XC4*PC4Y, -XC4], 
                 [0,0,0, PC4X, PC4Y,1,-YC4*PC4X, -YC4*PC4Y, -YC4]])
    
    #Use SVD to get the eigenvector corresponding with the smallest eigenvalue
    _, _, v = np.linalg.svd(A)
    H_Est = np.reshape(v[8], (3,3))
    H_Est = H_Est/H_Est.item(8)
    return H_Est

##-----------Defining my Decompose Homography Matrix Function-----##

def DecomposeHomography(HomographyMatrix, Intrinsics_Matrix):
    Inv_K = np.linalg.inv(Intrinsics_Matrix) #Invert the Camera Intrinsic Matrix
    KinvH = np.dot(Inv_K, HomographyMatrix) #Dot Product with the Homography Matrix
    Lambda_Scale1 = np.linalg.norm(KinvH[:, 0]) #Calculate a Lambda for r1
    Lambda_Scale2 = np.linalg.norm(KinvH[:,1]) #Calculate a Lambda for r2
    Lambda_S = np.mean([Lambda_Scale1,Lambda_Scale2]) #Average the two test lambdas to get most accurate fit.
    R1 = KinvH[:,0]/Lambda_S #Solve for r1
    R2 = KinvH[:,1]/Lambda_S #Solve for r2
    Translation_Matrix = -KinvH[:,2]/Lambda_S #Negating to match coordinate frame defined by the top-left corner. X down page, Y down page, Z into page.
    R3 = np.cross(R1, R2) #Solve for r3 by crossing r1 and r2. rotation matricies are orthogonal
    Rotation_Matrix_Temp = [R1, R2, R3] #Set to Array
    Rotation_Matrix = np.reshape(Rotation_Matrix_Temp, (3,3)) #Mace 3x3 Matrix
    Rotation_Matrix = Rotation_Matrix.T #When debugging, the order was wrong, so take the transpose to correct that

    return Rotation_Matrix, Translation_Matrix






##------Defining my Roll, Pitch, Yaw Calculation Function--------##
'''I was told by professor at Office Hours to use Scipy for this so as to not induce angle wrapping errors from arctan2'''
def CalcRollPitchYaw(RotationMatrix):
    r = R.from_matrix(RotationMatrix)
    Proper_Matrix = r.as_matrix()
    Convert_Matrix = R.from_matrix(Proper_Matrix)
    Euler_Angles = Convert_Matrix.as_euler('zxy', degrees=True)
    return Euler_Angles
    # Roll = np.arctan2(RotationMatrix[2][1], RotationMatrix[2][2])
    # Pitch = np.arctan2(-RotationMatrix[2][0], np.sqrt(RotationMatrix[2][1]**2 + RotationMatrix[2][1]**2))
    # Yaw = np.arctan2(RotationMatrix[1][0], RotationMatrix[0][0])
    # return Roll, Pitch, Yaw

#----------------"MAIN" Program Sctipt-------------------##

#Defining Empty Arrays for Plotting Later
X_Trans_Plot = [] 
Y_Trans_Plot = []
Z_Trans_Plot = []
Roll_Plot = []
Pitch_Plot = []
Yaw_Plot = []
Frame_Store = []

#Checking if Video Successfully Opened
if (original_video.isOpened() == False):
    print("Error Opening File!")

# While Video is successfully loaded:
while(original_video.isOpened()):
    count = count + 1 #Increase Counter
    Frame_Store.append(count) #Saving Frame Data for Plotting
    success, img = original_video.read() #Read Image
    if success:
        GrayScale = cv.cvtColor(img, cv.COLOR_RGB2GRAY) #Set Image to GrayScale
        GaussBlur = cv.GaussianBlur(GrayScale, (7,7), 0, 0) #Using the Gaussian Blur with a 7x7 kernel to blur the image before processing.
        EdgeDetection = cv.Canny(GaussBlur, 100, 300) #Using Canny edge detection with lower threshold of 100 and upper threshold of 300 to find edges.
        #Note to Self: Lower Threshold means if less than lower pixel gradient, rejects as edge. Higher than upper means edge. In b/w only accepted if atteched to accepted.

        H, Thetas, Ds = HoughTrans(EdgeDetection) #Applying my Hough Transform Function to Start the Line-Finding Process on the Edge-Detected Image
        PeaksIDX = Find_Hough_Peaks(H, 4, 50) #Finding the Peaks of the Lines in the Image. I'm looking for 4. I also do not want to index "close" lines w/in 50 indicies in case they are stronger.
        Line_Info = Create_Hough_Line(img, PeaksIDX, Ds, Thetas) #Create the Lines and Plot them on screen.

        Sorted_Line_Info = Sort_Lines(Line_Info) #Sort the created Lines before finding the intersections
        L1, L2, L3, L4 = InitializeLines(Sorted_Line_Info) #Initialize the Lines Based on the Conditions Set Before

        #Finding the Corners
        Top_Left_Corner_X, Top_Left_Corner_Y = Solve_Intersections(L1, L3) #Lines 1 and 3 give the top left corner
        Bottom_Left_Corner_X, Bottom_Left_Corner_Y = Solve_Intersections(L3,L2) #Lines 2 and 3 give the bottom left corner
        Top_Right_Corner_X, Top_Right_Corner_Y = Solve_Intersections(L1, L4) #Lines 1 and 4 give the top right corner
        Bottom_Right_Corner_X, Bottom_Right_Corner_Y = Solve_Intersections(L2, L4) #Lines 2 and 4 give the bottom right corner.

        #Adjusting the corners to the paper coordinate frame
        XO, YO, XBL, YBL, XTR, YTR, XBR, YBR = SetupCoordFrame(Top_Left_Corner_X, Top_Left_Corner_Y, Bottom_Left_Corner_X, Bottom_Left_Corner_Y, Top_Right_Corner_X, Top_Right_Corner_Y, Bottom_Right_Corner_X, Bottom_Right_Corner_Y)

        #Solving Homography
        H_Est = SolveHomographyMatrix(XO, YO, XBL, YBL, XTR, YTR, XBR, YBR, PC1X, PC1Y, PC2X, PC2Y, PC3X, PC3Y, PC4X, PC4Y)

        #Decomposing Rotation and Translation Matrices
        Rot_Matrix, Trans_Matrix = DecomposeHomography(H_Est, Intrinsics)

        #Calculating Roll, Pitch, and Yaw
        Euler_Angles = -1*CalcRollPitchYaw(Rot_Matrix) #Negate to Correct for Coordinate Frame defined above.
        print("Roll is:", Euler_Angles[1], "Pitch is:", Euler_Angles[2], "Yaw is:", Euler_Angles[0])
        print("Translation Matrix is:\n", Trans_Matrix)

        #Saving Data for Plotting Later
        Roll_Plot.append(Euler_Angles[1])
        Pitch_Plot.append(Euler_Angles[2])
        Yaw_Plot.append(Euler_Angles[0])
        X_Trans_Plot.append(Trans_Matrix[0])
        Y_Trans_Plot.append(Trans_Matrix[1])
        Z_Trans_Plot.append(Trans_Matrix[2])
        

        #Displaying Corners on Screen
        cv.circle(img, (Top_Left_Corner_X,Top_Left_Corner_Y),4, (255,0,255), 4)
        cv.circle(img, (Bottom_Left_Corner_X,Bottom_Left_Corner_Y),4, (255,0,255), 4)
        cv.circle(img, (Top_Right_Corner_X,Top_Right_Corner_Y),4, (255,0,255), 4)
        cv.circle(img, (Bottom_Right_Corner_X,Bottom_Right_Corner_Y),4, (255,0,255), 4)
        cv.imshow("Detected Lines and Corners", img); cv.waitKey(1)
        
    else:
        original_video.release() #release video
        break

##-------------------Plotting Results---------------------------##

figure, (ax1, ax2) = plt.subplots(1,2)

ax1.set_title('Roll, Pitch, and Yaw Angles vs. Frame')
ax1.set_xlabel('Frame Number')
ax1.set_ylabel('Angle (Degrees)')
ax1.plot(Frame_Store[1:], Roll_Plot, 'r-', label = 'Roll')
ax1.plot(Frame_Store[1:], Pitch_Plot, 'b-', label = 'Pitch')
ax1.plot(Frame_Store[1:], Yaw_Plot, 'g-', label = 'Yaw')
ax1.legend()

ax2.set_title('X, Y, and Z Translations vs Frame')
ax2.set_xlabel('Frame Number')
ax2.set_ylabel('Directional Translation (Degrees)')
ax2.plot(Frame_Store[1:], X_Trans_Plot, 'r-', label = 'X Translation')
ax2.plot(Frame_Store[1:], Y_Trans_Plot, 'b-', label = 'Y Translation')
ax2.plot(Frame_Store[1:], Z_Trans_Plot, 'g-', label = 'Z Translation')
ax2.legend()

plt.show()

