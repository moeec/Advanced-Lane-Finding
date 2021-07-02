import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip

################Distortion correction calculated via camera calibration###########

# Adjustable checkerboard Dimensions
cbrow = 6
cbcol = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    cal_img = cv2.imread(fname)
    gray = cv2.cvtColor(cal_img, cv2.COLOR_BGR2GRAY)

# Find Chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


################Pipeline For Binary Image###########



class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted_right = [] 
        # x values of the last n fits of the line
        self.recent_xfitted_left = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx_right = None
        #average x values of the fitted line over the last n iterations
        self.bestx_left = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_right = None  
        #polynomial coefficients for the most recent fit
        self.best_fit_left = None  
        #polynomial coefficients for the most recent fit
        self.current_fit_right = [np.array([False])]  
        #polynomial coefficients for the most recent fit
        self.current_fit_left = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs_right = np.array([0,0,0], dtype='float') 
        #difference in fit coefficients between last and new fits
        self.diffs_left = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels onb right
        self.allx_right = None  
        #y values for detected line pixels on right
        self.ally_right = None  
        #x values for detected line pixels on left
        self.allx_left = None  
        #y values for detected line pixels on lfet
        self.ally_left = None
        #keeping track of frames
        self.frame = 0

global track_lines
track_lines = Line()
        
def abs_sobel_thresh(img, orient='x', thresh_min=90, thresh_max=255): # not use in  initial assignment, left for reference and for harder challenge 
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    #binary_output = np.copy(img) # Remove this line
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def clr_thresh(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    print(color_binary.shape)
    return color_binary


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    track_lines.allx = nonzerox
    track_lines.ally = nonzeroy
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image - Leaving this if needed, this was commented out to have a clean output
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        #(win_xleft_high,win_y_high),(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),
        #(win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
     
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
       
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    track_lines.allx_right = rightx
    track_lines.ally_right = righty
    track_lines.allx_left = leftx
    track_lines.ally_left = lefty
          
    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
            
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions - left for reference 
    #out_img[lefty, leftx] = [255, 0, 0]
    #out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines - left for reference
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #cv2.imwrite('fp.jpg', plt.plot(left_fitx, ploty, color='yellow'))
    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit


def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100
    
    left_fit = track_lines.current_fit_left
    right_fit = track_lines.current_fit_right

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 0.01, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result, left_fitx, right_fitx, ploty, left_fit, right_fit


def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty, left_fit, right_fit

def warp(img):

    img_size = (img.shape[1], img.shape[0])
    #Four source coordinates
    src = np.float32(
        [[585,460],
         [1127,720],
         [200,720],
         [695,460]])
    
    #Four desired coordinates
    dst = np.float32(
        [[320,0],
         [960,720],
         [320,720],
         [960,0]])
    
    #Compute perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)
    
    #Compute inverse perspective transform, Minv
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    #Create warped image using linar interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped


def unwarp(img):

    img_size = (img.shape[1], img.shape[0])
    #Four source coordinates
    src = np.float32(
        [[585,460],
         [1127,720],
         [200,720],
         [695,460]])
    
    #Four desired coordinates
    dst = np.float32(
        [[320,0],
         [960,720],
         [320,720],
         [960,0]])
    
    #Compute perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)
    
    #Compute inverse perspective transform, Minv
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    #Create warped image using linar interpolation
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)
    return unwarped

def measure_curvature_pixels(ploty, left_fit, right_fit, left_fitx, right_fitx):
    #Calculates the curvature of polynomial functions in pixels.
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # Choosing the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad


def measure_position_meters(image_w_lanes_marked, left_fit, right_fit):
    # Define conversions in x from pixels space to meters(x just needed for this calculation)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Choose the y value corresponding to the bottom of the image
    bottom_of_image = image_w_lanes_marked[0]

    # Calculate left and right line positions at the bottom of the image
    left_x_bottom_position = left_fit[0]*bottom_of_image**2 + left_fit[1]*bottom_of_image + left_fit[2]  
    right_x_bottom_position = right_fit[0]*bottom_of_image**2 + right_fit[1]*bottom_of_image + right_fit[2]
    
    #Average out above numbers to get single value for left & right    
    left_x_bottom_position_avg = np.mean(left_x_bottom_position)
    right_x_bottom_position_avg = np.mean(right_x_bottom_position) 
    
    # Calculate the x position of the center of the lane 
    center_of_lane_x_position = (left_x_bottom_position_avg + left_x_bottom_position_avg)//2
    
    # Calculate the deviation from center of the lane 
    # The camera on the care is assumed to be in the center of the car
    car_position = (center_of_lane_x_position - left_x_bottom_position_avg ) * xm_per_pix 
    
    return car_position

def draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty):     
    # Create an image to draw on and an image to show the selection window
    out_img = binary_warped 
    window_img = np.zeros_like(binary_warped)

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_lane_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_lane_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
    # Horizontally stacking left and right & a Bug with fillPoly, needs explict cast to 32bit
    lane_pts = np.int32(np.hstack((left_lane_pts, right_lane_pts)))
   
    left_lane_pts = np.int32([left_lane_pts])

    right_lane_pts = np.int32([right_lane_pts])

    # Draw the lane onto the warped blank image
    cv2.fillPoly(out_img, pts = lane_pts, color = (0,255,0),)
    # Combine the result with the original image
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return result
    
def display_curvature_and_car_pos_info(img, curve_radius, car_position_from_center):
    # Displaying curvature & car position on frame
    
    #Choosing font Hershey Simplex
    font = cv2.FONT_HERSHEY_SIMPLEX     
    
    #Placement of text for Curve Radius info
    text = 'Curve Radius: ' + '{:02.4f}'.format(curve_radius/1000) + 'km'
    cv2.putText(img, text, (30,70), font, 1, (255,255,255), 2, cv2.LINE_AA)
    
    #Placement of text for Car Position info  
    text = 'Car Position From Center: ' + '{:02.4f}'.format(car_position_from_center) + 'm'
    cv2.putText(img, text, (30,120), font, 1, (255,255,255), 2, cv2.LINE_AA)
    
    return img

def ad_lane_finding_pipeline(img):
    #pipeline for video edit
    global leftx
    global rightx
    global lefty
    global righty
    track_lines.frame +=1
    pers_transform = warp(img)
    hls_img = hls_select(pers_transform, thresh=(90, 255))
    gradx = abs_sobel_thresh(pers_transform, orient='x', thresh_min=30, thresh_max=255)
   
    grady = abs_sobel_thresh(pers_transform, orient='y', thresh_min=30, thresh_max=100)
    mag_binary = mag_thresh(pers_transform, sobel_kernel=3, mag_thresh=(20, 100))
    dir_binary = dir_threshold(pers_transform, sobel_kernel=3, thresh=(0, np.pi/2))
    
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        
    combined_final = np.zeros_like(dir_binary)
    combined_final[(combined == 1)|(hls_img == 1)] = 1


    
    #Detect lines, this will detect lines in the first frame
    if (track_lines.frame == 1):
        global poly_image
        global left_fitx
        global right_fitx 
        global ploty
        global left_fit
        global right_fit
        poly_image, left_fitx, right_fitx, ploty, left_fit, right_fit = fit_polynomial(combined_final)
        draw_step = draw_poly_lines(poly_image, left_fitx, right_fitx, ploty)
        track_lines.bestx_right = (track_lines.current_fit_right+right_fit)/2 
        track_lines.bestx_left = (track_lines.current_fit_left+left_fit)/2 
        track_lines.diffs_right = right_fit - track_lines.current_fit_right
        track_lines.diffs_left = left_fit - track_lines.current_fit_left
        Check_diffs_left = track_lines.diffs_left.item(0)
        Check_diffs_left = abs(Check_diffs_left)
        track_lines.recent_xfitted_right = right_fitx
        track_lines.recent_xfitted_left =  left_fitx
        track_lines.current_fit_right = right_fit
        track_lines.current_fit_left = left_fit
        rightx = track_lines.allx_right   
        righty = track_lines.ally_right   
        leftx = track_lines.allx_left     
        lefty = track_lines.ally_left
       
    else:
        #------------------Searching Around Previous Region-------------------------
        #Picking up values from previous frame
        right_fitx = track_lines.current_fit_right 
        left_fit = track_lines.current_fit_left
        rightx = track_lines.allx_right  
        righty = track_lines.ally_right 
        leftx = track_lines.allx_left   
        lefty = track_lines.ally_left 
        poly_image, left_fitx, right_fitx, ploty, left_fit, right_fit = search_around_poly(combined_final)
        draw_step = draw_poly_lines(poly_image, left_fitx, right_fitx, ploty) 
    
    #Verify lines were detected and proceed onto next frame
    if (left_fitx.size & right_fitx.size == 0):
        #toggle track_lines to False 
        track_lines.detected = False
        #Reset track__lines.frame to 1 sending pipeline back to full image detecttion and not searching around in previous areas
        track_lines.frame == 1
        print("No Lane Line detected")
    
    unwarp_step = unwarp(draw_step)
    unwarp_step = np.uint8(unwarp_step)
    image_w_lanes_marked = cv2.addWeighted(img, 0.8, unwarp_step, 1, 0) 
    left_curveradius, right_curveradius = measure_curvature_pixels(ploty, left_fit, right_fit, left_fitx, right_fitx)
    curve_radius = (left_curveradius+right_curveradius)/2
    track_lines.radius_of_curvature = curve_radius
    car_position_from_center = measure_position_meters(image_w_lanes_marked, left_fit, right_fit)
    track_lines.line_base_pos = car_position_from_center 
    final_image_w_text = display_curvature_and_car_pos_info(image_w_lanes_marked, curve_radius, car_position_from_center)
    #Tracking of lines detected & drawn
    
    track_lines.bestx_right = (track_lines.current_fit_right+right_fit)/2 
    track_lines.bestx_left = (track_lines.current_fit_left+left_fit)/2 
    track_lines.diffs_right = right_fit - track_lines.current_fit_right
    track_lines.diffs_left = left_fit - track_lines.current_fit_left
    Check_diffs_left = track_lines.diffs_left.item(0)
    Check_diffs_left = abs(Check_diffs_left)
    track_lines.recent_xfitted_right = right_fitx
    track_lines.recent_xfitted_left =  left_fitx
    track_lines.current_fit_right = right_fit
    track_lines.current_fit_left = left_fit
    

    return final_image_w_text

#Reading in images for Prespective Transform for Assignment

img1 = mpimg.imread("test_images/straight_lines1.jpg")
img2 = mpimg.imread("test_images/straight_lines2.jpg")
img3 = mpimg.imread("test_images/test1.jpg")
img4 = mpimg.imread("test_images/test2.jpg")
img5 = mpimg.imread("test_images/test3.jpg")
img6 = mpimg.imread("test_images/test4.jpg")
img7 = mpimg.imread("test_images/test5.jpg")
img8 = mpimg.imread("test_images/test6.jpg")

#Running Prespective transform on images
pers_transform_test1 = warp(img1)
pers_transform_test2 = warp(img2)
pers_transform_test3 = warp(img3)
pers_transform_test4 = warp(img4)
pers_transform_test5 = warp(img5)
pers_transform_test6 = warp(img6)
pers_transform_test7 = warp(img7)
pers_transform_test8 = warp(img8)

#Writing images files with results from Prespective transform of images above
cv2.imwrite("output_images/straight_lines1_output.jpg", pers_transform_test1)
cv2.imwrite("output_images/straight_lines2_output.jpg", pers_transform_test2)
cv2.imwrite("output_images/test1_output.jpg", pers_transform_test1)
cv2.imwrite("output_images/test2_output.jpg", pers_transform_test1)
cv2.imwrite("output_images/test3_output.jpg", pers_transform_test1)
cv2.imwrite("output_images/test4_output.jpg", pers_transform_test1)
cv2.imwrite("output_images/test5_output.jpg", pers_transform_test1)
cv2.imwrite("output_images/test6_output.jpg", pers_transform_test1)


poly_demo_input = mpimg.imread("test_images/test1.jpg")

t = ad_lane_finding_pipeline(poly_demo_input)

#poly_demo, poly_demo_left_fitx, poly_demo_right_fitx, poly_demo_ploty, poly_demo_left_fit, poly_demo_right_fit = fit_polynomial(poly_demo_input)
#poly_demo_draw_step = draw_poly_lines(poly_demo, poly_demo_left_fitx, poly_demo_right_fitx, poly_demo_ploty)
poly_demo_transform = warp(poly_demo_input)
poly_demo_hls_img = hls_select(poly_demo_transform, thresh=(90, 255))
poly_demo_gradx = abs_sobel_thresh(poly_demo_transform, orient='x', thresh_min=30, thresh_max=255)   
poly_demo_grady = abs_sobel_thresh(poly_demo_transform, orient='y', thresh_min=30, thresh_max=100)
poly_demo_mag_binary = mag_thresh(poly_demo_transform, sobel_kernel=3, mag_thresh=(20, 100))
poly_demo_dir_binary = dir_threshold(poly_demo_transform, sobel_kernel=3, thresh=(0, np.pi/2))
    
poly_demo_combined = np.zeros_like(poly_demo_dir_binary)
poly_demo_combined[((poly_demo_gradx == 1) & (poly_demo_grady == 1)) | ((poly_demo_mag_binary == 1) & (poly_demo_dir_binary == 1))] = 1       
poly_demo_combined_final = np.zeros_like((poly_demo_dir_binary))
poly_demo_combined_final[(poly_demo_combined == 1)|(poly_demo_hls_img == 1)] = 1
poly_demo_combined_final *= 255  # scale the image

poly_demo_image, poly_demo_left_fitx, poly_demo_right_fitx, poly_demo_ploty, poly_demo_left_fit, poly_demo_right_fit = fit_polynomial(poly_demo_combined_final)
poly_demo_draw_step = draw_poly_lines(poly_demo_image, left_fitx, right_fitx, ploty)
    
cv2.imwrite("output_images/poly_demo.jpg", poly_demo_draw_step)

# Run Advanced Lane finding pipeline
video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
combined_clip = clip1.fl_image(ad_lane_finding_pipeline)
combined_clip.write_videofile(video_output, audio=False)



