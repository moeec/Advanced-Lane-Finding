##  Mwesigwa Musisi-Nkambwe Writeup For Advanced Lane Finding Project

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


All code for the project can be found in:

Advanced_Lane_Finding.ipynb - Advanced Lane Finding Code
Chess_Board.ipynb - Chess Board undistortion that displays files (takes up quite alot memory)
Chess_Board_Save_File.ipynb - Chess Board undistortion that saves files
Perspective_Transform.ipynb - Perspective Transform Demonstration
Poly_Demo.ipynb - 2nd Order Polynomial demonstration
Poly_Demo2.ipynb - 2nd Order Polynomial demonstration with perspective transform
Thresholded_Binary_Image.ipynb - Combined Threshold Binary demonstration

[//]: # (Image References)

[image1]: ./output_images/calibration1_output.jpg "Undistorted Calibration Image 1"
[image2]: ./output_images/calibration2_output.jpg "Undistorted Calibration Image 2"
[image3]: ./output_images/calibration3_output.jpg "Undistorted Calibration Image 3"
[image4]: ./output_images/calibration4_output.jpg "Undistorted Calibration Image 4"
[image5]: ./output_images/calibration5_output.jpg "Undistorted Calibration Image 5"
[image6]: ./output_images/calibration6_output.jpg "Undistorted Calibration Image 6"
[image7]: ./output_images/calibration7_output.jpg "Undistorted Calibration Image 7"
[image8]: ./output_images/calibration8_output.jpg "Undistorted Calibration Image 8"
[image9]: ./output_images/calibration9_output.jpg "Undistorted Calibration Image 9"
[image10]: ./output_images/calibration10_output.jpg "Undistorted Calibration Image 10"
[image11]: ./output_images/calibration11_output.jpg "Undistorted Calibration Image 11"
[image12]: ./output_images/calibration12_output.jpg "Undistorted Calibration Image 12"
[image13]: ./output_images/calibration13_output.jpg "Undistorted Calibration Image 13"
[image14]: ./output_images/calibration14_output.jpg "Undistorted Calibration Image 14"
[image15]: ./output_images/calibration15_output.jpg "Undistorted Calibration Image 15"
[image16]: ./output_images/calibration16_output.jpg "Undistorted Calibration Image 16"
[image17]: ./output_images/calibration17_output.jpg "Undistorted Calibration Image 17"
[image18]: ./output_images/calibration18_output.jpg "Undistorted Calibration Image 18"
[image19]: ./output_images/calibration19_output.jpg "Undistorted Calibration Image 19"
[image20]: ./output_images/calibration20_output.jpg "Undistorted Calibration Image 20"
[image21]: ./output_images/straight_lines1_output.jpg "Perspective Transform Image 21"
[image22]: ./output_images/straight_lines2_output.jpg "Perspective Transform Image 22"
[image23]: ./output_images/test1_output.jpg "Perspective Transform Image 23"
[image24]: ./output_images/test2_output.jpg "Perspective Transform Image 24"
[image25]: ./output_images/test3_output.jpg "Perspective Transform Image 25"
[image26]: ./output_images/test4_output.jpg "Perspective Transform Image 26"
[image26]: ./output_images/test5_output.jpg "Perspective Transform Image 27"
[image26]: ./output_images/test6_output.jpg "Perspective Transform Image 28"



[video1]: ./project_video.mp4 "Original Video"
[video1]: ./project_video_output.mp4 "Processed Output Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

**Camera Calibration**

Briefly state I computed the camera matrix and distortion coefficients.
Provide an example of a distortion corrected calibration image.

**Pipeline (test images)**

Provide an example of a distortion-corrected image.
 
Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

Demonstrate the combination of methods (i.e., color transforms, gradients) used to create a binary image containing likely lane pixels with visual verification

Describe how (and identify where in your code) you performed a perspective transform ("birds-eye view").
Provide an example of a transformed image.

Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Methods have been used to identify lane line pixels in the rectified binary image. The left and right line have been identified and fit with a curved functional form (e.g., spine or polynomial). 
Example images with line pixels identified and a fit overplotted are included.

Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

An example image of my result plotted back down in on the road in which the lane area is identified clearly.

The fit from the rectified image was warped back onto the original image and plotted to identify the lane boundaries. 
This demonstrates that the lane boundaries were correctly identified. 
An example image with lanes, curvature, and position from center is included above

**Pipeline (video)**

The link to the final video output can be found above. 
My pipeline performs reasonably well on the entire project video
The output here is a new video where the lanes are identified in every frame, and outputs are generated regarding the radius of curvature of the lane and vehicle position within the lane. The pipeline correctly maps out curved lines and overcomes fails when shadows or pavement color changes are present.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

Found Below.


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

**Camera Calibration**

I start by first understanding what the dimensions are of the chessboard, these are cbrow and cbcol that are easily editable in the code. The chessboard given in this project is a 9 X 6 board. I load in 20 images of the chessboard at different angles and distances found in the /camera_cal folder.
I collect "object points" & "image points", of the chessboard made up of (x, y, z) coordinates of the chessboard corners in the real world. An assumption of the chessboard having an (x,y) plane and z = 0 is made. It  is assumed that all 20 calibration images are of the same board. I collect all the "object points" & "image points" in two respective arrays called "objpoints" & "imgpoints".
I then map the corners of the 2D image "Image Points" with the 3D corners of the real undistorted chessboard corners called "Object Points" 

The "Object points" will be the known object co-ordintes of the chessboard corners for an 9 x 6 board. These points will be 3d co-ordinates X Y Z and Z from the top left corner (0,0,0) to the bottom right corner (7,10,0).



![Example Output_Image][/output_images/calibration1_output.jpg "Output Image"]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.


In my code in file Chess_Board I display Images of Before & After Correction &  save image files with the result.
I used the objpoints and imgpoints collected above to compute the camera calibration and distortion coefficients. With these coefficients I used the function I created "cal_undistort" to undistort the images. 
The "cal_undistort"  uses OpenCV's module cv2.calibrateCamera(). 
The output from the function can be found in the /output_images folders, this is image1 - image 20 above.

![Calibration Image](/output_images/calibration1_output.jpg)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary images that can be found at:

![Combined Binary Output_Image][/output_images/straight_lines1_binary_output.jpg"]
![Combined Binary Output_Image][/output_images/straight_lines2_binary_output.jpg"]

(thresholding steps at lines #67 through #78 in `Thresholded_Binary_Image.ipynb`).

Here I am selecting pixels where both the xx and yy gradients meet the threshold criteria, or the gradient magnitude and direction are both within their threshold values. Here I have taken gradient measurements (x, y, magnitude, direction) to isolate lane-line pixels.I have used the thresholds of the x and y gradients, the overall gradient magnitude, and the gradient direction to focus on pixels that are parts of the lane lines.
I have used Sobel X & Y, Magnitude and Directional Gradients to achieve this.


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

A perspective transform  is an image such that we can effectively change the view we have of objects from a different angle or direction. To get a perspective transform ("Bird Eyeview" in this case) I used the a function called `warp()`, this can be found in my code `Perspective_Transform.ipynb` at lines 17 through 43.

I had computed the perspective transform, M, given source and destination points.
Below is a code snippet of  the code used to perform this task,  Open CV module's cv2.getPerspectiveTransform is used.

```def warp(img):

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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 1127, 720     | 960, 720      |
| 200, 720      | 320, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Post Perspective Transform][output_images/straight_lines1_dst_src_output.jpg]
![Post Perspective Transform][test_images/straight_lines1_dst_src.jpg]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Once calibration, thresholding, and a perspective transform are performed on the road image, I have a black & white binary image where the lane lines are easily visible. By plotting a histogram of where the across the bottom half of image I am able to identify where the lane marking are. The histogram adds up the pixel values along each column in the image. Pixels in the image are either  0 or 1, making the areas where there are peaks good indications of the x-position at the base of the lane lines. This will be the starting point the code's search for the lines. I then construct a sliding window found around the lines centers, this sliding window slide up from the bottom to the top of the image for the right and left lane lines are respectively. 
A line is drawn over the detected pixels in the sliding windows and fit to a 2nd order polynomial using np.polyfit this can found at lines #197 through #198.

The output image of this can be found at:
 
![Polynomial fit Image][/output_images/poly_demo.jpg]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Next I computed the radius of curvature of the fit using f(y)=Ay^2+By+C that was calcuated when we fit the second order polynomial.
This equation with radius of a curvature found at:

[Radius of a Curvature](https://www.intmath.com/applications-differentiation/8-radius-curvature.php)

The code to do this 

Once the radius of curvature based on pixel values is calculated I move onto converting this to real world space. The same calculation is repeated after converting our x and y values to real world dimensions.

By measuring length and width of the section of lane in the warped image and applying U.S. regulations (http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC) that require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each we can calculate realworld values. The code for this can be found at lines #333 through #351.

The position of the car with respect to center can also be calculated using same method above. I use my `measure_position_meters()` function to do this. With y pixels equal to 30/720 meters per pixel and x pixels equal to 3.7/700 meters per.
In the code I located the left and right line positions at the bottom of the image by , I then took average


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![Road Plot Back][output_images/poly_demo2.jpg]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

In the video link above you will find the final output of the Advanced lane Finding Project. the code used to perform this task is `Advanced_Lane_Finding.ipynb`, this can be found in the root of my jupyter folder.  
The original video clip project_video.mp4 is subjected to the ad_pipeline 1 - 6.
The pipeline function `ad_lane_finding_pipeline()` can be found at lines #333 through #610.
A video is but a series of images so by using the moviepy.editor import from VideoFileClip I am able to have each frame be put thru the pipeline above much like the way images were processed above.
The pipeline is:

1. A Perspective Transform is performed on the frame using `warp()` function found at lines #382 through #407.
2. Perspective Transform of Image processed thru Sobel, Magnitude and Directional thresholds/different color spaces, this can be found at lines #530 through #535.
3. Above thresholds are combined, this  can be found at lines #537 through #541
4(First Frame or lines not detected in previous frame). If this is the first frame, or no lane lines were detected in the previous frame the output image from #4 is then passed to the `fit_polynomial()` function found at lines #272 through #301. In this function lane lines are detected and a 2nd order polynomial is fit to the lines. The `fit_polynomial()` function calls the `find_lane_pixels()` that can be found at lines #181 through #269. 
5. A polygon is then drawn to fit the lane using `draw_poly_lines()`function found at lines #483 through #504.
6. A inverse perspective transform is then perform on the output of #5 using `unwarp()` function found at lines #410 through #435. This returns us from a "Bird's Eyeview" to view similar to the original frame. 
7. The original image is then blended with the image from #6 using `cv2.addWeighted()` from the OpenCV real-time optimized Computer Vision library. This gives us an image similar to the original image with lane regions colored green.
8. The road curvature is then calculated with `measure_curvature_pixels()` function found at lines #437 through #455. Here the average of the curvature of the left and right lane lines is taken into consideration.
9. Next the position of the car in relation to the center is calculated using the `measure_position_meters()` function found at lines #458 through #481.
10. The infomation collected in #8 & #9 is then displayed on the image using `display_curvature_and_car_pos_info()` function found at lines #506 through #520.
11. The image from #10 is then returned from the `ad_lane_finding_pipeline()` function
12. This image/frame is then combined with previous frames then saved to "project_video_output.mp4" code for this can be found at line #674.

While the above process is going on I use class Line with variable track_lines to keep track of pixels detected, left and right fit data history and frame count.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach in this project is based on a pipeline (above) that involved alot of matrix math using NumPy, matrix math is performed on images and combined thresholds using OpenCV real-time optimized Computer Vision library. I had originally just combined gradient & magnitude thresholds to find lane lines and had trouble with the parts of 
the video where there is a lightly colored pavement to the left of the image and where there are shadows formed by trees. By combinging my initial binary images with a sobel threshold I was able to detect lanes in these tricky areas. For car position I was able to get the lowest left and right lane activated pixels in each image, average this out then measure this from the center.

There were moments when data was not of a specific type or did not conform to requirements needed to perform NumPy or OpenCV functions, through research, input from mentors and perseverance I was able to complete the task.
I would think the pipeline might fail in rain due to the droplets from the sky creating "noise" in the image making it harder to detect lane lines.
Having a faster computer or just cutting down video in smaller segments when testing the code could have speeded up the process. For instance I could have just cut teh clip to areas where I was encountering issues, then tweak my code with just that video segment. 
