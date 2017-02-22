# **Advanced Lane Detection**

The goal of this project is to detect lane lines in a video steam as accurately as possible and find the position of the vehicle on the road and the curvature of the road. This was achieved using the following steps.

#### Camera Calibration and Undistortion

I calibrated the camera using 20 images given by Udacity of a 10x7 chessboard. I extracted object points and image points from all the 20 images and fed them to the `calibrateCamera()` of OpenCV to get calibration and distortion matrices. Then I can apply the `undistort()` function to undistort every image.

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pnts, img_pnts, gray.shape[::-1],None,None)

def undistort(image, matrix, distMatrix):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.undistort(image, matrix, distMatrix)
```

![](https://github.com/muddassir235/Advanced-Lane-Detection/blob/master/output_images/undistorted.png?raw=true)

#### Perspective Transform
Using OpenCV's `getPerspectiveTransfrom()` function I extract the Perspective transform matrix and then using the `warpPerspective()` function I get a birds-eye view of the road.

```python
def warp(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)
    return warped, Minv
```

![](https://github.com/muddassir235/Advanced-Lane-Detection/blob/master/output_images/warp.png?raw=true)

#### Emphasizing Lane Lines
Using a combination of HSV and Gradient Thresholds we extract a binary image which emphasizes the lane lines from the warped image.

```python
def get_lanes(img):
    b_hs_yellow = get_hs_thresholded(img=img, h_thresh=(0,40), s_thresh=(80,255), v_thresh=(200,255))
    b_hs_white = get_hs_thresholded(img=img,h_thresh=(0,255), s_thresh=(0,40), v_thresh=(220,255))

    b = np.zeros_like(b_hs_yellow)

    b_grad = get_grad_thresh(img)

    b[(b_hs_white == 1)|(b_hs_yellow == 1)|(b_grad == 1)] = 1

    return b
```

![](https://github.com/muddassir235/Advanced-Lane-Detection/blob/master/output_images/binary_warped.png?raw=true)

#### Detecting the Lane Pixels
I detect the lane pixels using the [method](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/c41a4b6b-9e57-44e6-9df9-7e4e74a1a49a) given by Udacity. Using Histogram on vertical levels of the image, I find the maximum values in the Histogram and define a region around them and consider those pixels to be lanes.

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588cef47_screen-shot-2017-01-28-at-11.21.09-am/screen-shot-2017-01-28-at-11.21.09-am.png)

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588cf5e0_screen-shot-2017-01-28-at-11.49.20-am/screen-shot-2017-01-28-at-11.49.20-am.png)

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(binary_warped.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
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
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

Then I fit a polynomial to the points found and plot it.

```python
# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
```

#### Find Curvature of the lanes
If the curvature of the line is defined by

```python
A*y**2 + B*y + C
```

then, I can find the curvature of the lanes, using the following formula:

```python
Radius = (1 + (2*A*y + B)**2)**(3/2) / abs(2*A)
```

So using the following code I can find the curvature of the lane in meters.

```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
print(left_curverad, 'm', right_curverad, 'm')
# Example values: 632.1 m    626.2 m
```



#### Warp the lanes back on to the image
I warp back the image to the Original one using the `Minv` matrix which we had gotten from the `getPerspectiveTransfrom()` function.

```python
# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))         
```

![](https://github.com/muddassir235/Advanced-Lane-Detection/blob/master/output_images/pipeline.png?raw=true)

#### Plotting Curvature and lane distance on Resultant Image
I plot the distance from the left and right lane as well as the curvature on to the Resultant image as follows:

```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720.0 # meters per pixel in y dimension
xm_per_pix = 3.7/700.0 # meters per pixel in x dimension

# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

left_pos = (left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2])
right_pos = (right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2])

lanes_mid = (left_pos+right_pos)/2.0

distance_from_mid = binary_warped.shape[1]/2.0 - lanes_mid

mid_dist_m = xm_per_pix*distance_from_mid

curvature = 'Radius: '+ str(left_curverad) + ' m, ' + str(right_curverad) + " m"
lane_dist = 'Distance From Road Center: '+str(mid_dist_m)+' m'
font = cv2.FONT_HERSHEY_SIMPLEX
result = cv2.putText(result,curvature,(25,50), font, 1, (255,255,255),2,cv2.LINE_AA)
result = cv2.putText(result,lane_dist,(25,100), font, 1, (255,255,255),2,cv2.LINE_AA)
```
![](https://github.com/muddassir235/Advanced-Lane-Detection/blob/master/output_images/with_position.PNG?raw=true)

## Result
We got the following result after applying the pipeline to the `project_video.mp4`

![](https://github.com/muddassir235/Advanced-Lane-Detection/blob/master/output_images/ezgif.com-video-to-gif.gif?raw=true)

#### Discussion

* I had a really hard time figuring out the right function for getting a binary image which emphasizes the lane lines.
* Also I think my lane finding pipeline can be improved greatly by Improving the stage where the lane lines are extracted from the binary image using a histogram.
* Also another improvement would be averaging with lane line from previous frames in order to get a smoother fit and reduce jitter.
