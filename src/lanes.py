import cv2
import numpy as np
import os
import sys

def make_coordinates(image, line_parameters):
    """ Finds the x and y coordinates of the line
    """
    slope, intercept = line_parameters
    height = image.shape[0]
    width = image.shape[1]
    
    y1 = height
    y2 = int(y1*0.6)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    
    if x1 and x2 <= width and y1 and y2 <= height:
        return np.array([x1, y1, x2, y2])

def averaged_slope_intercept(image, lines):
    """ Finds the average of the left and right lane lines 
    obtained from the Hough Transform
    """
    left_fit = []
    right_fit = []
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        if slope or intercept is not None:
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept)) 
    
    if len(left_fit) and len(right_fit) is not 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
        
        return np.array([left_line, right_line])
    
def canny(image):
    """ Takes in the clean image and apply canny filter for image processing
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    
    return canny

def display_lines(image, lines):
    """ Displays the lines obtained from make_coordinates in a line_image for final image processing
    """
    line_image = np.zeros_like(image)
    
    if lines is not None:
        print(lines)
        for x1, y1, x2, y2 in lines:
            print(x1, y1, x2, y2)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    return line_image

def region_of_interest(image):
    """ Identify region of interest for lane detection
    """
    height, width = image.shape
    polygons = np.array([[(150, height), (610, height), (350, 170)]])
    # polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def lane_detector(image):
    """ Lane detector wrapper function 
    """
    
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)

    threshold = 75
    minLineLength = 15
    maxLineGap = 0

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    if lines is not None:
        averaged_lines = averaged_slope_intercept(lane_image, lines)
        line_image = display_lines(lane_image, averaged_lines)
        final_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        
        cv2.imshow("Result", final_image)
        cv2.waitKey(0)

""" 
def video_detector(filename):
    
    cap = cv2.VideoCapture(filename)
    
    while(cap.isOpened()):
        _, frame = cap.read()
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = averaged_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result", final_image)
        
        if cv2.waitKey(10) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
"""     
if __name__ == "__main__" :
      
    # filename = "images/test2.jpg"
    filename = "videos/videoplayback.mp4"
    
    if ".mp4" in filename:
        print("entered video")
        cap = cv2.VideoCapture(filename)
        while(cap.isOpened()):
            _, frame = cap.read()
            lane_detector(frame)
            if cv2.waitKey(10) == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()

    elif ".jpg" in filename:
        print("entered iamge")
        image = cv2.imread(filename)
        lane_detector(image)