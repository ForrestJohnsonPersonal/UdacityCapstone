from styx_msgs.msg import TrafficLight
import statistics
import cv2 # computer vision library
#import helpers # helper functions

import random
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg# for loading in images
from  collections import Counter
import tensorflow as tf

from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm

class TLClassifier(object):    
    def __init__(self):
        #TODO load classifier
        
        # Frozen inference graph files. NOTE: change the path to where you saved the models.
        SSD_GRAPH_FILE = 'frozen_inference_graph.pb'#model/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
        RFCN_GRAPH_FILE = 'model/rfcn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
        FASTER_RCNN_GRAPH_FILE = 'model/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'

        # Colors (one for each class)
        cmap = ImageColor.colormap
#        print("Number of colors =", len(cmap))
        self.COLOR_LIST = sorted([c for c in cmap.keys()])
        detection_graph = self.load_graph(SSD_GRAPH_FILE)
        # detection_graph = self.load_graph(RFCN_GRAPH_FILE)
        # detection_graph = self.load_graph(FASTER_RCNN_GRAPH_FILE)
        self.detection_graph = detection_graph
                
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        
        self.redYellowLight_Ythreshold = 13
        self.yellowGreenLight_Ythreshold = 18#19
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        with tf.Session(graph=self.detection_graph) as sess:                
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                                feed_dict={self.image_tensor: image_np})

        #    print(classes, boxes, scores)
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.6#0.8
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            height, width, color = image.shape
            box_coords = self.to_image_coords(boxes, height, width)
            box_coordsInt = []
            for boxfloats in box_coords:
                box_coordsInt.append([int(boxfloats[0]),int(boxfloats[1]),int(boxfloats[2]),int(boxfloats[3])])

            # Each class with be represented by a differently colored box
 #           self.draw_boxes(image, box_coords, classes)

#            print(classes, scores)
            # First check only for boxes of classification = 10 (traffic light).
            index = 0
            redFound = False
            for classObj in classes:
                if classObj == 10:
                    #Then only accept them if they are at least a certain size (which reflects if they are close enough).
         #           print((abs(boxes[index][0]-boxes[index][2])*abs(boxes[index][1]-boxes[index][3])))
                    if (abs(boxes[index][0]-boxes[index][2])*abs(boxes[index][1]-boxes[index][3])) > 0.0003:#0.001:
                        #Then make sure they are within a certain mid-section of the path ahead.
                        if ((boxes[index][0] > 0.1) and (boxes[index][1] > 0.3) and (boxes[index][3] < 0.8) and (boxes[index][2] < 0.8)):
                            #Extract the image within the boxed region:
                  #          boxedPILImage = image.crop((box_coordsInt[index][1], box_coordsInt[index][0], box_coordsInt[index][3], box_coordsInt[index][2]))
                  #          boxedPILImage = boxedPILImage.resize((32, 32))
                            standardImage = self.standardize_input(image[box_coordsInt[index][0]:box_coordsInt[index][2], box_coordsInt[index][1]:box_coordsInt[index][3], :])
                            BoxedOpencvImage = cv2.cvtColor(standardImage, cv2.COLOR_BGR2HSV)#cv2.cvtColor(np.array(boxedPILImage), cv2.COLOR_BGR2HSV)
                            if self.estimate_label(BoxedOpencvImage) == [1,0,0]:#NOTE: Assuming vertical oriented stop light arrays here!!
                                redFound = True
                                return TrafficLight.RED
                                print('red light')
                                #break
  #                          else:
                                print('NOT red light')

  #                      else:
                            print('box not within midsection')
 #                   else:
                        print('box too far away')
 #               else:
                    print('box not for traffic lights')
                index += 1
        
        
        return TrafficLight.UNKNOWN

    #
    # Utility funcs
    #
    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph
    
    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def draw_boxes(self, image, boxes, classes, thickness=4):
        """Draw bounding boxes on the image"""
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            color = COLOR_LIST[class_id]
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)
    
    #
    #   Traffic Light Box Red/Yellow/Green extraction
    #
    
    # This function should take in an RGB image and return a new, standardized version
    def standardize_input(self, image):

        ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    #   standard_im = np.copy(image)
        # Resize image and pre-process so that all "standard" images are the same size  
        standard_im = cv2.resize(image, (32, 32))

        return standard_im

    ## Given a label - "red", "green", or "yellow" - return a one-hot encoded label
    def one_hot_encode(self, label):

        ## TODO: Create a one-hot encoded label that works for all classes of traffic lights
        one_hot_encoded = [] 

        if(label == 'red'):
            one_hot_encoded = [1,0,0]
        elif(label == 'yellow'):
            one_hot_encoded = [0,1,0]
        else:#must be green
            one_hot_encoded = [0,0,1]

        return one_hot_encoded

    def standardize(self, image_list):

        # Empty image data array
        standard_list = []

        # Iterate through all the image-label pairs
        for item in image_list:
            image = item[0]
            label = item[1]

            # Standardize the image
            standardized_im = self.standardize_input(image)

            # One-hot encode the label
            one_hot_label = self.one_hot_encode(label)    

            # Append the image, and it's one hot encoded label to the full, processed list of image data 
            standard_list.append((standardized_im, one_hot_label))

        return standard_list

        #We have to look at the modes in the image pixel values. If the top ones are all grouped together this should be a light,
        #or perhaps if it is just slightly broadened mode,
        #but in order to catch some edge cases we check if they are not grouped -in which case lets check V-channel and go with that
        #as all data shows it should have good contrast when S-channel does not, provided edges are cropped.
    def Check_HueSat_ModesAllgrouped(self, top_rows_idx, sum_HueS_values, axisElementsToConsider):
        top_rows_idx.sort()#arrange in increasing order.
        rowsStdaboveThreshold = self.Check_Row_MedianStd(sum_HueS_values, axisElementsToConsider, 2)
        if ((top_rows_idx[0] + len(top_rows_idx) - 1) == top_rows_idx[-1]):
            return 1
        elif (rowsStdaboveThreshold == 1):#This is broadening it to account for very blurred red light but still acceptable.
            return 1
        else:
            return 0

    ## TODO: Create a brightness feature that takes in an HSV image and outputs a feature vector and/or value
    ## This feature should use HSV colorspace values
    def create_HSVLocation_feature(self, hsvImage):

        hsvImage = hsvImage[1:-2, 5:-5, :]

        ## TODO: Create and return a feature value and/or vector
        #For this I will sum all pixels in the S-channel and get the median Y-axis location

        axisElementsToConsider = 11#7 #this will set how many of the modes we care about analyzing.
                                #From visual inspection it is relative to pixel count and observed ratio of lamp size.
                                #I could see how you could use an CNN to help find the optimal value here.

        # First check S-channel.
        #Add up all the pixel values in the S channel but just along the rows. (looked up from https://www.geeksforgeeks.org/numpy-sum-in-python/)
        sum_HueS_values = np.sum(hsvImage[:,:,1],1)#need to output an array of Hue values that are still relative to their Y-location but all column values summed for that row
        top_rows_idx = np.argsort(sum_HueS_values)[-axisElementsToConsider:]#get top number of rows
        CroppedLevel = 5#I could see how you could use an CNN to help find the optimal value here.

        if self.Check_HueSat_ModesAllgrouped(top_rows_idx, sum_HueS_values, axisElementsToConsider) == 1:
            useHueSat_orValue = 1
        else:#Switch to using V-channel
            useHueSat_orValue = 2
            hsvImage = hsvImage[:, CroppedLevel:-CroppedLevel, :]#Data visualization shows likely we can crop the vertical sides of image to avoid background summation error.
            sum_HueV_values = np.sum(hsvImage[:,:,2],1)#need to output an array of Hue values that are still relative to their Y-location but all column values summed for that row
            top_rows_idx = np.argsort(sum_HueV_values)[-axisElementsToConsider:]#get top number of rows

        col_median, recommend_V_Chan = self.Get_Hue_Xlocation(hsvImage, top_rows_idx, axisElementsToConsider, useHueSat_orValue)

        if recommend_V_Chan == 1:#Check AGAIN if we need to switch to V-channel....
            hsvImage = hsvImage[:, CroppedLevel:-CroppedLevel, :]#Data visualization shows likely we can crop the vertical sides of image to avoid background summation error.
            sum_HueV_values = np.sum(hsvImage[:,:,2],1)#need to output an array of Hue values that are still relative to their Y-location but all column values summed for that row
            top_rows_idx = np.argsort(sum_HueV_values)[-axisElementsToConsider:]#get top number of rows

        # find the median y location
        row_median = statistics.median(top_rows_idx)
        print("Top Rows = "+str(top_rows_idx))
        print("Row_Median = "+str(row_median))
        print("Col_Median = "+str(col_median))
        return row_median, col_median, CroppedLevel;

        #we use this to help determine if the peak pixel average x location, which is helpful for future feature checking on if
        #we are looking at the center of a light or at the edge for some other false positive.
    def Get_Hue_Xlocation(self, hsvImage, ptop_rows_idx, colToSample, useHueSat_orValue):
        satImage = hsvImage[:,:,useHueSat_orValue]

        #Go through and mask all pixels not in top rows we cared about.
        for row in range(0,len(satImage[0])):
            if row not in ptop_rows_idx:#check if this row number does NOT exist in our top rows list.
                for col in range(0, len(satImage[1])):
                    satImage[row][col] = 0 #set all column elements to zero.

        # Add up all the pixel values in the channel but just along the columns. (looked up from https://www.geeksforgeeks.org/numpy-sum-in-python/)
        sum_HueChan_values = np.sum(satImage,0)#need to output an array of Hue values that are still relative to their X-location but all column values summed for that row
        recommend_V_Channel = 0
        if (self.Check_Column_MedianStd(sum_HueChan_values, colToSample, 5) == 0):
            recommend_V_Channel = 1

        top_columns_idx = np.argsort(sum_HueChan_values)[-colToSample:]#get top number of columns
        print("Top Columns = "+str(top_columns_idx))
        median = statistics.median(top_columns_idx)

        return median, recommend_V_Channel;

    #got this from (https://stackoverflow.com/questions/15389768/standard-deviation-of-a-list)
    def Check_Column_MedianStd(self, sum_HueChan_values, colToSample, threshold):
        columns_sort_idx = np.argsort(sum_HueChan_values)[-(colToSample-3):]#I could see how you could use an CNN to help find the optimal value here.
        HueChan_stdev = statistics.stdev(columns_sort_idx)
        print("Col_SChan_Std = "+str(HueChan_stdev))
        if (HueChan_stdev > threshold):
            return 0
        else:
            return 1
        return 1

    #got this from (https://stackoverflow.com/questions/15389768/standard-deviation-of-a-list)
    def Check_Row_MedianStd(self, sum_HueChan_values, rowToSample, threshold):
        rows_sort_idx = np.argsort(sum_HueChan_values)[-(rowToSample-5):]#I could see how you could use an CNN to help find the optimal value here.
        HueChan_stdev = statistics.stdev(rows_sort_idx)
        print("Row_SChan_Std = "+str(HueChan_stdev))
        if (HueChan_stdev > threshold):
            return 0
        else:
            return 1
        return 1

    def Check_Is_Greencolor(self, rgb_image, xLocation, yLocation, CroppedLevel):
        rgb_image = rgb_image[3:-3,CroppedLevel:-CroppedLevel,:]
        redImage = rgb_image[:,:,0]
        greenImage = rgb_image[:,:,1]
        topredImageSum = np.sum(rgb_image[0:self.redYellowLight_Ythreshold,:,0])
        topSumAll = np.sum(rgb_image[0:self.redYellowLight_Ythreshold,:,:])#Get total RGB pixel value in top region
        bottomgreenImageSum = np.sum(rgb_image[self.yellowGreenLight_Ythreshold:-1,:,1])
        bottomSumAll = np.sum(rgb_image[self.yellowGreenLight_Ythreshold:-1,:,:])#Get total RGB pixel value in bottom region
        middleyellowImageSum = np.sum(rgb_image[self.redYellowLight_Ythreshold:self.yellowGreenLight_Ythreshold,:,0]) + np.sum(rgb_image[self.redYellowLight_Ythreshold:self.yellowGreenLight_Ythreshold,:,1]) + np.sum(rgb_image[self.redYellowLight_Ythreshold:self.yellowGreenLight_Ythreshold,:,2])
        middleSumAll = np.sum(rgb_image[self.redYellowLight_Ythreshold:self.yellowGreenLight_Ythreshold,:,:])#Get total RGB pixel value in middle region
        sumRed = np.sum(redImage)#Get total Red pixel value in image.
        sumGreen = np.sum(greenImage)#Get total Green pixel value in image.
        sumAll = np.sum(rgb_image[:,:,:])#Get total RGB pixel value.

        if (bottomgreenImageSum/bottomSumAll > topredImageSum/topSumAll):#I could see how you could use an CNN to help find the optimal value here.
            return 1
            if (middleyellowImageSum/middleSumAll > bottomgreenImageSum/bottomSumAll):
                return 2
            else: 
                return 1
        else:
            return 0
        return 0

    def Check_Is_Redcolor(self, rgb_image, xLocation, yLocation, CroppedLevel):
        redImage = rgb_image[:,CroppedLevel:-CroppedLevel,0]
        greenImage = rgb_image[:,CroppedLevel:-CroppedLevel,1]
        sumRed = np.sum(redImage)/(32*32.0)#Get average Red pixel value in image.
        print("sumRed = "+str(sumRed))
        sumGreen = np.sum(greenImage)/(32*32.0)#Get average Green pixel value in image.
        print("sumGreen = "+str(sumGreen))
        if (sumRed > (sumGreen + 8)):#I could see how you could use an CNN to help find the optimal value here.
            return 1
        else:
            return 0
        return 0

    def Check_Is_Yellowcolor(self, rgb_image, xLocation, yLocation, CroppedLevel):
        rgb_image = rgb_image[3:-3,CroppedLevel:-CroppedLevel,:]
        redImage = rgb_image[:,:,0]
        greenImage = rgb_image[:,:,1]
        topredImageSum = np.sum(rgb_image[0:self.redYellowLight_Ythreshold,:,0])
        topSumAll = np.sum(rgb_image[0:self.redYellowLight_Ythreshold,:,:])#Get total RGB pixel value in top region
        bottomgreenImageSum = np.sum(rgb_image[self.yellowGreenLight_Ythreshold:-1,:,1])
        bottomSumAll = np.sum(rgb_image[self.yellowGreenLight_Ythreshold:-1,:,:])#Get total RGB pixel value in bottom region
        middleyellowImageSum = np.sum(rgb_image[self.redYellowLight_Ythreshold:self.yellowGreenLight_Ythreshold,:,0]) + np.sum(rgb_image[self.redYellowLight_Ythreshold:self.yellowGreenLight_Ythreshold,:,1])
        middleSumBlue = np.sum(rgb_image[self.redYellowLight_Ythreshold:self.yellowGreenLight_Ythreshold,:,2])#Get total Blue pixel value in middle region
        middleSumAll = np.sum(rgb_image[self.redYellowLight_Ythreshold:self.yellowGreenLight_Ythreshold,:,0]) + np.sum(rgb_image[self.redYellowLight_Ythreshold:self.yellowGreenLight_Ythreshold,:,1]) + np.sum(rgb_image[self.redYellowLight_Ythreshold:self.yellowGreenLight_Ythreshold,:,2])
        sumRed = np.sum(redImage)#Get total Red pixel value in image.
        sumGreen = np.sum(greenImage)#Get total Green pixel value in image.
        sumAll = np.sum(rgb_image[:,:,:])#Get total RGB pixel value.
        print("topredImageSum = "+str(topredImageSum))
      #  sumGreen = np.sum(greenImage)/(32*32.0)#Get average Red pixel value in image.
        print("middleyellowImageSum = "+str(middleyellowImageSum))
        print("middleSumBlue = "+str(topredImageSum))
        #I could see how you could use an CNN to help find the optimal value here.
        if (middleyellowImageSum/middleSumAll > ((topredImageSum/topSumAll)*1.9)):
            return 1
        else:
            return 0
        return 0

    #Mask and remove background black portion of traffic light
    def create_MaskedImage_feature(self, rgb_image):
        # Define our color selection boundaries in RGB values
        lower_black = np.array([0,0,0]) 
        upper_black = np.array([150,150,250])#????????????this was just a test, then realized this won't be so easy
        # Define the masked area
        mask = cv2.inRange(rgb_image, lower_black, upper_black)

        # Mask the image
        masked_image = np.copy(rgb_image)

        masked_image[mask != 0] = [0, 0, 0]
        plt.imshow(masked_image)

        return masked_image

    # This function should take in HSV image input
    # Analyze that image using your feature creation code and output a one-hot encoded label
    def estimate_label(self, hsvImage):
    
        ## TODO: Extract feature(s) from the RGB image and use those features to
        ## classify the image and output a one-hot encoded label
        predicted_label = []

        rgb_image = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2RGB)
        featureMedianS_Xlocation, featureMedianS_Ylocation, CroppedLevel = self.create_HSVLocation_feature(hsvImage)#)rgb_image)
        #Take in the average S-value and make sure it's Y-location is within one of 3 ranges.
        if (featureMedianS_Xlocation < self.redYellowLight_Ythreshold):# hsvfeature_thresholds[0]):
                predicted_label = self.one_hot_encode('red')
                print('red')
        elif (featureMedianS_Xlocation < self.yellowGreenLight_Ythreshold):# hsvfeature_thresholds[1]):
            if self.Check_Is_Yellowcolor(rgb_image, featureMedianS_Xlocation, featureMedianS_Ylocation, CroppedLevel) == 1:
                predicted_label = self.one_hot_encode('yellow')
                print('yellow')
            else:
                predicted_label = self.one_hot_encode('red')
                print('red by default for S-value')
        else: #must be equal or greater than hsvfeature_thresholds[1]
            #first check red & green color content to be sure in that pixel
            CheckGreen = self.Check_Is_Greencolor(rgb_image, featureMedianS_Xlocation, featureMedianS_Ylocation, CroppedLevel)
            if CheckGreen == 1:
                predicted_label = self.one_hot_encode('green')
                print('green')
            elif CheckGreen == 2:
                predicted_label = self.one_hot_encode('yellow')
                print('yellow')
            else:
                predicted_label = self.one_hot_encode('red')#somehow got this backwards
                print('red by default')

        return predicted_label