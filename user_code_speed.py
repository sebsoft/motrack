import tflite_runtime.interpreter as tflite
import sys
import os
from timeit import default_timer as timer
from datetime import datetime
import cv2
import numpy as np
import logging



PIXELS_METER_R2L = 0.01272



def userMotionCode(filename1,filename2, tracktime):

    # Function to read labels from text files.
    def ReadLabelFile(file_path):
      
      with open(file_path, 'r') as f:
        lines = f.readlines()
      ret = {}
      for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
      return ret
    

    def init_tf(model, label):
    
      global interpreter
      global floating_model
      global input_details
      global output_details
      global labels
      global width
      global height
      runs = 1

      labels = ReadLabelFile(label)
    
      # Load TFLite model and allocate tensors.
      interpreter = tflite.Interpreter(model_path='models/mobilenet_v1.tflite' , num_threads=4)

      #interpreter = tf.lite.Interpreter(model_path=model)
      interpreter.allocate_tensors()


      # Suppose you're interested in the following classes:
      #interested_classes = [3, 17, 32]

      #for i in range(len(interested_classes)):
      #  class_index = interested_classes[i]
      #  output_index = class_index - 1  # Class labels in MobileNetV1 are 1-indexed
      #  output_scores = interpreter.tensor(output_details[0]['index'])()
      #  output_scores[0][output_index] = 0.0      # Get input and output tensors.


      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()
      height = input_details[0]['shape'][1]
      width = input_details[0]['shape'][2]
      floating_model = False
        
      if input_details[0]['dtype'] == np.float32:
            floating_model = True

      return 
    
    def run_inference_tf(filename1, filename2 ,tracktime, runs):

       
       

       #process first image
       image1 = cv2.imread(filename1)
       
       initial_h, initial_w, channels = image1.shape
       frame = cv2.resize(image1, (width, height))
       # add N dim
       input_data = np.expand_dims(frame, axis=0)
       
       if floating_model:
          input_data = (np.float32(input_data) - 127.5) / 127.5
       
       
       interpreter.set_tensor(input_details[0]['index'], input_data)
       interpreter.invoke()
          
       detected_boxes_1 = interpreter.get_tensor(output_details[0]['index'])
       detected_classes_1 = interpreter.get_tensor(output_details[1]['index'])
       detected_scores_1 = interpreter.get_tensor(output_details[2]['index'])
       num_boxes_1 = interpreter.get_tensor(output_details[3]['index'])

       

       #process second image
       image2 = cv2.imread(filename2)
       initial_h, initial_w, channels = image2.shape
       frame = cv2.resize(image2, (width, height))
       # add N dim
       input_data = np.expand_dims(frame, axis=0)
       
       if floating_model:
          input_data = (np.float32(input_data) - 127.5) / 127.5
       
       
       interpreter.set_tensor(input_details[0]['index'], input_data)
       interpreter.invoke()
            
       detected_boxes_2 = interpreter.get_tensor(output_details[0]['index'])
       detected_classes_2 = interpreter.get_tensor(output_details[1]['index'])
       detected_scores_2 = interpreter.get_tensor(output_details[2]['index'])
       num_boxes_2 = interpreter.get_tensor(output_details[3]['index'])
       
       font = cv2.FONT_HERSHEY_SIMPLEX
       fontScale = 0.5
       red = (0, 0, 255)
       blue = (255, 0, 0)
       black = (0,0, 0)
       white = (255,255,255)


       #proces results of first image
       
       carfound1 = False
       limit = 0.4
       #draw a box around the detection area
       boxX = [1, 400, 1799, 220]
       #image1 = cv2.rectangle(image1, (boxX[0],boxX[1]), (boxX[2],boxX[3]), blue, 1)
       center1 = (0,0)

       for i in range(int(num_boxes_1)):
          center1 = (0,0)
          top, left, bottom, right = detected_boxes_1[0][i]
          classId = int(detected_classes_1[0][i])
          score = detected_scores_1[0][i]
          if score > limit:
              xmin = int(left * initial_w)
              ymin = int(bottom * initial_h)
              xmax = int(right * initial_w)
              ymax = int(top * initial_h)
              length = round((xmax-xmin) * PIXELS_METER_R2L , 1 )
              size = (xmax-xmin)*(ymin-ymax)
              box1 = [xmin, ymin, xmax, ymax]
              center1 = (int((box1[0] + box1[2]) /2)  ,int((box1[1]+ box1[3])/2))
              logging.info("startimage contains %s  score = %i size = %s ",
                        labels[classId], score*100, size )     
  
              # check if object is on the road and has car size
              if center1[1] < 400 and center1[1] > 220 and size > 20000 :  
                    carfound1 = True
                    logging.info("AOI startimage contains %s  score = %i box = %s ",
                        labels[classId], score*100, box1 )                    
                    #draw the position in image 1 and 2
                    speed = 0
                    image1 = cv2.rectangle(image1, (box1[0],box1[1]), (box1[2],box1[3]), red, 1)
                    image2 = cv2.rectangle(image2, (box1[0],box1[1]), (box1[2],box1[3]), red, 1)
                    text = labels[classId] + " " + str(length) + " m"  
                    text_size, _ = cv2.getTextSize(text, font, fontScale,1)
                    text_w, text_h = text_size
                    x = box1[0] + 5
                    y = box1[1] + 15
                    image1 = cv2.rectangle(image1, (x,y +5), (x + text_w, y - text_h - 5), black, -1)
                    image1 = cv2.putText(image1, text,(x,y) , font, fontScale, white, 1, cv2.LINE_AA)
                    carcenter1 = center1
                    carsize1 = size
                    
                    
       current_datetime = datetime.now()
       text = current_datetime.strftime("%d/%m/%Y, %H:%M:%S")
       text_size, _ = cv2.getTextSize(text, font, fontScale,1)
       text_w, text_h = text_size
       x = 5
       y = 15
       image1 = cv2.rectangle(image1, (x,y +5), (x + text_w, y - text_h - 5), black, -1)
       image1 = cv2.putText(image1, text,(x,y) , font, fontScale, white, 1, cv2.LINE_AA)
       image2 = cv2.rectangle(image2, (x,y +5), (x + text_w, y - text_h - 5), black, -1)
       image2 = cv2.putText(image2, text,(x,y) , font, fontScale, white, 1, cv2.LINE_AA)
       
       cv2.imwrite(filename1,image1)             
       
       center2 =(0,0)
       
       filename2_orig = filename2
       #process result of second image
       for i in range(int(num_boxes_2)):
          center2 = (0,0)
          top, left, bottom, right = detected_boxes_2[0][i]
          classId = int(detected_classes_2[0][i])
          score = detected_scores_2[0][i]
          if score > limit:
              xmin = int(left * initial_w)
              ymin = int(bottom * initial_h)
              xmax = int(right * initial_w)
              ymax = int(top * initial_h)
              length = round((xmax-xmin) * PIXELS_METER_R2L , 1 )
              size = (xmax-xmin)*(ymin-ymax)
              box2 = [xmin, ymin, xmax, ymax]
              center2 = (int((box2[0] + box2[2]) /2)  ,int((box2[1]+ box2[3])/2)) 

              logging.info("stopimage contains %s  score = %i size = %s ",
                        labels[classId], score*100, size )     
              
              # check if object is on the road
              if center2[1] < 400 and center2[1] > 220 and size > 20000   :
				  
                    logging.info("AOI stopimage contains %s  score = %i box = %s ",
                        labels[classId], score*100, box2 )
                    #box around detected vehicle
                    image2 = cv2.rectangle(image2, (box2[0],box2[1]), (box2[2],box2[3]), red, 1)
                    speed2 = 0
                    # a car should be found in the first image
                    # the center of the object in the second image should be left of the first image
                    # the centers should be horizontally aligned within some limits

                    #logging.info("Carcenter1 %s ", carcenter1)
                    #logging.info("center2 %s ", center2)
                    #logging.info("Dif %s ", abs(carcenter1[1]-center2[1]))
                    if carfound1==True and carcenter1[0] > center2[0] and abs(carcenter1[1]-center2[1]) < 100 :
                        
                        #line connecting boundingbox centers first and second image
                        image2 = cv2.line(image2, carcenter1, center2, blue, 1)
                        
                        # put speed text in image
                        speed2 =  round((((carcenter1[0]-center2[0]) * PIXELS_METER_R2L) / (tracktime/1000)) * 3.6 , 0) 
                        text = labels[classId] + " " + str(length) + " m" + " " + str(int(speed2)) + " km/h "
                        text_size, _ = cv2.getTextSize(text, font, fontScale,1)
                        text_w, text_h = text_size
                        x = box2[0] + 5
                        y = box2[1] + 15
                        image2 = cv2.rectangle(image2, (x,y +5), (x + text_w, y - text_h - 5), black, -1)
                        image2 = cv2.putText(image2, text,(x,y) , font, fontScale, white, 1, cv2.LINE_AA)
                        
                        logging.info("Track Speed: Length %i  in %i ms  speed %i km",
                                      (center1[0]-center2[0]), tracktime,  speed2)

                        filename2 = filename2.replace(".jpg","")
                        
                        filename2 = filename2 + "_" + labels[classId] + "_" + str(length) + "_" + str(int(speed2)) + "_.jpg"
                        #print("file path is %s" % output)
                        #cv2.imwrite(output,image2)

                    else:
                        #no car found in first image - not able to calculate speed
                        speed2 = 0
                        text = labels[classId] + " " + str(length) + " m" + " "
                        text_size, _ = cv2.getTextSize(text, font, fontScale,1)
                        text_w, text_h = text_size
                        x = box2[0] + 5
                        y = box2[1] + 15
                        image2 = cv2.rectangle(image2, (x,y +5), (x + text_w, y - text_h - 5), black, -1)
                        image2 = cv2.putText(image2, text,(x,y) , font, fontScale, white, 1, cv2.LINE_AA)

                        filename2 = filename2.replace(".jpg","")
                        
                        #output = output.replace("Start","Result")
                        filename2 = filename2 + "_" + labels[classId] + "_" + str(length) + "_.jpg"
                        #print("file path is %s" % output)
                        #cv2.imwrite(output,image2)
                         
                    
                    #construct filename
				
              cv2.imwrite(filename2,image2)    
                    
              #else:
                    #draw = cv2.putText(draw, labels[classId],(box[0] + 10, box[1] - 20) , font, fontScale, red, 1, cv2.LINE_AA)
                    #draw = cv2.rectangle(draw, (box[0],box[1]), (box[2],box[3]), red, 1)
       
       
       
       #os.remove(filename1)
       os.remove(filename2_orig)
       #print("file path is %s" % output)
       
       
       
       end = timer()
       #print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
       
    
    


  #------------------------------------------------------------------------------    
    
    init_tf('models/mobilenet_v1.tflite','models/coco_labels.txt')
    result = run_inference_tf(filename1, filename2,tracktime , 1) 
