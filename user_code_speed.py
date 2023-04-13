import tflite_runtime.interpreter as tflite
import sys
import os
from timeit import default_timer as timer
import cv2
import numpy as np
import logging



PIXELS_METER_R2L = 0.0156


def userMotionCode(filename1,filename2, tracktime, speed):

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

      # Get input and output tensors.
      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()
      height = input_details[0]['shape'][1]
      width = input_details[0]['shape'][2]
      floating_model = False
        
      if input_details[0]['dtype'] == np.float32:
            floating_model = True

      return 
    
    def run_inference_tf(filename1, filename2 ,tracktime, speed, runs):
       
       #image 2 is the last image
       output = filename1
       image1 = cv2.imread(filename1)
       
       initial_h, initial_w, channels = image1.shape
       frame = cv2.resize(image1, (width, height))
       # add N dim
       input_data = np.expand_dims(frame, axis=0)
       
       if floating_model:
          input_data = (np.float32(input_data) - 127.5) / 127.5
       
       #interpreter.set_num_threads(4)
       interpreter.set_tensor(input_details[0]['index'], input_data)
       
       
       #  Start synchronous inference and get inference result
       # Run inference.
           
       if runs == 1:
          start = timer()
          interpreter.invoke()
          end = timer()
          logging.info("Running inferencing for  %i  times. Elapsed time is %i ms",
                        runs, ((end - start)/runs)*1000 )
       else:
          start = timer()
          print('Initial run, discarding.')
          interpreter.invoke()
          end = timer()
          print('First run time is ', (end - start)*1000, 'ms')
          start = timer()
          for i in range(runs):
             interpreter.invoke()
          
       detected_boxes_1 = interpreter.get_tensor(output_details[0]['index'])
       detected_classes_1 = interpreter.get_tensor(output_details[1]['index'])
       detected_scores_1 = interpreter.get_tensor(output_details[2]['index'])
       num_boxes_1 = interpreter.get_tensor(output_details[3]['index'])

       image2 = cv2.imread(filename2)
       
       initial_h, initial_w, channels = image2.shape
       frame = cv2.resize(image2, (width, height))
       # add N dim
       input_data = np.expand_dims(frame, axis=0)
       
       if floating_model:
          input_data = (np.float32(input_data) - 127.5) / 127.5
       
       #interpreter.set_num_threads(4)
       interpreter.set_tensor(input_details[0]['index'], input_data)
       
       #  Start synchronous inference and get inference result
       # Run inference.
       
           
       if runs == 1:
          start = timer()
          interpreter.invoke()
          end = timer()
          
          logging.info("Running inferencing for  %i  times. Elapsed time is %i ms",
                        runs, ((end - start)/runs)*1000 )
       else:
          start = timer()
          print('Initial run, discarding.')
          interpreter.invoke()
          end = timer()
          print('First run time is ', (end - start)*1000, 'ms')
          start = timer()
          for i in range(runs):
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


       center1 = (0,0)
       carfound1 = False
       for i in range(int(num_boxes_1)):
          top, left, bottom, right = detected_boxes_1[0][i]
          classId = int(detected_classes_1[0][i])
          score = detected_scores_1[0][i]
          if score > 0.5:
              xmin = int(left * initial_w)
              ymin = int(bottom * initial_h)
              xmax = int(right * initial_w)
              ymax = int(top * initial_h)
              length = round((xmax-xmin) * PIXELS_METER_R2L , 1 )
              size = (xmax-xmin)*(ymin-ymax)
              logging.info("image 1 contains %s  score = %i size = %s ",
                        labels[classId], score*100, size )     
  
              #print(labels[classId], 'score = ', score)
              
              # check if object is on the road and has car size
              if ymin < 400 and ymin > 150 and size > 10000 :  
                    carfound1 = True
                    box1 = [xmin, ymin, xmax, ymax] 
                    logging.info("sel image 1 contains %s  score = %i box = %s ",
                        labels[classId], score*100, box1 )                    
                    crop = image1[box1[3]:box1[1],box1[0]:box1[2]]
                    center1 = (int((box1[0] + box1[2]) /2)  ,int((box1[1]+ box1[3])/2))
                    #image2 = cv2.rectangle(image2, (box1[0],box1[1]), (box1[2],box1[3]), red, 1)
       
       
       for i in range(int(num_boxes_2)):
          top, left, bottom, right = detected_boxes_2[0][i]
          classId = int(detected_classes_2[0][i])
          score = detected_scores_2[0][i]
          if score > 0.5:
              xmin = int(left * initial_w)
              ymin = int(bottom * initial_h)
              xmax = int(right * initial_w)
              ymax = int(top * initial_h)
              length = round((xmax-xmin) * PIXELS_METER_R2L , 1 )
              size = (xmax-xmin)*(ymin-ymax)

              logging.info("image 2 contains %s  score = %i size = %s ",
                        labels[classId], score*100, size )     
              
              # check if object is on the road
              if ymin < 400 and ymin > 150 and size > 10000 :  
                    
                    box2 = [xmin, ymin, xmax, ymax]
                    logging.info("sel image 2 contains %s  score = %i box = %s ",
                        labels[classId], score*100, box2 )
                    
                    center2 = (int((box2[0] + box2[2]) /2)  ,int((box2[1]+ box2[3])/2)) 
                    image2 = cv2.rectangle(image2, (box2[0],box2[1]), (box2[2],box2[3]), red, 1)
                    speed2 = 0
                    if (carfound1==True):
                        #image2[box1[3]:(box1[3]+crop.shape[0]),box1[0]:box1[0]+crop.shape[1]] = crop
                        image2 = cv2.rectangle(image2, (box1[0],box1[1]), (box1[2],box1[3]), red, 1)
                        
                        speed2 = round((((center1[0]-center2[0]) * PIXELS_METER_R2L) / (tracktime/1000)) * 3.6 , 0) 
                        image2 = cv2.line(image2, center1, center2, blue, 1)
                        # put speed text in image
                        text = labels[classId] + " " + str(length) + " m" + " " + str(int(speed2)) + " km/h "
                        text_size, _ = cv2.getTextSize(text, font, fontScale,1)
                        text_w, text_h = text_size
                        x = box2[0] + 5
                        y = box2[1] + 15
                        image2 = cv2.rectangle(image2, (x,y +5), (x + text_w, y - text_h - 5), black, -1)
                        
                        image2 = cv2.putText(image2, text,(x,y) , font, fontScale, white, 1, cv2.LINE_AA)
                        
                        logging.info("Track Speed: Length %i  in %i ms  speed %i km",
                                      (center1[0]-center2[0]), tracktime,  speed2)
                    #else:
                        #image2 = cv2.putText(image2, labels[classId] + " " + str(length) + " m" + " " + str(int(speed)) + " km/h" ,(box2[0] + 10, box2[1] - 20) , font, fontScale, red, 1, cv2.LINE_AA)
                        
                    
                    #construct filename
                    output = output.replace(".jpg","")
                    output = output + "_" + labels[classId] + "_" + str(length) + "_" + str(int(speed2)) + "_.jpg"
                    
              #else:
                    #draw = cv2.putText(draw, labels[classId],(box[0] + 10, box[1] - 20) , font, fontScale, red, 1, cv2.LINE_AA)
                    #draw = cv2.rectangle(draw, (box[0],box[1]), (box[2],box[3]), red, 1)
       
       
       
       os.remove(filename1)
       os.remove(filename2)
       #print("file path is %s" % output)
       
       cv2.imwrite(output,image2)
       
       end = timer()
       #print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
       
    
    


  #------------------------------------------------------------------------------    
    
    init_tf('models/mobilenet_v1.tflite','models/coco_labels.txt')
    result = run_inference_tf(filename1, filename2,tracktime ,speed, 1) 