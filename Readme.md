# MoTrack - Demo code for testing picamera2 libcam with legacy
### Uses python3 and opencv and Tracks largest moving object in camera view,  tensorflow lite is used for object detection and vehicle length and speed calculation.


![image](https://user-images.githubusercontent.com/5960276/231835732-c91f6541-248d-4330-a7e0-f8a21e9920da.png)

based on the work of Claude Pageau   https://github.com/pageauc/MoTrack-Picam2-Demo

Configured and tested for rpi 3 using v3 camera.  Some specific constants in the code to filter out objects which are not on the road. Currently only R2L movements are recordes and processed.
