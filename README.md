# line_detector

The script is used to detect horizontal lines and vertical lines in an image. 


Two algorithms are provided. One is based on the pixel discontinuity (method == 'pix') in the image, and the other is based on Hough transformation (method == 'hough').


Put your image in the same folder, then to parse lines, run:

python linedetect.py

The parsed horizontal and vertical lines will be saved into two lists as y = [y1, y2, y3...], x = [x1, x2, x3...]. 
