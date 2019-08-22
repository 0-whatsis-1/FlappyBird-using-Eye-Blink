# FlappyBird-using-Eye-Blink

Overview : 
This code uses dlib's face detector(HOG-based) to detect the player's face, 
then the facial landmark predictor is used to detect eyes of the player.
Further, eye coordinates for both the eyes are calculated.
In particular 6 eye coordinates(4-vertical, 2-horizontal) are used for
calculating eye aspect ratio.
The eye aspect ratio equation is based on the work by Soukupová and Čech
in their 2016 paper, Real-Time Eye Blink Detection using Facial Landmarks.

Dependencies : 
opencv3, imutils, scipy, dlib, pygame

Usage : 
Run the file run.py in the terminal also specify the command line argument
(--shape-predictor) followed by the the facial landmark
detector(shape_predictor_68_face_landmarks.dat).
python3 run.py --shape-predictor shape_predictor_68_face_landmarks.dat

