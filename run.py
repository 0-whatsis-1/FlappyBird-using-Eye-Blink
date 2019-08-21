import cv2
import dlib
import pygame
from pygame.locals import *
import sys
import random
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
from imutils.video import VideoStream
import argparse
import time


class FlappyBird:
	def __init__(self):
		self.screen = pygame.display.set_mode((500, 800))
		self.bird = pygame.Rect(65, 50, 50, 50)
		self.background = pygame.image.load("images/bg.png").convert()
		self.birdSprites = [pygame.image.load("images/1.png").convert_alpha(),
			pygame.image.load("images/2.png").convert_alpha(),
			pygame.image.load("images/dead.png")]
		self.wallUp = pygame.image.load("images/bottom.png").convert_alpha()
		self.wallDown = pygame.image.load("images/top.png").convert_alpha()
		self.gap = 220
		self.wallx = 400
		self.birdY = 350
		self.jump = 0
		self.jumpSpeed = 10
		self.gravity = 5
		self.dead = False
		self.sprite = 0
		self.counter = 0
		self.offset = random.randint(-110, 110)

	def updateWalls(self):
		self.wallx -= 2
		if self.wallx < -80:
			self.wallx = 400
			self.counter += 1
			self.offset = random.randint(-110, 110)

	def birdUpdate(self):
		if self.jump:
			self.jumpSpeed -= 1
			self.birdY -= self.jumpSpeed
			self.jump -= 1
		else:
			self.birdY += self.gravity
			self.gravity += 0.2
		self.bird[1] = self.birdY
		upRect = pygame.Rect(self.wallx, 360 + self.gap - self.offset + 10, self.wallUp.get_width() - 10, self.wallUp.get_height())
		downRect = pygame.Rect(self.wallx, 0 - self.gap - self.offset - 10, self.wallDown.get_width() - 10, self.wallDown.get_height())
		if upRect.colliderect(self.bird):
			self.dead = True
		if downRect.colliderect(self.bird):
			self.dead = True
		if not 0 < self.bird[1] < 720:
			self.bird[1] = 50
			self.birdY = 50
			self.dead = False
			self.counter = 0
			self.wallx = 400
			self.offset = random.randint(-110, 110)
			self.gravity = 5


	def eye_aspect_ratio(self, eye):
		""" Calculates the Eye Aspect Ratio """
		# Compute the euclidean distances A, B & C
		# Between the two sets of vertical eye landmarks (x, y)-coordinates
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])
		# Between the horizontal eye landmark (x, y)-coordinates
		C = dist.euclidean(eye[0], eye[3])
		# Computes the eye aspect ratio by the formula
		ear = (A + B) / (2.0 * C)
		return ear


	def run(self):
		ap = argparse.ArgumentParser()		# Construct the argument parser
		#Parse the arguments
		ap.add_argument("-p", "--shape-predictor", required=True)
		args = vars(ap.parse_args())

		EYE_AR_THRESH = 0.3  # Eye Aspect Ratio Threshold -> To indicate blink
		EYE_AR_CONSEC_FRAMES = 3   # Number of consecutive frames the eye must be below the threshold
		counter = 0  # Frame counter
		total = 0  # total number of blinks

		# Initialize dlib's face detector(HOG-based) & then create the facial landmark predictor
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(args["shape_predictor"])

		# Grab the indexes of the facial landmarks for the eyes
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

		# Start the video stream thread
		vs = VideoStream(src=0).start()
		time.sleep(1.0)

		clock = pygame.time.Clock()
		pygame.font.init()
		font = pygame.font.SysFont("Arial", 50)

		while True:
			# Capture the frame from the threaded video stream, resize it & convert it to grayscale
			frame = vs.read()
			frame = imutils.resize(frame, width=450)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Detect faces in the grayscale frame
			rects = detector(gray, 0)

			# Loop over the face detections
			for rect in rects:
				# Determine the facial landmarks for the face region.
				# Convert the facial landmark (x, y) coordinates to a numpy array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)
				# Left & Right eye coordinates
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				# Calculate the eye aspect ratio for both eyes
				leftEAR = self.eye_aspect_ratio(leftEye)
				rightEAR = self.eye_aspect_ratio(rightEye)
				ear = (leftEAR + rightEAR) / 2.0  # Average eye aspect ratio

				# If the Eye Aspect Ratio is below the blink threshold
				if ear < EYE_AR_THRESH:
					counter += 1  # Increment the frame counter

				# Else the eye aspect ratio is not below the blink threshold
				else:
					# If the eyes were closed for a sufficient number of frames
					# & the bird is not dead
					if counter >= EYE_AR_CONSEC_FRAMES and not self.dead:
						total += 1  # Increment the total number of blinks
						self.jump = 17
						self.gravity = 5
						self.jumpSpeed = 10
					counter = 0  # Reset the eye frame counter as game is over

			cv2.imshow("Flappy Bird", frame)
			key = cv2.waitKey(1) & 0xFF
			clock.tick(60)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()

			self.screen.fill((255, 255, 255))
			self.screen.blit(self.background, (0, 0))
			self.screen.blit(self.wallUp, (self.wallx, 360 + self.gap - self.offset))
			self.screen.blit(self.wallDown, (self.wallx, 0 - self.gap - self.offset))
			self.screen.blit(font.render(str(self.counter), -1, (255, 255, 255)), (200, 50))
			if self.dead:
				self.sprite = 2
			elif self.jump:
				self.sprite = 1
			self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
			if not self.dead:
				self.sprite = 0
			self.updateWalls()
			self.birdUpdate()
			pygame.display.update()

if __name__ == "__main__":
	FlappyBird().run()
