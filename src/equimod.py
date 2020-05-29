#----------------------------------------------------------------------------------------------------------
#
# takes an equirectangular image as the input and converts this to a cube map image of the following format
#
#	+----+----+----+
#	| Y+ | X+ | Y- |
#	+----+----+----+
#	| X- | Z- | Z+ |
#	+----+----+----+
#
# which when unfolded should take the following format
#            +----+
#	         | Z+ |
#	    +----+----+----+---+
#	    | Y- | X+ | Y+ | X-|
#	    +---+----+----+----+
#	         | Z- |
#	         +----+
#------------------------------------------------------------------------------------------------------------

from PIL import Image	# Python Imaging Library
import math				# Maths functions
import sys				# Allows us to access function args
import os               # Allows us to split the text for saving the file
import numpy as np
from pylab import *
import cv2

image = Image.open('images/equi2.jpg')
inputWidth, inputHeight = image.size;
sqr = inputWidth/4.0
outputWidth = sqr*3
outputHeight = sqr*2

output = []

def getTheta(x,y):
	rtn = 0
	if(y<0):
		rtn =  math.atan2(y,x)*-1
	else:
		rtn = math.pi+(math.pi-math.atan2(y,x))
	return rtn
	
for loopY in range(0,int(outputHeight)):		

	for loopX in range(0,int(outputWidth)):	

		#print(str(loopY)+", "+str(loopX))
	
		tx = 0
		ty = 0
		x = 0;
		y = 0;
		z = 0;
	
		if(loopY<sqr): # top half
		
			if(loopX<sqr): 		# top left box [Y+]
				
				tx = loopX
				ty = loopY
				x = tx-0.5*sqr
				y = 0.5*sqr
				z = ty-0.5*sqr
			
			elif(loopX<2*sqr):	# top middle [X+]
			
				tx = loopX-sqr
				ty = loopY
				x = 0.5*sqr
				y = (tx-0.5*sqr)*-1
				z = ty-0.5*sqr
			
			else:					# top right [Y-]
			
				tx = loopX-sqr*2
				ty = loopY
				x = (tx-0.5*sqr)*-1
				y = -0.5*sqr
				z = ty-0.5*sqr
			
		else:	# bottom half
		
			if(loopX<sqr+1): 		# bottom left box [X-]
			
				tx = loopX
				ty = loopY - sqr
				x = int(-0.5*sqr)
				y = int(tx-0.5*sqr)
				z = int(ty-0.5*sqr)
				
				#print(str(x)+", "+str(y)+", "+str(z))
			
			elif(loopX<2*sqr+1):	# bottom middle [Z-]
			
				tx = loopX-sqr
				ty = loopY-sqr
				x = (ty-0.5*sqr)*-1
				y = (tx-0.5*sqr)*-1
				z = 0.5*sqr			# was -0.5 might be due to phi being reversed
			
			else:					# bottom right [Z+]

				tx = loopX-sqr*2
				ty = loopY - sqr
				x = ty-0.5*sqr
				y = (tx-0.5*sqr)*-1
				z = -0.5*sqr 		# was +0.5 might be due to phi being reversed
					
		# now find out the polar coordinates
		
		rho = math.sqrt(x*x+y*y+z*z)
		normTheta = getTheta(x,y)/(2*math.pi)			# /(2*math.pi) normalise theta
		normPhi = (math.pi-math.acos(z/rho))/math.pi	# /math.pi normalise phi
		
		# use this for coordinates 
		
		iX = normTheta*inputWidth
		iY = normPhi*inputHeight

		# catch possible overflows
		
		if(iX>=inputWidth):
			iX=iX-(inputWidth)
		if(iY>=inputHeight):
			iY=iY-(inputHeight)
			
		output.append(image.getpixel((int(iX),int(iY))))

#ref1 = Image.open('images/ref1.jpg')
#ref2 = Image.open('images/ref2.jpg')

outputImage = Image.new("RGB",((int(outputWidth)),(int(outputHeight))), None)
outputImage.putdata(output)

image.show()

#ref1.show()
#ref2.show()
print(np.shape(outputImage))
outputImage.show()

