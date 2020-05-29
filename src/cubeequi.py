from PIL import Image	# Python Imaging Library
import math				# Maths functions
import sys				# Allows us to access function args
import os				# Allows us to split the text for saving the file
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

#------------------------------------------------------------------------------------------------------------
#the cube input should be of the following format
#                +----+
#	         | Z+ |
#	    +----+----+----+---+
#	    | Y- | X+ | Y+ | X-|
#	    +---+----+----+----+
#	         | Z- |
#	         +----+
#------------------------------------------------------------------------------------------------------------
input=cv2.imread('images/cube.jpg')
posz = Image.open('images/zpos.jpg')
negz = Image.open('images/zneg.jpg')
posx = Image.open('images/xpos.jpg')
negx = Image.open('images/xneg.jpg')
posy = Image.open('images/ypos.jpg')
negy = Image.open('images/yneg.jpg')

squareLength = posx.size[0]
halfSquareLength = squareLength/2

outputWidth = squareLength*6
outputHeight = squareLength*3

#6, 3 used for scaling the image for better vision, although the real values are 2 and 1

output = []

def unit3DToUnit2D(x,y,z,faceIndex):
	
	if(faceIndex=="X+"):
		x2D = y+0.5
		y2D = z+0.5
	elif(faceIndex=="Y+"):
		x2D = (x*-1)+0.5
		y2D = z+0.5
	elif(faceIndex=="X-"):
		x2D = (y*-1)+0.5
		y2D = z+0.5
	elif(faceIndex=="Y-"):
		x2D = x+0.5
		y2D = z+0.5
	elif(faceIndex=="Z+"):
		x2D = y+0.5
		y2D = (x*-1)+0.5
	else:
		x2D = y+0.5
		y2D = x+0.5
		
	# need to do this as image.getPixel takes pixels from the top left corner.
	
	y2D = 1-y2D
	
	return (x2D,y2D)

def projectX(theta,phi,sign):
	
	x = sign*0.5
	faceIndex = "X+" if sign==1 else "X-"
	rho = float(x)/(math.cos(theta)*math.sin(phi))
	y = rho*math.sin(theta)*math.sin(phi)
	z = rho*math.cos(phi)
	return (x,y,z,faceIndex)
	
def projectY(theta,phi,sign):
	
	y = sign*0.5
	faceIndex = "Y+" if sign==1 else "Y-"
	rho = float(y)/(math.sin(theta)*math.sin(phi))
	x = rho*math.cos(theta)*math.sin(phi)
	z = rho*math.cos(phi)
	return (x,y,z,faceIndex)
	
def projectZ(theta,phi,sign):

	z = sign*0.5
	faceIndex = "Z+" if sign==1 else "Z-"
	rho = float(z)/math.cos(phi)
	x = rho*math.cos(theta)*math.sin(phi)
	y = rho*math.sin(theta)*math.sin(phi)
	return (x,y,z,faceIndex)
	
def getColour(x,y,index):
	
	if(index=="X+"):
		return posx.getpixel((x,y))
	elif(index=="X-"):
		return negx.getpixel((x,y))
	elif(index=="Y+"):
		return posy.getpixel((x,y))
	elif(index=="Y-"):
		return negy.getpixel((x,y))
	elif(index=="Z+"):
		return posz.getpixel((x,y))
	elif(index=="Z-"):
		return negz.getpixel((x,y))
	
	
def convertEquirectUVtoUnit2D(theta,phi):
	
	# calculate the unit vector
	
	x = math.cos(theta)*math.sin(phi)
	y = math.sin(theta)*math.sin(phi)
	z = math.cos(phi)
	
	# find the maximum value in the unit vector
	
	maximum = max(abs(x),abs(y),abs(z))
	xx = x/maximum
	yy = y/maximum
	zz = z/maximum
	
	# project ray to cube surface
	
	if(xx==1 or xx==-1):
		(x,y,z, faceIndex) = projectX(theta,phi,xx)
	elif(yy==1 or yy==-1):
		(x,y,z, faceIndex) = projectY(theta,phi,yy)
	else:
		(x,y,z, faceIndex) = projectZ(theta,phi,zz)
	
	(x,y) = unit3DToUnit2D(x,y,z,faceIndex)
	
	x*=squareLength
	y*=squareLength
		
	x = int(x)
	y = int(y)

	return {"index":faceIndex,"x":x,"y":y}
	
# 1. loop through all of the pixels in the output image

for loopY in range(0,int(outputHeight)):		# 0..height-1 inclusive

	for loopX in range(0,int(outputWidth)):
	
		# 2. get the normalised u,v coordinates for the current pixel
		
		U = float(loopX)/(outputWidth-1)		# 0..1
		V = float(loopY)/(outputHeight-1)		# no need for 1-... as the image output needs to start from the top anyway.		
		
		# 3. taking the normalised cartesian coordinates calculate the polar coordinate for the current pixel
	
		theta = U*2*math.pi
		phi = V*math.pi
		
		# 4. calculate the 3D cartesian coordinate which has been projected to a cubes face
		
		cart = convertEquirectUVtoUnit2D(theta,phi)
		
		# 5. use this pixel to extract the colour
		
		output.append(getColour(cart["x"],cart["y"],cart["index"]))
		
		
# 6. write the output array to a new image file
		
outputImage = Image.new("RGB",((int(outputWidth)),(int(outputHeight))), None)
outputImage.putdata(output)

#negy.show()
#posx.show()
#posy.show()
#negx.show()
#negz.show()


cv2.imshow("Input",input)
outputImage.show()


img=cv2.imread('images/xpos.jpg',0)

# the following section can be removed as per wish as it contains some frequency analysis I had done on the image faces.
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.imshow(magnitude_spectrum,cmap='gray')
plt.show()
plt.imshow(img,cmap='gray')
plt.show()

hist,bins = np.histogram(magnitude_spectrum.ravel(),256,[0,256])
print hist
plt.plot(hist) 
plt.show() 

blur1 = cv2.GaussianBlur(img,(5,5),0)
dft2 = cv2.dft(np.float32(blur1),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift2= np.fft.fftshift(dft2)

magnitude_spectrum2 = 20*np.log(cv2.magnitude(dft_shift2[:,:,0],dft_shift2[:,:,1]))

hist2,_= np.histogram(magnitude_spectrum2.ravel(),256,[0,256])
plt.plot(hist2) 
plt.show() 

#bins = 10**(np.arange(0,4))
#print "bins: ", bins
#plt.xscale('log')
#plt.hist(hist,histtype='stepfilled',bins=bins)
#plt.show()

image1=np.copy(outputImage)
gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
dft1 = cv2.dft(np.float32(gray),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift1 = np.fft.fftshift(dft1)

magnitude_spectrum1 = 20*np.log(cv2.magnitude(dft_shift1[:,:,0],dft_shift1[:,:,1]))

hist1,_ = np.histogram(magnitude_spectrum1.ravel(),256,[0,256])
plt.plot(hist1) 
plt.show() 

cv2.waitKey(0)

