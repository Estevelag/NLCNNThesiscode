import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.stats import gamma
from scipy.stats import entropy
import matplotlib.pyplot as plt
import cv2 as cv


##Euclidean distance function
def distance_squared(patch1,patch2):
  return np.sum(np.multiply(patch1-patch2, patch1-patch2))

# Function to get the standard deviation in a patch
def standard_deviation(patch1):
  return np.std(patch1)

# New distance function
def distance_statistics(patch1,patch2):
  return np.sum(patch1-patch2)


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def haparam(h0,h1,reference_patch):# h parameter from rené
  SNR=signaltonoise(reference_patch,axis=None)
  return h0+h1/(1+1/SNR) 

#Kullberg leidberg distance
def KullbackLeiberdifference(reference_patch,patch_to_compare,gammause=True,graph=False): # distance between two pdfs
  if gammause== True:
    x=[i for i in range(0,100,1)]
    shape, loc, scale = gamma.fit(patch_to_compare)
    reference_patch = gamma.rvs(shape, loc, scale,size=100)
    shape1, loc1, scale1 = gamma.fit(reference_patch)
    patch_to_compare = gamma.rvs(shape1, loc1, scale1,size=100)
    if graph==True:
      fig, ax = plt.subplots(1, 2)
      ax[0].hist(reference_patch, density=True, histtype='stepfilled',bins=50, alpha=0.2)
      ax[0].set_title('Gamma reference patch')
      x = np.linspace(0,300,1000)
      y= gamma.pdf(x,shape1, loc1, scale1)
      ax[0].plot(x,y)

      ax[1].hist(patch_to_compare, density=True, histtype='stepfilled',bins=50, alpha=0.2)
      ax[1].set_title('Gamma patch to compare')
      y2= gamma.pdf(x,shape, loc, scale)
      ax[1].plot(x,y2)
      fig.show()
  else:
    reference_patch=reference_patch.flatten()
    patch_to_compare=patch_to_compare.flatten()
  return entropy(reference_patch, qk=patch_to_compare)





##Euclidean distance function
def distance_squared(patch1,patch2):
  return np.sum(np.multiply(patch1-patch2, patch1-patch2))

# Function to get the standard deviation in a patch
def standard_deviation(patch1):
  return np.std(patch1)

# New distance function
def distance_statistics(patch1,patch2):
  return np.sum(patch1)

########### Importing the image

SAR1 = cv2.imread("0902N.png")
imgGray = cv2.cvtColor(SAR1, cv2.COLOR_BGR2GRAY)
imgGray = imgGray.astype('uint8') # think better this thing to unify it into a normal image

##can only use odd numbers in this N and r
import time
import os


def artificialNLM(imgGray, N,r,distance):
  '''Function of artifiacl nonlocal means'''
  # making more borders of the image so its not cropped
  right = int(N/2)+1
  bottom = right
  left = right
  top = right
  imgGray = cv.copyMakeBorder(imgGray, top, bottom, left, right, cv2.BORDER_REFLECT )


  ## Starting the nonlocal means algorithm
  rows,cols = imgGray.shape
  newimage=np.ones((rows,cols))
  newimage2=np.ones((rows,cols))
  start_time = time.time()
  for i in range(0,rows-N):
    for j in range(0,cols-N):
      window=imgGray[i:i+N,j:j+N]
      center_pixel=imgGray[i+int(N/2),j+int(N/2)]
      reference_patch=imgGray[i+int(N/2)-int(r/2):i+int(N/2)+int(r/2)+1,j+int(N/2)-int(r/2):j+int(N/2)+int(r/2)+1]
      stdev=standard_deviation(reference_patch)
      h=0.3*stdev #Parameter to adjust the weights 
      weightsmatrix=np.zeros((N,N))# This is where the weights are going to be stored
      #Now let us search for similar patches in the window we are on
      for k in range(0,N-r):# Moves through all the patches in the window
        for m in range (0,N-r):
          patch_to_compare=window[k:k+r,m:m+r]
          # The next line is for a pixel weighting
          if distance=='Euclidean':
            weightsmatrix[k+int(r/2),m+int(r/2)]=np.exp(-max(([distance_squared(reference_patch,patch_to_compare)-2*stdev**2,0]))/h**2)
            
          if distance== 'Kullbackleiberg':
            weightsmatrix[k+int(r/2),m+int(r/2)]=np.exp(-KullbackLeiberdifference(reference_patch,patch_to_compare,gammause=False)/h**2)
          # The next commmented line is to take into account the whole patch to add to the weights
          #weightsmatrix[k:k+r,m:m+r]=np.exp(-max([distance_squared(reference_patch,patch_to_compare)-2*stdev**2,0])/h**2) # makes the weight the same for the patch resembling it
      #Now lets get the average value of the pixel and all of the patches that resemble it
      newimage[i+int(N/2)+1,j+int(N/2)+1]=np.average(window,weights=weightsmatrix)

  newimage= np.clip(newimage, 0, 255)
  top=int(N/2)+1
  newimage3= newimage[top:newimage.shape[0]-top,top : newimage.shape[1]-top]
  return newimage3



SAR1 = cv2.imread("09159.jpg")
imgGray = cv2.cvtColor(SAR1, cv2.COLOR_BGR2GRAY)
imgGray = imgGray.astype('uint8') # this is to unify it into a normal image





N=11 # Size of the window
r= 5 # Size of the patch
distance='Kullbackleiberg'

start_time = time.time() 
newimage3=artificialNLM(imgGray, N,r,distance)
i='OwnNLM_Kullbag_11_5.png'
path='./'
cv2.imwrite(os.path.join(path , i), newimage3)
cv2.imshow(newimage3)
cv2.imshow(imgGray)
print("--- %s seconds ---" % (time.time() - start_time))