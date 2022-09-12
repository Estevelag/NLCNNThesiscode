import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

SAR1 = cv2.imread("0902N.png")
SARG1 = cv2.cvtColor(SAR1, cv2.COLOR_BGR2GRAY)

SAR2 = cv2.imread("0914N.png")
SARG2 = cv2.cvtColor(SAR2, cv2.COLOR_BGR2GRAY)

SAR3 = cv2.imread("0926N.png")
SARG3 = cv2.cvtColor(SAR3, cv2.COLOR_BGR2GRAY)

print(SARG1.shape,SARG2.shape,SARG3.shape)

SARims = np.array([SARG1,SARG2,SARG3])#Getting all the images in an array
SARAverage = np.average(SARims,axis=0)
cv2.imshow(SARG1)
cv2.imshow(SARAverage)


def emptyzeros(matrix):
  matrix[ matrix == 0]=1
  return matrix
  
# review if normalization
#Lambda estimation from real SAR images
#Gamma distribution estimation
lambdaMLE=[]
lambdaME=[]
N=3 # Size of the window
rows,cols = SARG1.shape
filteredimageMLE=np.ones((rows,cols))#Initializing the two filters
filteredimageME=np.ones((rows,cols))
for i in range(0,rows-N,N):
  for j in range(0,cols-N,N):
    imsSAR = np.array([SARG1[i:i+N,j:j+N],SARG2[i:i+N,j:j+N],SARG3[i:i+N,j:j+N]])#Getting all the images in an array
    NewimgN = np.average(imsSAR,axis=0)# getting the average value of each pixel. average image
    
    
    #Getting all the values from a window
    arr1=(SARG1[i:i+N,j:j+N]).flatten()
    matrixvalues=np.concatenate(((SARG2[i:i+N,j:j+N]).flatten(),arr1),axis=None)#add
    matrixvalues=np.concatenate((matrixvalues,SARG3[i:i+N,j:j+N].flatten()),axis=None)##geeting the same values of a 

    ##Moment estimation
    meanvalue=np.mean(matrixvalues)# the average value for a window of pixels
    Variance=np.std(matrixvalues)# the standard deviation
    alpha=(meanvalue/Variance)**2 #moment estimation of alpha
    beta=Variance**2/meanvalue #moment estima    #Getting all the values from a window
    arr1=(SARG1[i:i+N,j:j+N]).flatten()
    matrixvalues=np.concatenate(((SARG2[i:i+N,j:j+N]).flatten(),arr1),axis=None)#add
    matrixvalues=np.concatenate((matrixvalues,SARG3[i:i+N,j:j+N].flatten()),axis=None)##geeting the same values of a tion of Beta
    lambdaME.append(beta/alpha)#radar instensity estimator ME
    x = np.linspace(0,5,1000)# getting a position for the gamma distribution
    ye= gamma.pdf(x, alpha, 0, beta)##Seeing the gamma function
    intensitymoment=x[np.where(ye == np.max(ye))]# This gets the most probable value of the pixel, if done with more than N=1 it probably isn't the best
    ##Maximum likelihood estimation
    #matrixvalues[matrixvalues == 0] = 0.0001##correction for fit function
    shape, loc, scale = gamma.fit(matrixvalues)
    lambdaMLE.append(1/(shape*scale))#radar instensity estimator MLE
    y = gamma.pdf(x, shape, loc, scale)
    intensityMLE=x[np.where(y == np.max(y))]# This gets the most probable value of the pixel, if done with more than N=1 it probably isn't the best


#Plotting the estimated lambda with ME
fig, ax = plt.subplots(1, 1)
ax.hist(lambdaME, density=True, histtype='stepfilled',bins=100, alpha=1)
ax.set_title('lambda estimation ME')
fig.show()# This gets the most probable value of the pixel, if done with more than N=1 it probably isn't the best

#Plotting the estimated lambda
fig, ax = plt.subplots(1, 1)
ax.hist(lambdaMLE, density=True, histtype='stepfilled',bins=100, alpha=1)
ax.set_title('lambda estimation MLE')
fig.show()# This gets the most probable value of the pixel, if done with more than N=1 it probably isn't the best

# Plotting them both in the same thing 
n, b, patches = plt.hist(lambdaME, 50, histtype='stepfilled')
bin_max = np.where(n == n.max())
n1, b1, patches = plt.hist(lambdaMLE, 50, histtype='stepfilled')
bin_max1 = np.where(n1 == n1.max())
plt.legend(['ME', 'MLE'],loc='upper left')
plt.title(f"Comparación estimaciones de {len(lambdaME)} muestras en imágenes SAR")
print("El valor esperado de lambda según ME y MLE de Sentinel-1 es: ", b[bin_max][0]," y ", b1[bin_max1][0],' de ', len(lambdaME),' muestras calculadas')

print("la media y mediana estimadas con ME de las imagenes SAR fueron: ",np.mean(lambdaME),np.median(lambdaME))
print("la media y mediana estimadas con MLE de las imagenes SAR fueron: ",np.mean(lambdaMLE),np.median(lambdaMLE))


#El valor esperado de lambda según ME y MLE de Sentinel-1 es:  0.03468707386217057  y  1.0747074992677573e-06  de  43195  muestras calculadas