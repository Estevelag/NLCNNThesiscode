# Submuestrear la imagen como SAR o ver como igualar la densidad del ruido
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import gamma

def probgen(L,x,lamd):

  return (1/sp.special.gamma(L))*(L*lamd)**L*x**(L-1)*np.exp(-L*lamd*x)

def target( L, theta,lamd):#the arbitray pdf
    if theta < 0 or theta > 5:
        return 0
    else:
        return probgen(L, theta,lamd)

def gammaMCS(niters,L,lamd):
  sigma = 0.3
  theta = 0.1 #start somewhere
  if type(niters)== tuple:
    iter=(niters[0])*niters[1] -1
    samples = np.zeros(iter+1)
    samples[0] = theta
    lamd=1
    for i in range(iter):
        theta_p = theta + stats.norm(0, sigma).rvs()
        rho = min(1, target(L, theta_p,lamd)/target(L, theta,lamd ))
        u = np.random.uniform()# with this is like evaluating with a probability
        if u < rho:
            theta = theta_p
        samples[i+1] = theta
    np.random.shuffle(samples)
    samples = samples.reshape(niters[1],niters[0])
    samples=np.array(samples)
    
  else:
    niters = niters-1
    samples = np.zeros(niters+1)
    samples[0] = theta
    lamd=1
    for i in range(niters):
        theta_p = theta + stats.norm(0, sigma).rvs()
        rho = min(1, target(L, theta_p,lamd)/target(L, theta,lamd ))
        u = np.random.uniform()# with this is like evaluating with a probability
        if u < rho:
            theta = theta_p
        samples[i+1] = theta
  return samples

def plotting(x,y,title):
  plt.title(title)
  plt.plot(x, y)
  plt.show()

def noisify(imgGray,lamd):
  #Intentar con una imagen SAR estimar la refelectancia y A partir de esa reflectancia usarla en la distribución
  shape=imgGray.shape# Coger una imagen SAR del tamaño 500*500 para comparar por ventanas
  looks=3
  noisesss=gammaMCS(shape,looks,lamd)
  a=noisesss.reshape(imgGray.shape)
  Newimg1=np.multiply(a,imgGray)
  Newimg1=np.clip(Newimg1, 0, 255)
  cv2.imshow(Newimg1)
  return Newimg1

# Import and read the images
img = cv2.imread("lenaGray.png")  
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow(imgGray)
imgGray.shape
plt.figure(figsize = (30,15))
plt.imshow(imgGray,cmap='gray')
plt.title('Lenagray')
plt.show()

# Noisify the images
Newimg1=noisify(imgGray,1)
Newimg2=noisify(imgGray,1)
Newimg3=noisify(imgGray,1)

# To view the estimation of th elambda and the gamma function of one window of 9 pixels wide and height :

#testing grounds taking into account the real value For only one sliding window
lambdaMLE=[]
lambdaME=[]
i=200
j=200
N=9
rows,cols = imgGray.shape
ims = np.array([Newimg1[i:i+N,j:j+N],Newimg2[i:i+N,j:j+N],Newimg3[i:i+N,j:j+N]])#Getting all the images in an array
NewimgN = np.average(ims,axis=0)# getting the average value of each pixel
filteredimageMLE=np.ones((rows,cols))
filteredimageME=np.ones((rows,cols))
#Ratio of speckle
arr1=(np.divide(Newimg2[i:i+N,j:j+N],NewimgN)).flatten()
matrixvalues=np.concatenate(((np.divide(Newimg1[i:i+N,j:j+N],NewimgN)).flatten(),arr1),axis=None)#add
ratio3=np.divide(Newimg3[i:i+N,j:j+N],NewimgN)
matrixvalues=np.concatenate((matrixvalues,ratio3.flatten()),axis=None)##geeting the same values of a 

meanvalue=np.mean(matrixvalues)# the same error for a window of pixels
Variance=np.std(matrixvalues)
alpha=(meanvalue/Variance)**2 #moment estimation of alpha
beta=Variance**2/meanvalue #moment estimation of Beta
lambdaME.append(beta/alpha)#radar instensity estimator ME
x = np.linspace(0,5,1000)
ye= gamma.pdf(x, alpha, 0, beta)##Seeing the gamma function
intensitymoment=x[np.where(ye == np.max(ye))]# This gets the most probable value of the pixel, if done with more than N=1 it probably isn't the best
#Plotting ME
fig, ax = plt.subplots(1, 1)
ax.hist(matrixvalues, density=True, histtype='stepfilled',bins=50, alpha=0.2)
plt.plot(x, ye)
plt.title("Momentos estadísticos")
fig.show()


##End plotting
#matrixvalues[matrixvalues == 0] = 0.0000001
shape, loc, scale = gamma.fit(matrixvalues)#floc=0
lambdaMLE.append(1/(shape*scale))#radar instensity estimator MLE
y = gamma.pdf(x, shape, loc, scale)
intensityMLE=x[np.where(y == np.max(y))]
####Plotting MLE
fig, ax = plt.subplots(1, 1)
ax.hist(matrixvalues, density=True, histtype='stepfilled',bins=50, alpha=0.2)
fig.show()# This gets the most probable value of the pixel, if done with more than N=1 it probably isn't the best
plt.plot(x, y)
plt.title("Máxima verosimilitud")
##End plotting
filteredimageMLE[i:i+N,j:j+N]=(1/meanvalue)*NewimgN*intensityMLE*NewimgN #This gets the expected value normalizedby each point
filteredimageME[i:i+N,j:j+N]=((1/meanvalue)*NewimgN*intensitymoment)*NewimgN#The normalized by pixel expectedvaluetimes the average of them all 
print("Los valores esperados de la intensidad son:",intensitymoment,intensityMLE)
print("Los valores estimados por máxima verosimilitud y momento estadisticos respectivamente son: ",lambdaMLE,' y ',lambdaME)


# Now to do it for a lot of windos and seeing how it behaves we do:
#### Now for every sliding window and saving the results
lambdaMLE=[]
lambdaME=[]
N=7 # Size of the window
rows,cols = imgGray.shape
filteredimageMLE=np.ones((rows,cols))#Initializing the two filters
filteredimageME=np.ones((rows,cols))
for i in range(0,rows-N,N):
  for j in range(0,cols-N,N):
    ims = np.array([Newimg1[i:i+N,j:j+N],Newimg2[i:i+N,j:j+N],Newimg3[i:i+N,j:j+N]])#Getting all the images in an array
    NewimgN = np.average(ims,axis=0)# getting the average value of each pixel. average image
    #Ratio of speckle
    arr1=(np.divide(Newimg2[i:i+N,j:j+N],NewimgN)).flatten()
    matrixvalues=np.concatenate(((np.divide(Newimg1[i:i+N,j:j+N],NewimgN)).flatten(),arr1),axis=None)#add
    ratio3=np.divide(Newimg3[i:i+N,j:j+N],NewimgN)
    matrixvalues=np.concatenate((matrixvalues,ratio3.flatten()),axis=None)##geeting the same values of a 

    ##Moment estimation
    meanvalue=np.mean(matrixvalues)# the average value for a window of pixels
    Variance=np.std(matrixvalues)# the standard deviation
    alpha=(meanvalue/Variance)**2 #moment estimation of alpha
    beta=Variance**2/meanvalue #moment estimation of Beta
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

# And now to see how the lambda behaves under each estimator we use:

#### Plotting the lambda estimation of the MLE 
n, b, patches = plt.hist(lambdaME, 50, histtype='stepfilled')
bin_max = np.where(n == n.max())
n1, b1, patches = plt.hist(lambdaMLE, 50, histtype='stepfilled')
bin_max1 = np.where(n1 == n1.max())
plt.legend(['ME', 'MLE'],loc='upper left')
plt.title("Comparación estimaciones de 5236 muestras")
print("El valor esperado de lambda según ME y MLE es: ", b[bin_max][0]," y ", b1[bin_max1][0],' de ', len(lambdaME),' muestras calculadas')
