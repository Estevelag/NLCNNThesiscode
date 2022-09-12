import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2 as cv
from math import floor
import os
import tensorflow as tf
import subprocess
from os.path import exists

#To activate venv source ./venv/bin/activate 

####Function that Gets all the files in a directory
def listaArchivos(path):
  """This function gets all the files in a directory"""
  filelist=[]
  with os.scandir(path) as entries:
      for i in entries:
        filelist.append(i.name)
  return filelist

#Funtion to tranform the array in a tf record saveable

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


#FUNCTION TO SERIALIZE EACH EXAMPLE AND RETURN IT

def serialize_example(noisyimg, cleanimg, Matrixpix):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'noisy': _float_feature(noisyimg),
      'clean': _float_feature(cleanimg),
      'NLM': _float_feature(Matrixpix),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def ProcessandsavetfRecord(pathfilecrossval,name,h,w,imgGray,ImgGrayClean,namedirectory):
  with tf.io.TFRecordWriter('./'+pathfilecrossval+'/'+namedirectory+name) as writer: # 
    m=imgGray[h*64:(h+1)*64,w*64:(w+1)*64]
    mClean=ImgGrayClean[h*64:(h+1)*64,w*64:(w+1)*64]
    nameimgN=name.split('_')[0]+name.split('_')[1].split('.')[0]+str(1)+'.jpg'
    nameimgC=name.split('_')[0]+name.split('_')[1].split('.')[0]+str(0)+'.jpg'
    # save the jpg patches of the image in just the first cleanpart_0 directory and in the validation directory
    if pathfilecrossval=='validation' or pathfilecrossval=='cleanpart_0/train/' or 'cleanpart_0/test/':
      nameimgC=name.split('_')[0]+name.split('_')[1].split('.')[0]+str(0)+'.jpg'
      nameimgN=name.split('_')[0]+name.split('_')[1].split('.')[0]+str(1)+'.jpg'
      N=21# Parameter for the non local means big window
      right = int(N/2)
      bottom = right+1
      left = right+1
      top = right
      # Make the borders bigger to reflect to NLM the entire patch to process in c++
      m = cv.copyMakeBorder(m, top, bottom, left, right, cv2.BORDER_REFLECT ) # this makes the image 85*85 for the c++ code to return a 64*64
      mClean = cv.copyMakeBorder(mClean, top, bottom, left, right, cv2.BORDER_REFLECT )
      if pathfilecrossval=='validation':
        cv2.imwrite('./'+pathfilecrossval+'/'+nameimgN,m)
        cv2.imwrite('./'+pathfilecrossval+'/'+nameimgC,mClean)
      else:
        cv2.imwrite('./'+'cleanpart_0'+'/'+nameimgN,m)
        cv2.imwrite('./'+'cleanpart_0'+'/'+nameimgC,mClean)
    
    #Search for the noisy image and then process it with c++
    if exists('./'+'validation'+'/'+nameimgN):
      image_name='./'+'validation'+'/'+nameimgN
      NLM = np.array(subprocess.run(["./Roi","2", image_name], capture_output=True, text=True).stdout.split(' ')[1:]).astype(int)
    else:
      #print(pathfilecrossval)
      if exists('./'+'cleanpart_0'+'/'+nameimgN):
        image_name='./'+'cleanpart_0'+'/'+nameimgN
        NLM = np.array(subprocess.run(["./Roi","2", image_name], capture_output=True, text=True).stdout.split(' ')[1:]).astype(int)
      

    # Log transform and normalize everything
    m2=np.log(m+1)/(np.log(256))
    m2Clean=np.log(mClean+1)/(np.log(256))
    NLM=np.log(NLM+1)/(np.log(256))
    
    # Save as a tf record
    example = serialize_example(m2.flatten(),m2Clean.flatten(),NLM.flatten())
    writer.write(example)


def partition1(imgGray,ImgGrayClean,pathval,pathfilecrossval,i):# partition function with different strategies to process and save the images
  height=imgGray.shape[0]
  width=imgGray.shape[1]
  count=0 #Counter of the part of the image
  for h in range(0,floor(height/64)):
    for w in range(0,floor(width/64)):
      name=i.split('.')[0]+'_'+str(count)+'.tfrecord' # this is the name of the file for everyone

      # First write the tf record in the validation if its validation
      # assign a probability to see if this image is going to be saved in the validation or in all the cross validations:
      ValProb = np.random.choice(['normal','validation'], 1, p =[0.8, 0.2])

      if ValProb[0] == 'validation':
        # Write the image in the validation set
         ProcessandsavetfRecord(pathval,name,h,w,imgGray,ImgGrayClean,'')
      
      else:
        #Assign for each of the five cross validations to see if it willl be saved in train or test
        for crossValcounter in range(0,5):

          traintest = np.random.choice(['train','test'], 1, p =[0.8, 0.2])
          if traintest[0]== 'train':
            pathfilecrossval2=pathfilecrossval+'_'+str(crossValcounter)# This will save in the dierectory pathfilecrossval_0, pathfilecrossval_1,..pathfilecrossva_4
            ProcessandsavetfRecord(pathfilecrossval2,name,h,w,imgGray,ImgGrayClean,'train/')
          else:
            pathfilecrossval2=pathfilecrossval+'_'+str(crossValcounter)# This will save in the dierectory pathfilecrossval_0, pathfilecrossval_1,..pathfilecrossva_4
            ProcessandsavetfRecord(pathfilecrossval2,name,h,w,imgGray,ImgGrayClean,'test/')
        
        #np.save(os.path.join(path ,name2), m2) # save

        # save as modified tensor
        #tensor = tf.math.log(tf.math.add(tf.convert_to_tensor(m, dtype=tf.float32),1))# Log transform the image turned into a tensor
        #tensor = tf.divide(tensor,np.log(256)) # Scale the tensor
        #tf.io.write_file(os.path.join(path ,name), tensor, name=name)# Save the tensor of the image
        
        # update image counter
      #print('Voy en ',count)
      count+=1

def PartitionImgAll(path,pathnoisy,path2save): # partition all images in a directory and put them in another directory
  listaimagenes=listaArchivos(path)
  cont=0
  for i in listaimagenes:
    R=path+'/'+i
    k=cv2.imread(R)
    imgGrayClean = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)
    imgGrayClean = imgGrayClean.astype('uint8')

    N=pathnoisy+'/'+i
    q=cv2.imread(N)
    imgGray = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
    imgGray = imgGrayClean.astype('uint8')

    partition1(imgGray,imgGrayClean,'validation',path2save,i)# this works well with one image
    if cont%100==0:
      print(f'voy {cont} imagenes',i)
    cont+=1
    
#./clean is where the images to be partitioned are, 'cleanpart' is where they will be saved

# scale and transform. First transform to logarithm then scale and save them once and for all

PartitionImgAll('./clean_1c','./noisy','./cleanpart')