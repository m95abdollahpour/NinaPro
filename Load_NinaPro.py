
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
from scipy.signal import butter, lfilter, freqz, sosfilt
import os
import scipy.io 
import numpy as np
from sklearn.utils import shuffle
import cv2




# directory which contains the subjects data
directory = ''


def rep_EMG_exercise(rep, data_e):

   '''
   To extract train and test data with corrsponding labels
   
   --------------
   rep: list of repetations
   data_e: data of a specific exercise 
   '''
   train_data_e = []
   train_label_e = []
   test_data_e = []
   test_label_e = []
   for i in range(len(data_e['emg'])):
        
        if (data_e['rerepetition'][i] in rep):
            train_data_e.extend(np.reshape(data_e['emg'][i,:], (1,12)))
            train_label_e.extend([data_e['restimulus'][i,0]])
        else:
            test_data_e.extend(np.reshape(data_e['emg'][i,:], (1,12)))
            test_label_e.extend([data_e['restimulus'][i,0]])

   return train_data_e, train_label_e, test_data_e, test_label_e


   
   
   

Test_Data= []
Test_Label = []
Train_Data = []
Train_Label = []

for j in range(40):

    data_e1=scipy.io.loadmat(directory + '/DB2_s'+str(j)+'/DB2_s'+str(j)+'/S'+str(j)+'_E1_A1.mat')
    data_e2=scipy.io.loadmat(directory + '/DB2_s'+str(j)+'/DB2_s'+str(j)+'/S'+str(j)+'_E2_A1.mat')
    data_e3=scipy.io.loadmat(directory + '/DB2_s'+str(j)+'/DB2_s'+str(j)+'/S'+str(j)+'_E3_A1.mat')
   
    
    train_data_e1, train_label_e1, test_data_e1, test_label_e1 = rep_EMG_exercise([1,3,5,6], data_e1)
    train_data_e2, train_label_e2, test_data_e2, test_label_e2 = rep_EMG_exercise([1,3,5,6], data_e2)
    train_data_e3, train_label_e3, test_data_e3, test_label_e3 = rep_EMG_exercise([1,3,5,6], data_e3)
    
    train_data_e1.extend(train_data_e2)
    train_data_e1.extend(train_data_e3)
    Train_Data.extend(train_data_e1)
    
    test_data_e1.extend(test_data_e2)
    test_data_e1.extend(test_data_e3)
    Test_Data.extend(test_data_e1)
    
    train_label_e1.extend(train_label_e2)
    train_label_e1.extend(train_label_e3)
    Train_Label.extend(train_label_e1)
    
    test_label_e1.extend(test_label_e2)
    test_label_e1.extend(test_label_e3)
    Test_Label.extend(test_label_e1)
    

Train_Data = np.array(Train_Data)
Train_Label = np.array(Train_Label)


Test_Data = np.array(Test_Data)
Test_Label = np.array(Test_Label)



#Segmentation
Test_Data2 = []
Test_Label2 = []
Test_Rest = []


i = 0
window_size = 400
increment = 100 # overlap 
while (i < int(len(Test_Data)/increment)-3):
    # checkin the last and first datapoint of a segment to have the same label
    if (Test_Label[i*increment:i*increment+window_size][0] == Test_Label[i*increment:i*increment+window_size][window_size-1] ):
      # check if the labels is the rest
        if (Test_Label[i*increment:i*increment+window_size][0] == 0):
            if ((Test_Label[i*increment - 2000] or Test_Label[i*increment+2000]) == 0): #added  
                Test_Rest.append(Test_Data[i*increment:i*increment+window_size,:])
        else:
            Test_Data2.append(Test_Data[i*increment:i*increment+window_size,:])
            Test_Label2.append(Test_Label[i*increment:i*increment+window_size])
        i += 1
    else:
       i += 1

# randomly selecting few segments of the rest for imbalanced data problem
Test_Rest = np.array(Test_Rest)
randnums = np.random.randint(1,len(Test_Rest),np.int(len(Test_Data2)/50))
Test_Rest = Test_Rest[randnums]
Test_Data2 = np.concatenate((Test_Data2,Test_Rest))
Test_Label2 = np.concatenate((np.array(Test_Label2)[:,1],np.ravel(np.zeros((1,len(Test_Rest))))))




Train_Data2 = []
Train_Label2 = []
Train_Rest = []

i = 0
window_size = 400
increment = 100
while (i < int(len(Train_Data)/increment)-3):
    if (Train_Label[i*increment:i*increment+window_size][0] == Train_Label[i*increment:i*increment+window_size][window_size-1] ):
        if (Train_Label[i*increment:i*increment+window_size][0] == 0):
            if ((Train_Label[i*increment - 2000] or Train_Label[i*increment+2000]) == 0): #added: to make sure the rest labeled data is really rest
                Train_Rest.append(Train_Data[i*increment:i*increment+window_size,:])
        else:
            Train_Data2.append(Train_Data[i*increment:i*increment+window_size,:])
            Train_Label2.append(Train_Label[i*increment:i*increment+window_size])

        i += 1
    else:
       i += 1

    

	
# adding rest segments to the data
#the number of rest segments are chosen with respect to the number of other class samples to prevent unbalanced dataset
Train_Rest = np.array(Train_Rest)
randnums = np.random.randint(1,len(Train_Rest),np.int(len(Train_Data2)/50))
Train_Rest = Train_Rest[randnums]
Train_Data2 = np.concatenate((Train_Data2,Train_Rest))
Train_Label2 = np.concatenate((np.array(Train_Label2)[:,1],np.ravel(np.zeros((1,len(Train_Rest))))))





x_train, y_train = shuffle(Train_Data2, Train_Label2, random_state=100)
x_test = np.array(Test_Data2)
y_test = np.array(Test_Label2)


#downsampling to 1000 Hz
def Downsample(X):
    A = np.ones((np.shape(X)[0], 200, 12))
    for i in range(len(X)):
        for j in range(12):
            A[i, :, j] = scipy.signal.resample_poly(X[i, :, j], up =1000,  down = 2000)
    return A

x_train =  Downsample(x_train)
x_test =  Downsample(x_test)




#normalization: normalizing with respect to each EMG channel
for i in range(12):
  s = np.std(x_train[:,:,i])
  m = np.mean(x_train[:,:,i])
  x_train[:,:,i] -= m
  x_train[:,:,i] /= s
  x_test[:,:,i] -= m
  x_test[:,:,i] /= s
  
  