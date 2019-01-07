import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import color, io
from scipy.misc import imresize
from glob import glob

img_size = 64

#show出 numpy.array所有數值
#np.set_printoptions(threshold=np.nan)

#取得訓練圖片路徑
files_path = './train'

#儲存各類別的檔案命名規則
d_file_path = {'_1_path': '1_*.jpg', '_2_path': '2_*.jpg', '_3_path': '3_*.jpg', 
               '_4_path': '4_*.jpg', '_5_path': '5_*.jpg', '_6_path': '6_*.jpg',
               '_7_path': '7_*.jpg', '_8_path': '8_*.jpg', '_9_path': '9_*.jpg',
               '_10_path': '10_*.jpg'}

#儲存各類別之所有檔案路徑
d_file = {'_1_path': None, '_2_path': None, '_3_path': None, 
          '_4_path': None, '_5_path': None, '_6_path': None,
          '_7_path': None, '_8_path': None, '_9_path': None,
          '_10_path': None}

n_files=0
for n in d_file_path:
    d_file_path[n] = os.path.join(files_path, d_file_path[str(n)])
    d_file[n] = sorted(glob(d_file_path[n]))
    n_files = len(d_file[n]) + n_files       

#print(d_file_path)
#print(d_file)
print('Totally images:',n_files) #取得圖片總數
print('Classes:',len(d_file))

# 指定訓練圖片的格式
X = np.zeros((n_files, img_size, img_size, 3), dtype='float32')
y = np.zeros(n_files, dtype='int16')
count = 0

# 重新調整照片尺寸並給予標籤
class_label=0
for i in d_file:
    for f in d_file[i]:    
            img = io.imread(f)
            new_img = imresize(img, (img_size, img_size, 3))
            X[count] = np.array(new_img)
            y[count] = class_label
            count += 1
    class_label = class_label + 1
#print(y)

#MLP資料預處理(二維)
#Reshape成全連接成input的2維陣列
X_MLP = X.reshape((n_files,-1))
print('MLP shape:', X_MLP.shape)

#Split and shuffle dataset
X_MLP, X_MLP_test, Y, Y_test = train_test_split(X_MLP, y, test_size=0.2, random_state=42, shuffle=True)
MLP_train_data, MLP_test_data = X_MLP/255, X_MLP_test/255
MLP_train_target, MLP_test_target = Y, Y_test

np.save('MLP_train_feature.npy', MLP_train_data)
np.save('MLP_train_target.npy', MLP_train_target)
np.save('MLP_test_feature.npy', MLP_test_data)
np.save('MLP_test_target.npy', MLP_test_target)

print('MLP_number of train_data:', len(MLP_train_data))
print('MLP_number of test_data:', len(MLP_test_data))
print('MLP_train_data:')
print(MLP_train_data)
print('MLP_test_data:')
print(MLP_test_data)
print('MLP_train_target:')
print(MLP_train_target)
print('MLP_test_target:')
print(MLP_test_target)


#CNN資料預處理(四維)
X_CNN = X
print('CNN shape:', X_CNN.shape)

#Split and shuffle dataset
X_CNN, X_CNN_test, Y, Y_test = train_test_split(X_CNN, y, test_size=0.2, random_state=42, shuffle=True)
CNN_train_data, CNN_test_data = X_CNN/255, X_CNN_test/255
CNN_train_target, CNN_test_target = Y, Y_test

np.save('CNN_train_feature.npy', CNN_train_data)
np.save('CNN_train_target.npy', CNN_train_target)
np.save('CNN_test_feature.npy', CNN_test_data)
np.save('CNN_test_target.npy', CNN_test_target)

print('CNN_number of train_data:', len(CNN_train_data))
print('CNN_number of test_data:', len(CNN_test_data))
print('CNN_train_data:')
print(CNN_train_data)
print('CNN_test_data:')
print(CNN_test_data)
print('CNN_train_target:')
print(CNN_train_target)
print('CNN_test_target:')
print(CNN_test_target)

input('')