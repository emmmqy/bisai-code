
from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
def img2vector(image):
    img = cv2.imread(image,0) 
    rows, cols = 112,92
    img=cv2.resize(img,(cols,rows)) 
    imgVector = np.zeros(( rows , cols))
    imgVector = np.reshape(img,  (rows , cols))
    return imgVector

    image_path=[]
    train_face =[]
    train_label = np.zeros(40 * k)  # [0,0,.....0](共40*k个0)
    test_face =[]
    test_label = np.zeros(40 * (10 - k))
    np.random.seed(0)
    sample = np.random.rand(10).argsort() + 1
    for i in range(40):  
        people_num = i + 1
        for j in range(10):  
            image = orlpath + '/s' + str(people_num) + '/' + str(sample[j]) + '.jpg'
            image_path.append(image)
            img = img2vector(image)
            if j < k:
                
                train_face.append(img)
                train_label[i * k + j] = people_num
            else:
               
                test_face.append(img)
                test_label[i * (10 - k) + (j - k)] = people_num

    return train_face, train_label, test_face, test_label,image_path

def cosdist(A,B):
    num = np.dot(A,B) 
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom 
    sim = 0.5 + 0.5 * cos 
    return sim

train_face, train_label, test_face, test_label,image_path = load_orl(7)  
rows, cols = train_face[0].shape  
num_train = len(train_face) 
num_test = len(test_face) 
data_train_new=np.zeros([num_train,n_hist])
for i in range(num_train):
    image=train_face[i]
    lbp = local_binary_pattern(image, n_points, radius)
    max_bins = int(lbp.max() + 1)
    a, _ = (np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins)))
    a=np.array(a)
    data_train_new[i,:]=a 
fig=plt.figure()
a=fig.add_subplot(1,3,1)
a.imshow(image)
plt.xlabel('原始图片')

a=fig.add_subplot(1,3,2)
a.imshow(lbp)#显示图片
plt.xlabel('lbp图片')
a=fig.add_subplot(1,3,3)
a.plot(data_train_new[-1,:])#显示图片


data_test_new=np.zeros((num_test,n_hist))
for i in range(num_test):
    image = test_face[i]
    lbp = local_binary_pattern(image, n_points, radius)
    max_bins = int(lbp.max() + 1)
    a, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    data_test_new[i,:]=np.array(a)
data_test_new = np.array(data_test_new)  # mat change to array
data_train_new = np.array(data_train_new)
result=[]
true_num = 0
for i in range(num_test):
    testFace = data_test_new[i]
    diffMat = data_train_new - np.tile(testFace, (num_train, 1))  
    sqDiffMat = diffMat ** 2                                  
    sqDistances = sqDiffMat.sum(axis=1)  
    sortedDistIndicies = sqDistances.argsort()  
    indexMin = sortedDistIndicies[0]
    result.append(train_label[indexMin])
    if train_label[indexMin] == test_label[i]:
        true_num += 1
    else:
        pass
    accuracy = float(true_num) / num_test
print('测试集识别精度为: %.2f%%' % (accuracy * 100))

plt.figure()
plt.scatter(np.linspace(0,num_test,num_test),result,c='red',label='预测标签')
plt.scatter(np.linspace(0,num_test,num_test),test_label,label='测试集真实标签')
plt.legend()
plt.title('测试集分类结果')
plt.show()
root = tk.Tk()
root.withdraw()
file_path1 = filedialog.askopenfilename()
label1=int(file_path1.split('/')[-2][1:])
img_object=cv2.imread(file_path1)
fig=plt.figure()
a=fig.add_subplot(1,2,1)
a.imshow(img_object)
plt.xlabel('目标')
root = tk.Tk()
root.withdraw()
file_path2 = filedialog.askopenfilename()
img_object=cv2.imread(file_path2)
test_face = img2vector(file_path2)
data_test_new=np.zeros((1,n_hist))
image = test_face
lbp = local_binary_pattern(image, n_points, radius)
a, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
data_test_new=np.array(a)
data_test_new = np.array(data_test_new) 
diffMat = data_train_new - np.tile(data_test_new, (num_train, 1))  
sqDiffMat = diffMat ** 2                                  
sqDistances = sqDiffMat.sum(axis=1)  
sortedDistIndicies = sqDistances.argsort()  
indexMin = sortedDistIndicies[0]  
label2=train_label[indexMin]
a=fig.add_subplot(1,2,2)
a.imshow(img_object)
plt.xlabel('待测')
if label1==label2:
    plt.suptitle('匹配成功')
    print('匹配成功')
else:
    plt.suptitle('匹配失败')
    print('匹配失败')
root = tk.Tk()
root.withdraw()
file_path1 = filedialog.askopenfilename()# 选择图片
img_object=cv2.imread(file_path1)# 读取图片
test_face1 = img2vector(file_path1)
data_test_new=np.zeros((2,n_hist))
image = test_face1
lbp1 = local_binary_pattern(image, n_points, radius)
max_bins = int(lbp1.max() + 1)
a, _ = np.histogram(lbp1, normed=True, bins=max_bins, range=(0, max_bins))
data_test_new[0,:]=a
fig=plt.figure()
a=fig.add_subplot(1,2,1)
a.imshow(img_object)
plt.xlabel('目标')
root = tk.Tk()
root.withdraw()
file_path2 = filedialog.askopenfilename()
img_object=cv2.imread(file_path2)
test_face2 = img2vector(file_path2)
image = test_face2
lbp2 = local_binary_pattern(image, n_points, radius)
max_bins = int(lbp2.max() + 1)
a, _ = np.histogram(lbp2, normed=True, bins=max_bins, range=(0, max_bins))
data_test_new[1,:]=a
distance=cosdist(data_test_new[0,:],data_test_new[1,:])
alpha=0.7
a=fig.add_subplot(1,2,2)
a.imshow(img_object)
plt.xlabel('待测')
if distance>alpha:
    plt.suptitle('匹配成功')
    print('匹配成功')
else:
    plt.suptitle('匹配失败')
    print('匹配失败')



