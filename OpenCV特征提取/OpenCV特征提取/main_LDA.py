import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
def img2vector(image):
    img = cv2.imread(image, 0) 
    rows, cols = 112,92  
    img=cv2.resize(img,(cols,rows)) 
    imgVector = np.zeros((1, rows * cols))
    imgVector = np.reshape(img, (1, rows * cols)
    return imgVector
    image_path=[]
    train_face = np.zeros((40 * k, 112 * 92))
    train_label = np.zeros(40 * k)  # [0,0,.....0](共40*k个0)
    test_face = np.zeros((40 * (10 - k), 112 * 92))
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
                train_face[i * k + j, :] = img
                train_label[i * k + j] = people_num
            else:
                test_face[i * (10 - k) + (j - k), :] = img
                test_label[i * (10 - k) + (j - k)] = people_num

    return train_face, train_label, test_face, test_label,image_path
def PCA(data, r):
    data = np.float32(np.mat(data))
    rows, cols = np.shape(data)
    data_mean = np.mean(data, 0)  
    A = data - np.tile(data_mean, (rows, 1))  
    C = A * A.T  
    D, V = np.linalg.eig(C)  
    V_r = V[:, 0:r]  
    V_r = A.T * V_r
    for i in range(r):
        V_r[:, i] = V_r[:, i] / np.linalg.norm(V_r[:, i])  
    final_data = A * V_r
    return final_data, data_mean, V_r
def LDA(X_train, y_train, k):
    m , n = X_train.shape
    class_wise_mean = []
    for i in range(1,41):
        idx = np.where(y_train==i)
        class_wise_mean.append(np.mean(X_train[idx], axis=0))
    within_SM = np.zeros((n, n))
    for i, mean_vector in zip(range(1, 41), class_wise_mean):
        class_wise_M = np.zeros((n, n))
        idx = np.where(y_train==i)
        for img in X_train[idx]:
            img, mean_vector = img.reshape(n, 1), mean_vector.reshape(n, 1)
            class_wise_M += (img - mean_vector).dot((img - mean_vector).T)
        within_SM += class_wise_M
    total_mean = np.mean(X_train, axis=0)
    between_SM = np.zeros((n, n))
    for i, mean_vector in enumerate(class_wise_mean):
        idx = np.where(y_train==i+1)
        cnt = X_train[idx].shape[0]
        mean_vector = mean_vector.reshape(n, 1)
        total_mean = total_mean.reshape(n, 1)
        between_SM += cnt * (mean_vector - total_mean).dot((mean_vector - total_mean).T)
    a=np.linalg.inv(within_SM).dot(between_SM)
    evals, evecs = np.linalg.eigh(a)
    idx = np.argsort(evals)[::-1][:k]
    evecs = evecs[:, idx]
    return np.dot(evecs.T, X_train.T).T, evecs
def cosdist(A,B):
    num = np.dot(A,B) 
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom 
    sim = 0.5 + 0.5 * cos 
    return sim
print('提取文件')
print('——————————————————————————————————————')

orlpath = "ORL/ORL"
r=10                      
result=[]
train_face, train_label, test_face, test_label,image_path = load_orl(7)  
num_train = train_face.shape[0]  
num_test = test_face.shape[0]  
data_train_new, data_mean, V_r = PCA(train_face, 279)
temp_face = test_face - np.tile(data_mean, (num_test, 1))
data_test_new = temp_face * V_r 
data_test = np.array(data_test_new)  # mat change to array
data_train= np.array(data_train_new)
data_train_new,evecs=LDA(data_train, train_label, r)
data_test_new=np.dot(evecs.T, data_test.T).T
print('计算测试集准确率')
true_num = 0
for i in range(num_test):
    testFace = data_test_new[i, :]
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
plt.figure()
plt.scatter(np.linspace(0,num_test,num_test),result,c='red',label='预测标签')
plt.scatter(np.linspace(0,num_test,num_test),test_label,label='测试集真实标签')
plt.legend()
plt.title('测试集分类结果')
plt.show()
print('当降维到%d时,测试集人脸识别精度为: %.2f%%' % (r,accuracy * 100))
root = tk.Tk()
root.withdraw()
file_path1 = filedialog.askopenfilename()

label1=int(file_path1.split('/')[-2][1:]) 
img_object=cv2.imread(file_path1)

fig=plt.figure()
a=fig.add_subplot(1,2,1)
a.imshow(img_object)
plt.xlabel('目标脸')
root = tk.Tk()
root.withdraw()
file_path2 = filedialog.askopenfilename()

img_object=cv2.imread(file_path2)
test_face = img2vector(file_path2)
temp_face = test_face - data_mean
data_test_new = temp_face * V_r  
data_test_new = np.array(data_test_new)  # mat change to array

data_test_new=np.dot(evecs.T, data_test_new.T).T



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
file_path1 = filedialog.askopenfilename()
img_object=cv2.imread(file_path1)
test_face1 = img2vector(file_path1)
fig=plt.figure()
a=fig.add_subplot(1,2,1)
a.imshow(img_object)
plt.xlabel('目标脸')
root = tk.Tk()
root.withdraw()
file_path2 = filedialog.askopenfilename()

img_object=cv2.imread(file_path2)
test_face2 = img2vector(file_path2)
test_face=np.vstack((test_face1,test_face2))

temp_face = test_face - data_mean
data_test_new = temp_face * V_r  
data_test_new = np.array(data_test_new) 

data_test_new=np.dot(evecs.T, data_test_new.T).T


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



