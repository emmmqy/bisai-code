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
    imgVector = np.reshape(img, (1, rows * cols))
    return imgVector
    image_path=[]
    train_face = np.zeros((40 * k, 112 * 92))
    train_label = np.zeros(40 * k,dtype=int)  # [0,0,.....0](共40*k个0)
    test_face = np.zeros((40 * (10 - k), 112 * 92))
    test_label = np.zeros(40 * (10 - k),dtype=int)
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
    n,m=P.shape
    if TYPE == 1:
        y=np.zeros([n,41])
        for i in range(n):
            y[i,T[i]]=1
        T=np.copy(y)
    n,c=T.shape
    Weights = 2*np.random.rand(m,N)-1
    biases=np.random.rand(1,N)
    temp=np.matmul(P,Weights)+np.tile(biases,[n,1])
    if TF=='sig':
        H=1/(1+np.exp(temp))
    w_out=np.matmul(np.matmul(np.linalg.pinv(np.matmul(H.T,H)+1/C),H.T),T)
    return Weights ,biases ,w_out, TF, TYPE
def train_predic(P,Weights,biases,w_out,TF,TYPE):
    n,m=P.shape
    temp=np.matmul(P,Weights)+np.tile(biases,[n,1])
    if TF=='sig':
        H=1/(1+np.exp(temp))
    T=np.matmul(H,w_out)
    if TYPE==1:
        T_predict=np.argmax(T,axis=1)
    return T_predict
def compute_accuracy(T_true,T_predict,TYPE):
    if TYPE==0:
        accuracy=np.mean(np.sum(T_true-T_predict))
    if TYPE==1:
        n=0
        for i in range(len(T_true)):
           if T_true[i]==T_predict[i]:
               n=n+1
        accuracy=n/len(T_true)
    return accuracy  
# In[3]:
print('提取文件')
print('——————————————————————————————————————')
orlpath = "ORL/ORL"
train_face, train_label, test_face, test_label,image_path = load_orl(7)  
num_train = train_face.shape[0]  
num_test = test_face.shape[0]  
Weights,biases,w_out,TF,TYPE=train_elm(train_face,train_label,N=2000,C=100000,TF='sig',TYPE=1)
T_predict=train_predic(test_face,Weights,biases,w_out,TF,TYPE)
accuracy=compute_accuracy(test_label,T_predict,TYPE)
plt.figure()
plt.scatter(np.linspace(0,num_test,num_test),T_predict,c='red',label='预测标签')
plt.scatter(np.linspace(0,num_test,num_test),test_label,label='测试集真实标签')
plt.legend()
plt.title('测试集分类结果')
plt.show()
print('测试集识别精度为: %.2f%%' % (accuracy * 100))
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
label2=train_predic(test_face,Weights,biases,w_out,TF,TYPE)
a=fig.add_subplot(1,2,2)
a.imshow(img_object)
plt.xlabel('待测')
if label1==label2:
    plt.suptitle('匹配成功')
    print('匹配成功')
else:
    plt.suptitle('匹配失败')
    print('匹配失败')





