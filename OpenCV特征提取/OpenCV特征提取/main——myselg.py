import cv2
import numpy as np
import scipy.io as sio
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

P=sio.loadmat('PCA_data.mat')

def predict(data_train_new,data_mean,V_r,train_label,img):
    rows, cols = 112,92  
    test_face = np.reshape(img, (1, rows * cols))
    temp_face = np.array(test_face - data_mean)
    data_test_new =np.dot( temp_face , V_r)  
    num_train=data_train_new.shape[0]
    diffMat = data_train_new - np.tile(data_test_new, (num_train, 1))  
    sqDiffMat = diffMat ** 2                                  
    sqDistances = sqDiffMat.sum(axis=1) 
    
    sortedDistIndicies = sqDistances.argsort()  
    indexMin = sortedDistIndicies[0] 
    return train_label[0,indexMin]
def detect():
  face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
  camera = cv2.VideoCapture(0)
  counter=1
  while (True):
    ret, img = camera.read()
    faces = face_cascade.detectMultiScale(img, 1.3, 5)#捕获到的帧要转换为灰度图像
    counter = counter-1
    if counter==0:
        counter = 1
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#将框框加进图片帧中
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w, y:y+h]
            roi = cv2.resize(roi, (112, 92), interpolation=cv2.INTER_LINEAR)          
            params = predict(P['train'],P['data_mean'],P['V_r'],P['train_label'],roi)
            print(params)
            if params==40:
                params='Kong lingyu'
            cv2.putText(img, str(params), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.imshow("camera", img)
    
    if cv2.waitKey(1) & 0xff == ord("q"):
      break
  camera.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  detect()
