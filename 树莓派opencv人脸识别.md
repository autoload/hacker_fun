原文地址：https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826

步骤

1）人脸检测和数据收集；

2）训练识别器；

3）人脸识别。


1 材料清单
树莓派 V3
500 万像素 1080p 传感器 OV5647 迷你摄像头模块

2 安装OpenCV包
树莓派系统，设置好源，安装opencv 及对应版本numpy 
cv2.__version__ 确定安装成功

3 测试摄像头

```
import cv2
cap=cv2.VideoCapture(0) #调用摄像头‘0'一般是打开电脑自带摄像头，‘1'是打开外部摄像头（只有一个摄像头的情况）
width=1280
height=960
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)#设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)#设置图像高度
#显示图像
while True: 
  ret,frame=cap.read()#读取图像(frame就是读取的视频帧，对frame处理就是对整个视频的处理)
  #print(ret)#
  #######例如将图像灰度化处理，
  img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#转灰度图
  cv2.imshow("img",img)
  ########图像不处理的情况
  cv2.imshow("frame",frame)  
 
  input=cv2.waitKey(20)
  if input==ord('q'):#如过输入的是q就break，结束图像显示，鼠标点击视频画面输入字符
    break
  
cap.release()#释放摄像头
cv2.destroyAllWindows()#销毁窗口
```

4 人脸检测

Haar 级联分类器

```
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('/home/pi/.local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/pi/.local/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')

cap=cv2.VideoCapture(0) #调用摄像头‘0'一般是打开电脑自带摄像头，‘1'是打开外部摄像头（只有一个摄像头的情况）
width=1280
height=960
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)#设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)#设置图像高度
#显示图像
while True: 
  ret,img=cap.read()#读取图像(frame就是读取的视频帧，对frame处理就是对整个视频的处理)
  #print(ret)#
  #img = cv2.flip(img,-1)
  #######例如将图像灰度化处理，
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转灰度图
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  cv2.imshow("video",img)
  input=cv2.waitKey(20)
  if input==ord('q'):#如过输入的是q就break，结束图像显示，鼠标点击视频画面输入字符
      break
cap.release()#释放摄像头
cv2.destroyAllWindows()#销毁窗口
```

5 收集数据
新建人脸识别项目目录

新建 dtatset目录，并用它来储存人脸样本

```
import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('/home/pi/.local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
face_id = input('n enter user id end press <return> ==> ')
print("n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    count += 1
# Save the captured image into the datasets folder
    cv2.imwrite("dataset/User."+ str(face_id) + '.'+ str(count) + ".jpg", gray[y:y+h,x:x+w])
    cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff# Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
        break
# Do a bit of cleanup
print("n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

```

6 训练
新建trainer目录 训练后保存为.yml 文件放在目录中

```
import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("/home/pi/.local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
```

7 识别器
识别器将会返回其预测的 id 和索引，并展示识别器对于该判断有多大的信心
```
import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "/home/pi/.local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Wan'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
```

8 结语
感谢开源与分享
