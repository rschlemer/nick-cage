#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imutils import paths
import face_recognition
import pickle
import cv2
import os


# In[2]:


# dataset folder
iLocation = 'dataset'

# detection method
# cnn is more accurate but slower
# hog is less accurate but faster

detectionMethod = 'cnn'
# detectionMethod = 'hog'

# endcodings filename
encode = 'encodings.pickle'


# In[ ]:


iPaths = list(paths.list_images(iLocation))
kEncodings = []
names = []

for i, path in enumerate(iPaths):
    print(f'processing image {i+1}/{len(iPaths)}: {os.path.basename(path)}')
    name = path.split(os.path.sep)[-2]
    image = cv2.imread(path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, model=detectionMethod)    
    
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    for e in encodings:
        kEncodings.append(e)
        names.append(name)


# In[ ]:


print('serializing encodings')
data = {
    'encodings': encodings,
    'names': names
}
f = open(encode, 'wb')
f.write(pickle.dumps(data))
f.close()


# In[ ]:




