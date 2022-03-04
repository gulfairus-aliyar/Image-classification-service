from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
import json
from tensorflow import Graph, Session
import tensorflow as tf
from keras.datasets import cifar10
from imgApp.models import ImageFind

tf.keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


img_height, img_width=32,32
with open('./models/CNN.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/medical_trial_model.h5')

def index(request):
    context = {'a': 1}
    return render(request,'index.html',context)

def predictImage(request):
    print(request)
    print(request.POST.dict())
    context = {}
    if request.FILES:
        fileObj = request.FILES['filePath']
        fs = FileSystemStorage()
        filePathName = fs.save(fileObj.name, fileObj)
        filePathName = fs.url(filePathName)
        testimage = '.' + filePathName
        img = image.load_img(testimage, target_size=(img_height, img_width))
        x = image.img_to_array(img)
        x = x / 255
        x = x.reshape(1, img_height, img_width, 3)
        with model_graph.as_default():
            with tf_session.as_default():
                predi = model.predict(x)

        import numpy as np
        predictedLabel = labelInfo[str(np.argmax(predi[0]))]

        context = {'filePathName': filePathName, 'predictedLabel': predictedLabel[1]}
        image_find = ImageFind(image=context['filePathName'], classifier=context['predictedLabel'])
        image_find.save()
    return render(request, 'index.html', context)