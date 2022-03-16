import numpy as np
import pandas as pd
import cv2

from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image

import os
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


def extract_face(filename, required_size=(160, 160)):
    from mtcnn.mtcnn import MTCNN
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert("RGB")
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    if len(results) == 0:
        return []
    x1, y1, width, height = results[0]["box"]
    # deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def load_face(dir):
    faces = list()
    # enumerate files
    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path)
        faces.append(face)
    tmp = []
    for i in range(len(faces)):
        if faces[i] != []:
            tmp.append(faces[i])

    return tmp


def load_dataset(dir):
    # list for faces and labels
    X, y = list(), list()
    for subdir in os.listdir(dir):
        path = dir + subdir + "/"
        faces = load_face(path)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" % (len(faces), subdir))  # print progress
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


# load train dataset
trainX, trainy = load_dataset("C:/Users/catch/Desktop/Hackathon/Missing-Child-Identification-BTP-Thesis-master/dataset/train/")
# load test dataset

print(trainX.shape, trainy.shape)

testX, testy = load_dataset("C:/Users/catch/Desktop/Hackathon/Missing-Child-Identification-BTP-Thesis-master/dataset/val/")
print(testX.shape, testy.shape)

# save and compress the dataset for further use
np.savez_compressed("face_dataset_mo.npz", trainX, trainy, testX, testy)

##################################################### After saving


data = np.load("face_dataset_mo.npz")
trainX, trainy, testX, testy = (
    data["arr_0"],
    data["arr_1"],
    data["arr_2"],
    data["arr_3"],
)
print("Loaded: ", trainX.shape, trainy.shape, testX.shape, testy.shape)

facenet_model = load_model("facenet_keras.h5")
print("Loaded Model")


########################### Embeddings


def get_embedding(model, face):
    # scale pixel values
    face = face.astype("float32")
    # standardization
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]


# convert each face in the train set into embedding
emdTrainX = list()
for face in trainX:
    emd = get_embedding(facenet_model, face)
    emdTrainX.append(emd)

emdTrainX = np.asarray(emdTrainX)
print(emdTrainX.shape)

# convert each face in the test set into embedding
emdTestX = list()
for face in testX:
    emd = get_embedding(facenet_model, face)
    emdTestX.append(emd)
emdTestX = np.asarray(emdTestX)
print(emdTestX.shape)

# save arrays to one file in compressed format
np.savez_compressed(
    "face_embeddings_mo.npz", emdTrainX, trainy, emdTestX, testy
)
############################# Classification on whole dataset

embeddings = np.load("face_embeddings_mo.npz")
emdTrainX, trainy, emdTestX, testy = (
    embeddings["arr_0"],
    embeddings["arr_1"],
    embeddings["arr_2"],
    embeddings["arr_3"],
)


