import cv2
import glob, os
import numpy as np
from PIL import Image
import scipy

path_happiness = "C:/Users/My PC/Desktop/PycharmProjects/GUI/happiness/"
path_sadness = "C:/Users/My PC/Desktop/PycharmProjects/GUI/sadness/"
path_sample = [path_happiness, path_sadness]

def get_matrix_sample_input():
    matrix_sample = np.random.rand(1,256)
    for path in path_sample:
        for file in glob.glob( os.path.join(path, "*.jpg") ):
            image = cv2.imread(file)
            resized_image = cv2.resize(image, (16,16))
    # ta doi tu rgb sang r
            new_img = Image.fromarray(resized_image, "RGB")
            new_img = new_img.convert('L')
            img_matrix = np.asarray(new_img)
    # vector nay la dai dien cho 1 anh
            vector = np.array(img_matrix).flatten()
    # chuan hoa vector
            scalar = np.linalg.norm(vector)
            vector = np.true_divide(vector , scalar)
    # matrix nay de luu tat ca cac anh trong mau vao 1 ma tran
    #cac thanh phan trong matrix nay da duoc chuan hoa
            matrix_sample = np.concatenate([matrix_sample, [vector]])
    matrix_sample = scipy.delete(matrix_sample,0 , 0)
    return matrix_sample.T

def get_matrix_sample_output():
    matrix_sample = np.random.rand(1,2)
    vector_hapiness = np.array([1,0])
    vector_sadness = np.array([0,1])
    for i in range(len(glob.glob(os.path.join(path_happiness, "*.jpg")))):
        matrix_sample = np.concatenate([matrix_sample, [vector_hapiness]])
    for j in range(len(glob.glob(os.path.join(path_sadness, "*.jpg")))):
        matrix_sample = np.concatenate([matrix_sample, [vector_sadness]])
    matrix_sample = scipy.delete(matrix_sample, 0 ,0)
    return matrix_sample.T

# print (get_matrix_sample_input().shape)
