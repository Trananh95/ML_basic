import numpy as np
import cv2
import Process_Image
from PIL import Image
class Neural_network():
    learning_rate = 0.5
    def sig(self,x):
        return (1/(1+np.exp(-x)))

    def dsig(self,y):
        return (y*(1-y))

    def __init__(self, input, output, wm1, wm2):
        self.input = input
        self.target_output = output
        self.wm1 = wm1          #weight matrix in layer 1
        self.wm2 = wm2          # weight matrix in layer 2

    def forward_layer1(self, vector):
        neuron1 = np.dot(self.wm1, vector)
        return self.sig(neuron1)

    def forward_layer2(self, vector):
        input_layer2 = self.forward_layer1(vector)
        neuron2 = np.dot(self.wm2, input_layer2)
        return self.sig(neuron2)

    def backpropagation(self):
        # gia su o1 la 1*3
        o1 = self.forward_layer1(self.input)
        #gia su o2 la 1*2
        o2 = self.forward_layer2(self.input)
        #delta_2 la 1*2
        sens_2 = self.dsig(o2) * (self.target_output - o2)
        #wm2 la 3*2
        self.wm2 += self.learning_rate*np.dot(sens_2, o1.T)
        #input la 1*2
        #wm1 la 2*3
        #delta_1 phai la 1*3
        sens_1 = self.dsig(o1)*(np.dot(self.wm2.T, sens_2))
        self.wm1 += self.learning_rate*np.dot(sens_1, self.input.T)

    def predict_sample(self, vector):
        x = -1000
        real_output = self.forward_layer2(vector)
        for i in range (0, real_output.size):
            if (x < real_output.item(i)) :
                self.classifier = i
                x = real_output.item(i)
        if self.classifier == 0:
            return ("happy")
        else:
            return ("sad")

# test image
# wm1 = np.random.rand(64,256)
# wm2 = np.random.rand(2,64)
# target_input = Process_Image.get_matrix_sample_input()
# target_output = Process_Image.get_matrix_sample_output()
# print (target_output)
# z = 0
# neural = Neural_network(target_input, target_output, wm1 , wm2)
# while (z<100):
#     neural.backpropagation()
#     z += 1
# file = 'C:/Users/My PC/Desktop/PycharmProjects/GUI/happiness/2.jpg'
# image = cv2.imread(file)
# resized_image = cv2.resize(image, (16,16))
# new_img = Image.fromarray(resized_image, "RGB")
# new_img = new_img.convert('L')
# img_matrix = np.asarray(new_img)
# # vector nay la dai dien cho 1 anh
# vector = np.array(img_matrix).flatten()
# # chuan hoa vector
# scalar = np.linalg.norm(vector)
# vector = np.true_divide(vector , scalar)
# print(vector)
# print (neural.forward_layer2(vector))
# print (neural.predict_sample(vector))

# test network
# input =  np.array([[0,0],[3,2],[2,3],[1,1]]).T
# output = np.array([[0,1],[1,0],[1,0],[0,1]]).T
# wm1 = np.array([[0.1,0.4],[0.8,0.6]])
# wm2 = np.array([[0.3, 0.2], [0.9, 0.1]])
# z = 0
# neural = Neural_network(input, output, wm1 , wm2)
# while (z<10000):
#     neural.backpropagation()
#     z += 1
# print (wm2)
# vector = np.array([[1,1]]).T
# print (neural.forward_layer2(vector))
# print (neural.predict_sample(vector))

class Perceptron:
    def hardlim(self, x):
        if (x>0):
            return 1
        else:
            return 0

    def __init__(self, wm, input, output):
        self.wm = wm
        self.input = input
        self.target_output = output

    def forward_layer(self,input):
        neuron = np.dot(self.wm, input)
        for i in range (neuron.size):
            np.place(neuron, neuron == neuron.item(i), self.hardlim(neuron.item(i)))
        return neuron

    def modify_perceptron(self):
        for i in range(self.input.shape[1]): #.shape la tra ve hang * cot
            vector_input = np.array([self.input[:,i]])
            vector_target_output = np.array([self.target_output[:,i]])
            vector_real_output=  np.array([self.forward_layer(self.input)[:,i]])
            error = (vector_target_output - vector_real_output).T
            # print (error.shape)
            self.wm = self.wm + np.dot(error, vector_input)

    def predict_sample(self, vector):
        x = -1000
        for i in range(0, self.forward_layer(vector).size):
            if (x < self.forward_layer(vector).item(i)):
                self.classifier = i
                x = self.forward_layer(vector).item(i)
        if self.classifier == 0:
            return ("happy")
        else:
            return ("sad")

# wm = np.random.rand(2,256)
# # print(wm)
# target_input = Process_Image.get_matrix_sample_input()
# target_output = Process_Image.get_matrix_sample_output()
# perceptron = Perceptron(wm, target_input, target_output)
# z = 0
# while (z<1000):
#     perceptron.modify_perceptron()
#     z +=1
# print ("THANG!!@@#$")
# # print(perceptron.wm - wm)
# path_file = 'C:/Users/My PC/Desktop/PycharmProjects/GUI/sadness/2.jpg'
# image = cv2.imread(path_file)
# resized_image = cv2.resize(image, (16,16))
# new_img = Image.fromarray(resized_image, "RGB")
# new_img = new_img.convert('L')
# img_matrix = np.asarray(new_img)
# # vector nay la dai dien cho 1 anh
# vector = np.array(img_matrix).flatten()
# # chuan hoa vector
# scalar = np.linalg.norm(vector)
# vector = np.true_divide(vector , scalar)
# # print(vector)
# print (perceptron.forward_layer(vector))
# print(np.array([]).shape)