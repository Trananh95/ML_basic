import numpy as np
import cv2
from tkinter import *
import tkinter.filedialog
import tkinter.messagebox
from PIL import Image, ImageDraw
import os, glob
import Process_Image
import Neural_Network

width_canvas = 120
height_canvas = 120
path_happiness = "C:/Users/My PC/Desktop/PycharmProjects/GUI/happiness/"
path_sadness = "C:/Users/My PC/Desktop/PycharmProjects/GUI/sadness/"

class Application(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()
        self.xold = None
        self.yold = None
        self.b1 = "up"
    # 2 bien nay de khoi tao cho neural network
        self.wm1 = np.random.rand(64, 256)
        self.wm2 = np.random.rand(2, 64)
        self.wm_perceptron = np.random.rand(2, 256)
    # ham nay de tao cac widget
        self.create_widget()
        self.draw_canvas()

# tao ra cac doi tuong trong giao dien
    def create_widget(self):
        # tai sao khi nhet frame vao thi lai khong xuathien duoc button
        self.frame = Frame(self, height= 350, width=400, background="pink")
        self.frame.pack()
        self.frame.pack_propagate(0)

        self.button1 = Button(self.frame,text = "CLEAR", command = self.DeleteAll)
        self.button1.pack()
        self.button1.place(height =40,width = 120 )

        self.button2 = Button(self.frame, text = "SAVE AS HAPPINESS", command = self.save_hapiness)
        self.button2.pack()
        self.button2.place(y = 41, height = 40, width = 120)

        self.button3 = Button(self.frame, text = "SAVE AS SADNESS", command = self.save_sadness)
        self.button3.pack()
        self.button3.place(y = 83, height = 40, width = 120)

        var = StringVar()
        self.button4 = Label( self.frame, textvariable= var)
        var.set("Error_value")
        self.button4.pack()
        self.button4.place(y=280, height=30, width=120)

        self.button5 = Button(self.frame, text = "Predict", command = self.predict_perceptron)
        self.button5.pack()
        self.button5.place(y = 320, height = 30, width =120)

        self.button6 = Button(self.frame, text = "GET SAMPLE", command = self.get_sample)
        self.button6.pack()
        self.button6.place( y = 160 , height = 40 , width = 120)

        self.button7 = Button(self.frame, text = "TRAINING", command = self.train_network)
        self.button7.pack()
        self.button7.place( y = 230, height = 40 , width =120)

        self.button8 = Button(self.frame, text = "LOAD IMAGE", command = self.load_image)
        self.button8.pack()
        self.button8.place( x= 280, height = 40, width = 120)

        self.text_error = Text(self.frame, bg = "white")
        self.text_error.pack()
        self.text_error.place(y = 280, x = 200, height = 30)

        self.text_hapiness = Text(self.frame,bg = "white")
        self.text_hapiness.pack()
        self.text_hapiness.place(x = 200, y = 160, height = 30)
        self.text_hapiness.insert(INSERT, "Happiness sample: ")

        self.text_sadness = Text(self.frame, bg="white")
        self.text_sadness.pack()
        self.text_sadness.place(x =200, y=190, height=30)
        self.text_sadness.insert(INSERT, "Sadness Sample: ")

        self.text_predict = Text(self.frame, bg = "white")
        self.text_predict.pack()
        self.text_predict.place(y = 320, x = 200, height = 30)

        self.image = Image.new("RGB", (width_canvas, height_canvas), (255,255,255))
        self.draw = ImageDraw.Draw(self.image)

# ham cho nut clear de xoa
    def DeleteAll(self):
        self.text_hapiness.delete("1.0", END)
        self.text_sadness.delete("1.0", END)
        self.text_hapiness.insert(INSERT, "Happiness sample: ")
        self.text_sadness.insert(INSERT, "Sadness Sample: ")
        self.text_predict.delete('1.0', END)
        self.text_error.delete('1.0', END)
        self.drawing_area.delete("all")
        self.image = Image.new("RGB", (width_canvas, height_canvas), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

# ham de save canvas:
    def save_hapiness(self):
        num_files = len([f for f in os.listdir(path_happiness)
                         if os.path.isfile(os.path.join(path_happiness, f))])
        file_name = path_happiness +  str(num_files + 1) + ".jpg"
        self.image.save(file_name)
        tkinter.messagebox.showinfo("Save", "Done!")

    def save_sadness(self):
        num_files = len([f for f in os.listdir(path_sadness)
                         if os.path.isfile(os.path.join(path_sadness, f))])
        file_name = path_sadness + str(num_files) + ".jpg"
        self.image.save(file_name)
        tkinter.messagebox.showinfo("Save", "Done!")

# ham de tra ve ma tran cac vector cua anh o mau
    def get_sample(self):
        # luu tat ca cac mau
        self.sample_input = Process_Image.get_matrix_sample_input()
        # luu dau ra cua cac mau
        self.sample_target_output = Process_Image.get_matrix_sample_output()
        self.text_hapiness.insert(INSERT, len(glob.glob(os.path.join(path_happiness, "*.jpg"))))
        self.text_sadness.insert(INSERT, len(glob.glob(os.path.join(path_sadness, "*.jpg" ))))
        # tkinter.messagebox.showinfo("Feedback", "Getting samples is Done!")

# ham nay de tai anh len va kiem tra mau
    def load_image(self):
        # filename = tkinter.filedialog.askopenfilename(filetypes=(("Template files", "*.tplate")
        #                                                    , ("HTML files", "*.html;*.htm")
        #                                                    , ("All files", "*.*")))
        # if filename:
        #     try:
        #         self.settings["template"].set(filename)
        #     except:
        #         tkinter.messagebox.showerror("Open Source File", "Failed to read file \n'%s'" % filename)
        # print("thang123456")

        path_image = tkinter.filedialog.askopenfilename()
        img = cv2.imread(path_image, 0)
        print(path_image)
        # return 1


# ham de train, dua vao neural network
    def train_network(self):
    # input la 256, output la 2 => wm1 :32*256, wm2 = 2*32
    #     self.Network = Neural_Network.Neural_network(self.sample_input, self.sample_target_output, self.wm1, self.wm2)
    #     iterate = 0
    #     while(iterate<10000):
    #         self.Network.backpropagation()
    #         iterate += 1
        self.Network = Neural_Network.Perceptron(self.wm_perceptron, self.sample_input, self.sample_target_output)
        iterate = 0
        while(iterate<1000):
            self.Network.modify_perceptron()
            iterate += 1
        tkinter.messagebox.showinfo("Feedback", "Training is successful!")

    # def predict_neural_network(self):
    #     path_file = 'C:/Users/My PC/Desktop/PycharmProjects/GUI/happiness/2.jpg'
    #     self.image = Image.open(path_file)
    #     self.image = self.image.convert('L')
    #     # print(self.image.size)
    # # vector input la vector cua anh ta can du doan
    #     vector_input = self.image.resize((16, 16), Image.ANTIALIAS)
    #     vector_input = np.asarray(vector_input)
    #     vector_input = np.array(vector_input).flatten()
    #     scalar = np.linalg.norm(vector_input)
    #     vector_input = np.true_divide(vector_input, scalar)
    #     vector_input = np.array([vector_input]).T
    #     # print (vector_input[5:10])
    #     face = self.Network.predict_sample(vector_input)
    #     self.text_predict.insert(INSERT, face)
    #     output = self.Network.forward_layer2(vector_input)
    #     print (output)
    #     if (face == "happy"):
    #         error_value = np.linalg.norm(output - np.array([[1, 0]]).T)
    #     else:
    #         error_value = np.linalg.norm(output - np.array([[0, 1]]).T)
    #     self.text_error.insert(INSERT, error_value)

    def predict_perceptron(self):
        self.image = self.image.convert('L')
        # vector input la vector cua anh ta can du doan
        vector_input = self.image.resize((16, 16), Image.ANTIALIAS)
        vector_input = np.asarray(vector_input)
        vector_input = np.array(vector_input).flatten()
        scalar = np.linalg.norm(vector_input)
        vector_input = np.true_divide(vector_input, scalar)
        vector_input = np.array([vector_input]).T
        print (vector_input)
        # print(np.linalg.norm(vector_input))
        face = self.Network.predict_sample(vector_input)
        self.text_predict.insert(INSERT, face)
        output = self.Network.forward_layer(vector_input)
        if (face == "happy"):
            error_value = np.linalg.norm(output - np.array([[1, 0]]).T)
        else:
            error_value = np.linalg.norm(output - np.array([[0, 1]]).T)
        self.text_error.insert(INSERT, error_value)

    # Cac ham duoi day deu la cho canvas ve
    def draw_canvas(self):
        self.drawing_area = Canvas(self.frame, width = width_canvas, height= height_canvas, bg = "white")
        self.drawing_area.pack()
        self.drawing_area.place(x = 120)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)

    def b1down(self,event):
        self.b1 = "down"  # you only want to draw when the button is down
        # because "Motion" events happen -all the time-

    def b1up(self, event):
        self.b1 = "up"
        self.xold = None  # reset the line when you let go of the button
        self.yold = None

    def motion(self, event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold, self.yold, event.x, event.y, smooth = 'true', width = 3)
                self.draw.line(((self.xold, self.yold), (event.x, event.y)), (0,0,0), width = 3)
                # here's where you draw it. smooth. neat.
        self.xold = event.x
        self.yold = event.y

root = Tk()
root.title("GUI")
app = Application(root)
root.mainloop()


