import tkinter as tk
from tkinter import filedialog, dialog
import tkinter.messagebox
import os
import cv2
import numpy as n
from Fusion import VIFusion

window = tk.Tk()
window.title('图像配准融合')
window.geometry('350x400+500+250')

img_right = None
img_left = None
s = None

def open_right():
    global img_right
    if img_right is None:
        file_path = filedialog.askopenfilename(title=u'选择图片', initialdir=(os.path.expanduser(r'img')))
        img_right = cv2.imread(file_path)
        tk.messagebox.showinfo(title='', message='读取成功')
    else:
        tk.messagebox.showwarning(title='WARNING', message='已经读过了')


def open_left():
    global img_left
    if img_left is None:
        file_path = filedialog.askopenfilename(title=u'选择图片', initialdir=(os.path.expanduser(r'img')))
        img_left = cv2.imread(file_path)
        tk.messagebox.showinfo(title='', message='读取成功')
    else:
        tk.messagebox.showwarning(title='WARNING', message='已经读过了')

def wtimage():
    global s
    if img_left is None or img_right is None:
        tk.messagebox.showwarning(title='WARNING', message='请先读取图片')
    elif s is None:
        # tk.messagebox.showinfo(title='', message='运行时间较长，\n完成后会有提醒，\n请勿进行其余操作！')
        fusion = VIFusion()
        A = fusion.fusion(arr_visible=img_right,arr_infrared=img_left)
        A = n.where(A > 255.0, 255.0,A )
        A= n.where(A < 0.0, 0.0, A)
        s = A
        tk.messagebox.showinfo(title='', message='运行成功')
    else:
        tk.messagebox.showinfo(title='', message='已计算完毕')

def show_result():
    global s
    if s is None:
        tk.messagebox.showwarning(title='WARNING', message='请先进行配准')
    else:
        s = cv2.cvtColor(s.astype(n.uint8), cv2.COLOR_BGR2RGB)
        cv2.imshow('result', s / 255)
        cv2.waitKey(0)

def save_file():
    global s
    if s is None:
        tk.messagebox.showwarning(title='WARNING', message='请先进行配准')
    else:
        file_path = filedialog.asksaveasfilename(title=u'选择文件夹', initialdir=(os.path.expanduser(
            '../../../Desktop/Image-stitching-based-on-sift-master')))
        cv2.imwrite(file_path, s.result)
        tk.messagebox.showinfo(title='', message='保存成功')

def clear_all():
    global s
    global img_right
    global img_left
    s = img_right = img_left = None


l = tk.Label(window, text=u'基于小波变换的图像配准', font=('systemfixed', 17), width=30, height=2)
l.place(x=21, y=30, anchor='nw')
bt_left = tk.Button(window, text='选择红外图片', bg='white', font=('systemfixed', 14), command=open_left)
bt_right = tk.Button(window, text='选择可见光图片', bg='white', font=('systemfixed', 14), command=open_right)
bt_sift = tk.Button(window, text='小波变换处理', bg='white', font=('systemfixed', 14), command=wtimage)
# bt_showL = tk.Button(window, text='显示左图关键点', bg='white', font=('systemfixed', 14), command=show_left)
# bt_showR = tk.Button(window, text='显示右图关键点', bg='white', font=('systemfixed', 14), command=show_right)
# bt_show = tk.Button(window, text='显示关键点匹配', bg='white', font=('systemfixed', 14), command=show_match)
bt_show2 = tk.Button(window, text='图像配准', bg='white', font=('systemfixed', 14), command=show_result)
bt_show3 = tk.Button(window, text='保存结果', bg='white', font=('systemfixed', 14), command=save_file)
bt_all = tk.Button(window, text='清空', bg='white', font=('systemfixed', 14), command=clear_all)
bt_show4 = tk.Button(window, text='关闭', bg='white', font=('systemfixed', 14), command=window.destroy)
bt_left.place(x=60, y=100, anchor='nw')
bt_right.place(x=200, y=100, anchor='nw')
bt_sift.place(x=150, y=150, anchor='nw')
# bt_showL.place(x=40, y=200, anchor='nw')
# bt_showR.place(x=200, y=200, anchor='nw')
# bt_show.place(x=120, y=250, anchor='nw')
bt_show2.place(x=150, y=200, anchor='nw')
bt_show3.place(x=150, y=250, anchor='nw')
bt_all.place(x=170, y=300, anchor='nw')
bt_show4.place(x=170, y=350, anchor='nw')


window.mainloop()