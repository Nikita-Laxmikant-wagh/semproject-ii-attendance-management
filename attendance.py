import tkinter as tk
from tkinter import *
import os, cv2
import shutil
import csv
import numpy as np
from PIL import ImageTk, Image
import pandas as pd
import datetime
import time
import tkinter.font as font
import pyttsx3

# project module
import show_attendance
import takeImage
import trainImage
import automaticAttedance

def text_to_speech(user_text):
    engine = pyttsx3.init()
    engine.say(user_text)
    engine.runAndWait()


haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = (
    "./TrainingImageLabel/Trainner.yml"
)
trainimage_path = "/TrainingImage"
if not os.path.exists(trainimage_path):
    os.makedirs(trainimage_path)

studentdetail_path = (
    "./StudentDetails/studentdetails.csv"
)
attendance_path = "Attendance"

# Define a modern color scheme
PRIMARY_COLOR = "#1e3d59"  # Deep blue for main background
SECONDARY_COLOR = "#f5f0e1"  # Cream for text
ACCENT_COLOR = "#ff6e40"  # Orange for buttons and highlights
TEXT_COLOR = "#ffc13b"  # Gold for labels
BUTTON_BG = "#1e3d59"  # Deep blue for button background
BUTTON_FG = "#ffc13b"  # Gold for button text
INPUT_BG = "#2b4f76"  # Lighter blue for input fields
INPUT_FG = "#ffffff"  # White text for input fields
HIGHLIGHT_COLOR = "#ff6e40"  # Orange for highlights

window = Tk()
window.title("Face Recognition Attendance System")
window.geometry("1280x720")
dialog_title = "QUIT"
dialog_text = "Are you sure want to close?"
window.configure(background=PRIMARY_COLOR)

# Custom font styles
TITLE_FONT = ("Montserrat", 30, "bold")
HEADING_FONT = ("Montserrat", 24, "bold")
BUTTON_FONT = ("Montserrat", 16, "bold")
LABEL_FONT = ("Montserrat", 14)
INPUT_FONT = ("Montserrat", 16)
MESSAGE_FONT = ("Montserrat", 14)

# to destroy screen
def del_sc1():
    sc1.destroy()


# error message for name and no
def err_screen():
    global sc1
    sc1 = tk.Tk()
    sc1.geometry("400x110")
    sc1.iconbitmap("AMS.ico")
    sc1.title("Warning!!")
    sc1.configure(background=PRIMARY_COLOR)
    sc1.resizable(0, 0)
    tk.Label(
        sc1,
        text="Enrollment & Name required!!!",
        fg=ACCENT_COLOR,
        bg=PRIMARY_COLOR,
        font=HEADING_FONT,
    ).pack()
    tk.Button(
        sc1,
        text="OK",
        command=del_sc1,
        fg=SECONDARY_COLOR,
        bg=ACCENT_COLOR,
        width=9,
        height=1,
        activebackground="#e85d36",  # Darker orange for active state
        font=BUTTON_FONT,
        relief=FLAT,
        cursor="hand2",
    ).place(x=150, y=50)

def testVal(inStr, acttyp):
    if acttyp == "1":  # insert
        if not inStr.isdigit():
            return False
    return True

# Create a frame for the header
header_frame = Frame(window, bg=PRIMARY_COLOR, bd=0)
header_frame.pack(fill=X, pady=10)

# Logo and title in header
logo = Image.open("UI_Image/0001.png")
logo = logo.resize((60, 60), Image.LANCZOS)
logo1 = ImageTk.PhotoImage(logo)

l1 = tk.Label(header_frame, image=logo1, bg=PRIMARY_COLOR)
l1.pack(side=LEFT, padx=(480, 10))

titl = tk.Label(
    header_frame, 
    text="RCPIT", 
    bg=PRIMARY_COLOR, 
    fg=TEXT_COLOR, 
    font=TITLE_FONT
)
titl.pack(side=LEFT)

# Department heading with shadow effect
dept_frame = Frame(window, bg=PRIMARY_COLOR)
dept_frame.pack(fill=X, pady=5)

a = tk.Label(
    dept_frame,
    text="AIML Department",
    bg=PRIMARY_COLOR,
    fg=TEXT_COLOR,
    font=TITLE_FONT,
)
a.pack(pady=(0, 30))

# Create a main content frame
content_frame = Frame(window, bg=PRIMARY_COLOR)
content_frame.pack(expand=True, fill=BOTH, padx=50, pady=20)

# Create three frames for the options
option_frame1 = Frame(content_frame, bg=PRIMARY_COLOR, bd=2, relief=FLAT)
option_frame1.place(relx=0.1, rely=0.2, relwidth=0.2, relheight=0.4)

option_frame2 = Frame(content_frame, bg=PRIMARY_COLOR, bd=2, relief=FLAT)
option_frame2.place(relx=0.4, rely=0.2, relwidth=0.2, relheight=0.4)

option_frame3 = Frame(content_frame, bg=PRIMARY_COLOR, bd=2, relief=FLAT)
option_frame3.place(relx=0.7, rely=0.2, relwidth=0.2, relheight=0.4)

# Load and place images with hover effect
def on_enter(e, frame):
    frame.config(bg=HIGHLIGHT_COLOR)

def on_leave(e, frame):
    frame.config(bg=PRIMARY_COLOR)

# Register image
ri = Image.open("UI_Image/register.png")
ri = ri.resize((100, 100), Image.LANCZOS)  # Resize for consistency
r = ImageTk.PhotoImage(ri)
label1 = Label(option_frame1, image=r, bg=PRIMARY_COLOR)
label1.image = r
label1.pack(pady=20)
option_frame1.bind("<Enter>", lambda e: on_enter(e, option_frame1))
option_frame1.bind("<Leave>", lambda e: on_leave(e, option_frame1))

# Verify image
vi = Image.open("UI_Image/verifyy.png")
vi = vi.resize((100, 100), Image.LANCZOS)  # Resize for consistency
v = ImageTk.PhotoImage(vi)
label3 = Label(option_frame2, image=v, bg=PRIMARY_COLOR)
label3.image = v
label3.pack(pady=20)
option_frame2.bind("<Enter>", lambda e: on_enter(e, option_frame2))
option_frame2.bind("<Leave>", lambda e: on_leave(e, option_frame2))

# Attendance image
ai = Image.open("UI_Image/attendance.png")
ai = ai.resize((100, 100), Image.LANCZOS)  # Resize for consistency
a = ImageTk.PhotoImage(ai)
label2 = Label(option_frame3, image=a, bg=PRIMARY_COLOR)
label2.image = a
label2.pack(pady=20)
option_frame3.bind("<Enter>", lambda e: on_enter(e, option_frame3))
option_frame3.bind("<Leave>", lambda e: on_leave(e, option_frame3))

# Button styles
button_style = {
    "font": BUTTON_FONT,
    "bg": ACCENT_COLOR,
    "fg": SECONDARY_COLOR,
    "activebackground": "#e85d36",  # Darker orange for active state
    "activeforeground": SECONDARY_COLOR,
    "bd": 0,
    "relief": FLAT,
    "cursor": "hand2",
    "height": 2,
    "width": 17,
}

def TakeImageUI():
    ImageUI = Tk()
    ImageUI.title("Take Student Image")
    ImageUI.geometry("780x480")
    ImageUI.configure(background=PRIMARY_COLOR)
    ImageUI.resizable(0, 0)
    
    # Header frame
    header_frame = Frame(ImageUI, bg=PRIMARY_COLOR, height=80)
    header_frame.pack(fill=X)
    
    # Title with gradient-like effect
    titl = tk.Label(
        header_frame, 
        text="Register Your Face", 
        bg=PRIMARY_COLOR, 
        fg=TEXT_COLOR, 
        font=TITLE_FONT
    )
    titl.pack(pady=15)
    
    # Decorative line
    separator = Frame(ImageUI, height=2, bg=ACCENT_COLOR)
    separator.pack(fill=X, padx=50)

    # Content frame
    content_frame = Frame(ImageUI, bg=PRIMARY_COLOR)
    content_frame.pack(fill=BOTH, expand=True, padx=40, pady=20)
    
    # Header
    a = tk.Label(
        content_frame,
        text="Enter the details",
        bg=PRIMARY_COLOR,
        fg=SECONDARY_COLOR,
        font=HEADING_FONT,
    )
    a.grid(row=0, column=0, columnspan=2, pady=(0, 20))

    # ER no - using grid for better alignment
    lbl1 = tk.Label(
        content_frame,
        text="Enrollment No:",
        bg=PRIMARY_COLOR,
        fg=TEXT_COLOR,
        font=LABEL_FONT,
        anchor='e'
    )
    lbl1.grid(row=1, column=0, pady=10, sticky='e', padx=10)
    
    txt1 = tk.Entry(
        content_frame,
        width=17,
        validate="key",
        bg=INPUT_BG,
        fg=INPUT_FG,
        relief=FLAT,
        font=INPUT_FONT,
        insertbackground=SECONDARY_COLOR,  # Cursor color
    )
    txt1.grid(row=1, column=1, pady=10, padx=10, ipady=5)
    txt1["validatecommand"] = (txt1.register(testVal), "%P", "%d")

    # Name
    lbl2 = tk.Label(
        content_frame,
        text="Name:",
        bg=PRIMARY_COLOR,
        fg=TEXT_COLOR,
        font=LABEL_FONT,
        anchor='e'
    )
    lbl2.grid(row=2, column=0, pady=10, sticky='e', padx=10)
    
    txt2 = tk.Entry(
        content_frame,
        width=17,
        bg=INPUT_BG,
        fg=INPUT_FG,
        relief=FLAT,
        font=INPUT_FONT,
        insertbackground=SECONDARY_COLOR,  # Cursor color
    )
    txt2.grid(row=2, column=1, pady=10, padx=10, ipady=5)

    # Notification
    lbl3 = tk.Label(
        content_frame,
        text="Notification:",
        bg=PRIMARY_COLOR,
        fg=TEXT_COLOR,
        font=LABEL_FONT,
        anchor='e'
    )
    lbl3.grid(row=3, column=0, pady=10, sticky='e', padx=10)
    
    message = tk.Label(
        content_frame,
        text="",
        width=32,
        height=2,
        bg=INPUT_BG,
        fg=SECONDARY_COLOR,
        font=MESSAGE_FONT,
    )
    message.grid(row=3, column=1, pady=10, padx=10)

    # Buttons frame
    button_frame = Frame(content_frame, bg=PRIMARY_COLOR)
    button_frame.grid(row=4, column=0, columnspan=2, pady=20)
    
    def take_image():
        l1 = txt1.get()
        l2 = txt2.get()
        takeImage.TakeImage(
            l1,
            l2,
            haarcasecade_path,
            trainimage_path,
            message,
            err_screen,
            text_to_speech,
        )
        txt1.delete(0, "end")
        txt2.delete(0, "end")

    # Button style dictionary
    register_button_style = {
        "font": BUTTON_FONT,
        "bg": ACCENT_COLOR,
        "fg": SECONDARY_COLOR,
        "activebackground": "#e85d36",
        "activeforeground": SECONDARY_COLOR,
        "relief": FLAT,
        "cursor": "hand2",
        "width": 12,
        "height": 2,
        "bd": 0
    }
    
    # Take Image button
    takeImg = tk.Button(
        button_frame,
        text="Take Image",
        command=take_image,
        **register_button_style
    )
    takeImg.grid(row=0, column=0, padx=20)

    def train_image():
        trainImage.TrainImage(
            haarcasecade_path,
            trainimage_path,
            trainimagelabel_path,
            message,
            text_to_speech,
        )

    # Train Image button
    trainImg = tk.Button(
        button_frame,
        text="Train Image",
        command=train_image,
        **register_button_style
    )
    trainImg.grid(row=0, column=1, padx=20)


# Main buttons with hover effects
def on_enter_button(e, button):
    button.config(bg="#e85d36")  # Darker orange on hover

def on_leave_button(e, button):
    button.config(bg=ACCENT_COLOR)  # Return to normal color

# Register button
register_btn = tk.Button(
    window,
    text="Register New Student",
    command=TakeImageUI,
    **button_style
)
register_btn.place(relx=0.1, rely=0.7, anchor=W)
register_btn.bind("<Enter>", lambda e: on_enter_button(e, register_btn))
register_btn.bind("<Leave>", lambda e: on_leave_button(e, register_btn))

def automatic_attedance():
    automaticAttedance.subjectChoose(text_to_speech)

# Take Attendance button
attend_btn = tk.Button(
    window,
    text="Take Attendance",
    command=automatic_attedance,
    **button_style
)
attend_btn.place(relx=0.5, rely=0.7, anchor=CENTER)
attend_btn.bind("<Enter>", lambda e: on_enter_button(e, attend_btn))
attend_btn.bind("<Leave>", lambda e: on_leave_button(e, attend_btn))

def view_attendance():
    show_attendance.subjectchoose(text_to_speech)

# View Attendance button
view_btn = tk.Button(
    window,
    text="View Attendance",
    command=view_attendance,
    **button_style
)
view_btn.place(relx=0.9, rely=0.7, anchor=E)
view_btn.bind("<Enter>", lambda e: on_enter_button(e, view_btn))
view_btn.bind("<Leave>", lambda e: on_leave_button(e, view_btn))

# Exit button with different color
exit_btn = tk.Button(
    window,
    text="EXIT",
    command=quit,
    font=BUTTON_FONT,
    bg="#e74c3c",  # Red for exit button
    fg=SECONDARY_COLOR,
    activebackground="#c0392b",  # Darker red on active
    activeforeground=SECONDARY_COLOR,
    bd=0,
    relief=FLAT,
    cursor="hand2",
    height=2,
    width=10,
)
exit_btn.place(relx=0.5, rely=0.85, anchor=CENTER)
exit_btn.bind("<Enter>", lambda e: exit_btn.config(bg="#c0392b"))
exit_btn.bind("<Leave>", lambda e: exit_btn.config(bg="#e74c3c"))


window.mainloop()
