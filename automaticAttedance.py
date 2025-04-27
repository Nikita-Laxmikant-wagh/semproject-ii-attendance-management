import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import *
import os, cv2
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as tkk
import tkinter.font as font
import threading
import numpy as np
from imutils.object_detection import non_max_suppression

haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "TrainingImageLabel\\Trainner.yml"
trainimage_path = "TrainingImage"
studentdetail_path = "StudentDetails\\studentdetails.csv"
attendance_path = "Attendance"

# Enhanced face detection using multiple parameters
def detect_faces(image):
    # Create face detector
    face_cascade = cv2.CascadeClassifier(haarcasecade_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance image contrast
    gray = cv2.equalizeHist(gray)

    # Define multiple parameter combinations
    detection_params = [
        (1.05, 3, (20, 20)),
        (1.1, 3, (15, 15)),
        (1.15, 4, (20, 20)),
        (1.2, 5, (25, 25)),
        (1.3, 5, (30, 30)),
        (1.1, 5, (20, 20))
    ]

    all_faces = []

    # Run detection for each parameter set
    for scaleFactor, minNeighbors, minSize in detection_params:
        faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors, minSize=minSize)
        for face in faces:
            all_faces.append(face)

    # Remove duplicate/overlapping faces
    final_faces = []
    if len(all_faces) > 0:
        all_faces = np.array(all_faces)
        pick = non_max_suppression(all_faces, 0.3)
        final_faces = all_faces[pick].tolist()

    return final_faces

# Non-Maximum Suppression to remove overlapping face detections
def non_max_suppression(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []
    
    # Convert to floats
    boxes = boxes.astype("float")
    
    # Initialize the list of picked indexes
    pick = []
    
    # Grab the coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    # Compute area and sort by y2
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    # Process
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find overlapping boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # Compute overlap
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        
        # Delete overlapping boxes
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    return pick

# Try DNN face detection if available
def detect_faces_dnn(image):
    try:
        # Path to model files - adjust paths as needed
        modelFile = "models/opencv_face_detector_uint8.pb"
        configFile = "models/opencv_face_detector.pbtxt"
        
        # Check if model files exist
        if not (os.path.exists(modelFile) and os.path.exists(configFile)):
            return None  # Files don't exist, will use cascade instead
        
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                # Ensure coordinates are within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:  # Valid detection
                    faces.append([x1, y1, x2-x1, y2-y1])
        return faces
    except Exception as e:
        print(f"DNN detection error: {str(e)}")
        return None  # Error occurred, will use cascade instead

# for choose subject and fill attendance
def subjectChoose(text_to_speech):
    def selectImages():
        # Open file dialog to select multiple images
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if file_paths:
            # Update selected image count
            selected_img_label.configure(text=f"Selected: {len(file_paths)} images")
            # Display image names in the list
            image_listbox.delete(0, END)
            for path in file_paths:
                image_listbox.insert(END, os.path.basename(path))
            return file_paths
        return []

    def processImages(image_paths, recognizer, facecasCade, df):
        all_recognized_ids = set()  # To track unique recognized IDs
        recognition_results = []  # To store detailed results for each image
        
        # Create dataframe for attendance
        col_names = ["Enrollment", "Name"]
        attendance = pd.DataFrame(columns=col_names)
        
        for img_path in image_paths:
            # Read the image
            im = cv2.imread(img_path)
            if im is None:
                recognition_results.append((img_path, "Could not read image", 0, 0))
                continue
            
            # Try DNN detection first (if available)
            faces_dnn = detect_faces_dnn(im)
            
            # If DNN failed or not available, use cascade
            if faces_dnn is None or len(faces_dnn) == 0:
                faces = detect_faces(im)
            else:
                faces = faces_dnn
            
            if len(faces) == 0:
                recognition_results.append((img_path, "No faces detected", 0, 0))
                continue
            
            # Create a copy of the image for drawing
            result_image = im.copy()
            
            # Count faces and successful recognitions
            faces_detected = len(faces)
            faces_recognized = 0
            
            # Process each detected face
            for (x, y, w, h) in faces:
                try:
                    # Ensure the face region is within image boundaries
                    if y+h > im.shape[0] or x+w > im.shape[1]:
                        continue
                    
                    # Get face region and recognize
                    face_roi = cv2.cvtColor(im[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    Id, conf = recognizer.predict(face_roi)
                    
                    if conf < 80:  # Increased threshold for better recognition in class photos
                        faces_recognized += 1
                        all_recognized_ids.add(Id)
                        
                        # Get student name
                        aa = df.loc[df["Enrollment"] == Id]["Name"].values
                        if len(aa) > 0:
                            tt = f"{Id}-{aa[0]}"
                            # Add to attendance if not already present
                            if Id not in attendance["Enrollment"].values:
                                attendance.loc[len(attendance)] = [Id, aa[0]]
                            
                            # Mark recognized face on image
                            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(result_image, f"{Id}", (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            # ID not in database
                            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 165, 255), 2)
                            cv2.putText(result_image, f"ID:{Id}?", (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    else:
                        # Unrecognized face
                        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(result_image, "Unknown", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue
            
            # Save result image with '_recognized' suffix
            base_name, ext = os.path.splitext(img_path)
            result_path = f"{base_name}_recognized{ext}"
            cv2.imwrite(result_path, result_image)
            
            # Store recognition results
            recognition_results.append((img_path, result_path, faces_detected, faces_recognized))
        
        return attendance, all_recognized_ids, recognition_results

    def FillAttendance():
        sub = tx.get()
        if sub == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
            return
        
        # Get selected images
        image_paths = selectImages()
        if not image_paths:
            t = "No images selected!"
            text_to_speech(t)
            return
        
        try:
            # Initialize face recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            try:
                recognizer.read(trainimagelabel_path)
            except Exception as e:
                error_msg = "Model not found, please train model"
                Notifica.configure(
                    text=error_msg,
                    bg="black",
                    fg="yellow",
                    width=33,
                    font=("times", 15, "bold"),
                )
                Notifica.place(x=20, y=250)
                text_to_speech(error_msg)
                return
            
            # Load face cascade
            facecasCade = cv2.CascadeClassifier(haarcasecade_path)
            
            # Load student details
            try:
                df = pd.read_csv(studentdetail_path)
            except Exception as e:
                error_msg = "Student details file not found"
                Notifica.configure(
                    text=error_msg,
                    bg="black",
                    fg="yellow",
                    width=33,
                    font=("times", 15, "bold"),
                )
                Notifica.place(x=20, y=250)
                text_to_speech(error_msg)
                return
            
            # Show progress window
            progress_window = Toplevel()
            progress_window.title("Processing Images")
            progress_window.geometry("400x150")
            progress_window.configure(background="black")
            
            progress_label = Label(
                progress_window, 
                text="Processing images... Please wait.", 
                bg="black", 
                fg="yellow",
                font=("times", 14, "bold")
            )
            progress_label.pack(pady=20)
            
            progress_bar = ttk.Progressbar(
                progress_window,
                orient="horizontal",
                length=300,
                mode="indeterminate"
            )
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            # Update UI
            progress_window.update()
            
            # Process all selected images
            attendance, recognized_ids, recognition_results = processImages(
                image_paths, recognizer, facecasCade, df
            )
            
            # Close progress window
            progress_bar.stop()
            progress_window.destroy()
            
            if len(attendance) == 0:
                message = "No students recognized in any images!"
                Notifica.configure(
                    text=message,
                    bg="black",
                    fg="red",
                    width=33,
                    relief=RIDGE,
                    bd=5,
                    font=("times", 15, "bold"),
                )
                Notifica.place(x=20, y=250)
                text_to_speech(message)
                
                # Show results anyway
                show_results_window(recognition_results)
                return
            
            # Prepare attendance file
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            Hour, Minute, Second = timeStamp.split(":")
            
            # Add date column
            attendance[date] = 1
            
            # Save attendance to CSV
            path = os.path.join(attendance_path, sub)
            if not os.path.exists(path):
                os.makedirs(path)
            
            fileName = (
                f"{path}/"
                + sub
                + "_"
                + date
                + "_"
                + Hour
                + "-"
                + Minute
                + "-"
                + Second
                + ".csv"
            )
            
            attendance.to_csv(fileName, index=False)
            
            message = f"Attendance filled for {sub}. {len(recognized_ids)} students recognized."
            Notifica.configure(
                text=message,
                bg="black",
                fg="yellow",
                width=33,
                relief=RIDGE,
                bd=5,
                font=("times", 15, "bold"),
            )
            Notifica.place(x=20, y=250)
            text_to_speech(message)
            
            # Show results window
            show_results_window(recognition_results)
            
            # Display attendance results
            display_attendance_window(fileName, sub)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            Notifica.configure(
                text="An error occurred during processing",
                bg="black",
                fg="red",
                width=33,
                relief=RIDGE,
                bd=5,
                font=("times", 15, "bold"),
            )
            Notifica.place(x=20, y=250)
            text_to_speech("An error occurred while processing the images.")

    # NEW FUNCTION: Webcam attendance functionality
    def WebcamAttendance():
        sub = tx.get()
        if sub == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
            return
        
        try:
            # Initialize face recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            try:
                recognizer.read(trainimagelabel_path)
            except Exception as e:
                error_msg = "Model not found, please train model"
                Notifica.configure(
                    text=error_msg,
                    bg="black",
                    fg="yellow",
                    width=33,
                    font=("times", 15, "bold"),
                )
                Notifica.place(x=20, y=250)
                text_to_speech(error_msg)
                return
            
            # Load face cascade
            facecasCade = cv2.CascadeClassifier(haarcasecade_path)
            
            # Load student details
            try:
                df = pd.read_csv(studentdetail_path)
            except Exception as e:
                error_msg = "Student details file not found"
                Notifica.configure(
                    text=error_msg,
                    bg="black",
                    fg="yellow",
                    width=33,
                    font=("times", 15, "bold"),
                )
                Notifica.place(x=20, y=250)
                text_to_speech(error_msg)
                return
            
            # Create dataframe for attendance
            col_names = ["Enrollment", "Name"]
            attendance = pd.DataFrame(columns=col_names)
            
            # Open webcam window for face recognition
            webcam_window = Toplevel()
            webcam_window.title("Webcam Attendance")
            webcam_window.geometry("900x600")
            webcam_window.configure(background="#1c1c1c")
            webcam_window.protocol("WM_DELETE_WINDOW", lambda: stop_webcam())
            
            # Create frames for video and controls
            video_frame = Frame(webcam_window, bg="#1c1c1c")
            video_frame.grid(row=0, column=0, padx=20, pady=20)
            
            control_frame = Frame(webcam_window, bg="#1c1c1c")
            control_frame.grid(row=1, column=0, padx=20, pady=10)
            
            info_frame = Frame(webcam_window, bg="#1c1c1c")
            info_frame.grid(row=0, column=1, rowspan=2, padx=20, pady=20, sticky=N)
            
            # Create labels for video display
            video_label = Label(video_frame, bg="black")
            video_label.pack()
            
            # Create attendance list display
            attendance_label = Label(
                info_frame,
                text="Recognized Students:",
                bg="#1c1c1c",
                fg="yellow",
                font=("Verdana", 14, "bold")
            )
            attendance_label.pack(anchor=W, pady=(0, 10))
            
            # Frame for attendance list with scrollbar
            list_frame = Frame(info_frame, bg="#1c1c1c")
            list_frame.pack(fill=BOTH, expand=True)
            
            scrollbar = Scrollbar(list_frame)
            scrollbar.pack(side=RIGHT, fill=Y)
            
            attendance_listbox = Listbox(
                list_frame,
                bg="#333333",
                fg="yellow",
                font=("Verdana", 12),
                width=25,
                height=15,
                yscrollcommand=scrollbar.set
            )
            attendance_listbox.pack(side=LEFT, fill=BOTH, expand=True)
            scrollbar.config(command=attendance_listbox.yview)
            
            # Stats display
            stats_frame = Frame(info_frame, bg="#1c1c1c", pady=10)
            stats_frame.pack(fill=X)
            
            students_count_var = StringVar(value="Students: 0")
            students_count = Label(
                stats_frame,
                textvariable=students_count_var,
                bg="#1c1c1c",
                fg="light green",
                font=("Verdana", 12)
            )
            students_count.pack(anchor=W)
            
            # Control buttons
            start_button = Button(
                control_frame,
                text="Start Recognition",
                bg="#006400",
                fg="white",
                font=("Verdana", 12, "bold"),
                relief=RIDGE,
                command=lambda: start_recognition()
            )
            start_button.pack(side=LEFT, padx=10)
            
            stop_button = Button(
                control_frame,
                text="Stop Recognition",
                bg="#8B0000",
                fg="white",
                font=("Verdana", 12, "bold"),
                relief=RIDGE,
                command=lambda: stop_recognition()
            )
            stop_button.pack(side=LEFT, padx=10)
            
            save_button = Button(
                control_frame,
                text="Save Attendance",
                bg="#00008B",
                fg="white",
                font=("Verdana", 12, "bold"),
                relief=RIDGE,
                command=lambda: save_attendance()
            )
            save_button.pack(side=LEFT, padx=10)
            
            # Status label
            status_var = StringVar(value="Camera ready. Press 'Start Recognition' to begin.")
            status_label = Label(
                webcam_window,
                textvariable=status_var,
                bg="#1c1c1c",
                fg="white",
                font=("Verdana", 10)
            )
            status_label.grid(row=2, column=0, columnspan=2, pady=10)
            
            # Global variables for webcam operation
            webcam_running = False
            recognition_active = False
            cap = None
            recognized_ids = set()
            
            def update_attendance_display():
                attendance_listbox.delete(0, END)
                for _, row in attendance.iterrows():
                    attendance_listbox.insert(END, f"{row['Enrollment']} - {row['Name']}")
                students_count_var.set(f"Students: {len(attendance)}")
            
            def start_recognition():
                nonlocal webcam_running, recognition_active, cap
                if not webcam_running:
                    try:
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened():
                            status_var.set("Error: Could not open webcam.")
                            return
                        
                        webcam_running = True
                        recognition_active = True
                        status_var.set("Recognition active - Scanning for faces...")
                        video_loop()
                    except Exception as e:
                        status_var.set(f"Error: {str(e)}")
                else:
                    recognition_active = True
                    status_var.set("Recognition active - Scanning for faces...")
            
            def stop_recognition():
                nonlocal recognition_active
                recognition_active = False
                status_var.set("Recognition paused. Camera still active.")
            
            def stop_webcam():
                nonlocal webcam_running, cap
                webcam_running = False
                if cap is not None:
                    cap.release()
                webcam_window.destroy()
            
            def save_attendance():
                if len(attendance) == 0:
                    messagebox.showwarning("No Data", "No students have been recognized yet.")
                    return
                
                # Prepare attendance file
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                Hour, Minute, Second = timeStamp.split(":")
                
                # Add date column
                attendance[date] = 1
                
                # Save attendance to CSV
                path = os.path.join(attendance_path, sub)
                if not os.path.exists(path):
                    os.makedirs(path)
                
                fileName = (
                    f"{path}/"
                    + sub
                    + "_webcam_"
                    + date
                    + "_"
                    + Hour
                    + "-"
                    + Minute
                    + "-"
                    + Second
                    + ".csv"
                )
                
                attendance.to_csv(fileName, index=False)
                
                message = f"Attendance saved for {sub}. {len(attendance)} students recognized."
                status_var.set(message)
                messagebox.showinfo("Success", message)
                
                # Also display in main window
                Notifica.configure(
                    text=message,
                    bg="black",
                    fg="yellow",
                    width=33,
                    relief=RIDGE,
                    bd=5,
                    font=("times", 15, "bold"),
                )
                Notifica.place(x=20, y=250)
                text_to_speech(message)
                
                # Display attendance file
                display_attendance_window(fileName, sub)
                
            def process_frame(frame):
                # Create a working copy
                display_frame = frame.copy()
                
                # Try DNN detection first (if available)
                faces_dnn = detect_faces_dnn(frame)
                
                # If DNN failed or not available, use cascade
                if faces_dnn is None or len(faces_dnn) == 0:
                    faces = detect_faces(frame)
                else:
                    faces = faces_dnn
                
                # Process each detected face if recognition is active
                if recognition_active and len(faces) > 0:
                    for (x, y, w, h) in faces:
                        try:
                            # Draw rectangle around face
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                            
                            # If recognition is active, identify face
                            if recognition_active:
                                # Get face region and recognize
                                face_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                                Id, conf = recognizer.predict(face_roi)
                                
                                if conf < 80:  # Confidence threshold
                                    # Get student name
                                    aa = df.loc[df["Enrollment"] == Id]["Name"].values
                                    if len(aa) > 0:
                                        name = aa[0]
                                        
                                        # Add to attendance if not already present
                                        if Id not in recognized_ids:
                                            recognized_ids.add(Id)
                                            if Id not in attendance["Enrollment"].values:
                                                attendance.loc[len(attendance)] = [Id, name]
                                                # Update display
                                                update_attendance_display()
                                        
                                        # Mark recognized face on display
                                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                        cv2.putText(display_frame, f"{Id}-{name}", (x, y-10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    else:
                                        # ID not in database
                                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                                        cv2.putText(display_frame, f"ID:{Id}?", (x, y-10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                                else:
                                    # Unknown face
                                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                    cv2.putText(display_frame, "Unknown", (x, y-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        except Exception as e:
                            print(f"Error processing face: {str(e)}")
                            continue
                    
                return display_frame
            
            def video_loop():
                if not webcam_running:
                    return
                
                success, frame = cap.read()
                if success:
                    # Process the frame
                    processed_frame = process_frame(frame)
                    
                    # Convert to format for tkinter
                    cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2image)
                    
                    # Resize to fit display area if needed
                    display_width = 640
                    display_height = 480
                    img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
                    
                    imgtk = ImageTk.PhotoImage(image=img)
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)
                
                # Schedule the next capture
                video_label.after(10, video_loop)
            
            # Update initial attendance display
            update_attendance_display()
            
        except Exception as e:
            error_msg = f"Error initializing webcam: {str(e)}"
            print(error_msg)
            Notifica.configure(
                text="Error initializing webcam",
                bg="black",
                fg="red",
                width=33,
                relief=RIDGE,
                bd=5,
                font=("times", 15, "bold"),
            )
            Notifica.place(x=20, y=250)
            text_to_speech("An error occurred while starting the webcam.")

    def show_results_window(results):
        """Display image processing results"""
        results_window = Toplevel()
        results_window.title("Image Processing Results")
        results_window.geometry("800x500")
        results_window.configure(background="#1c1c1c")
        
        # Create header
        header_frame = Frame(results_window, bg="#1c1c1c")
        header_frame.pack(fill=X, pady=10)
        
        Label(
            header_frame, 
            text="Recognition Results", 
            bg="#1c1c1c",
            fg="yellow",
            font=("Verdana", 16, "bold")
        ).pack()
        
        # Create results frame with scrollbar
        results_frame = Frame(results_window, bg="#1c1c1c")
        results_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)
                       
        # Show results window continued
        scrollbar = Scrollbar(results_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        result_canvas = Canvas(results_frame, bg="#1c1c1c", yscrollcommand=scrollbar.set)
        result_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        scrollbar.config(command=result_canvas.yview)
        
        content_frame = Frame(result_canvas, bg="#1c1c1c")
        result_canvas.create_window((0, 0), window=content_frame, anchor=NW)
        
        row = 0
        for original_path, result_info, faces_detected, faces_recognized in results:
            # Image name
            img_name = os.path.basename(original_path)
            img_frame = Frame(content_frame, bg="#1c1c1c", pady=10)
            img_frame.grid(row=row, column=0, sticky=W)
            
            Label(
                img_frame, 
                text=f"Image: {img_name}", 
                bg="#1c1c1c",
                fg="white",
                font=("Verdana", 12)
            ).grid(row=0, column=0, sticky=W)
            
            # If result is a path (processed image)
            if isinstance(result_info, str) and os.path.exists(result_info):
                stats_text = f"Faces detected: {faces_detected}, Recognized: {faces_recognized}"
                Label(
                    img_frame, 
                    text=stats_text, 
                    bg="#1c1c1c",
                    fg="light green",
                    font=("Verdana", 11)
                ).grid(row=1, column=0, sticky=W)
                
                # Button to view result image
                Button(
                    img_frame,
                    text="View Processed Image",
                    command=lambda path=result_info: os.startfile(path),
                    bg="#333333",
                    fg="yellow",
                    font=("Verdana", 10),
                    relief=RIDGE,
                    bd=3
                ).grid(row=2, column=0, sticky=W, pady=5)
            else:
                # Error message
                Label(
                    img_frame, 
                    text=result_info, 
                    bg="#1c1c1c",
                    fg="orange red",
                    font=("Verdana", 11)
                ).grid(row=1, column=0, sticky=W)
            
            # Separator
            Frame(
                content_frame, 
                height=1, 
                bg="gray50"
            ).grid(row=row+1, column=0, sticky=EW, pady=5)
            
            row += 2
        
        # Update scrollregion after all items are added
        content_frame.update_idletasks()
        result_canvas.config(scrollregion=result_canvas.bbox("all"))
        
        # Button to close
        Button(
            results_window,
            text="Close",
            command=results_window.destroy,
            bg="black",
            fg="yellow",
            font=("Verdana", 12, "bold"),
            relief=RIDGE,
            bd=5,
            width=10
        ).pack(pady=10)

    def display_attendance_window(filename, subject_name):
        """Display attendance results in a window"""
        attendance_window = Toplevel()
        attendance_window.title(f"Attendance of {subject_name}")
        attendance_window.configure(background="black")
        
        try:
            with open(filename, newline="") as file:
                reader = csv.reader(file)
                r = 0
                
                for col in reader:
                    c = 0
                    for row in col:
                        label = Label(
                            attendance_window,
                            width=10,
                            height=1,
                            fg="yellow",
                            font=("times", 15, " bold "),
                            bg="black",
                            text=row,
                            relief=RIDGE,
                        )
                        label.grid(row=r, column=c)
                        c += 1
                    r += 1
        except Exception as e:
            Label(
                attendance_window,
                text=f"Error reading attendance file: {str(e)}",
                fg="red",
                bg="black",
                font=("times", 15, "bold")
            ).pack(pady=20)

    def Attf():
        """Browse attendance files"""
        sub = tx.get()
        if sub == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
        else:
            path = f"Attendance\\{sub}"
            if os.path.exists(path):
                os.startfile(path)
            else:
                messagebox.showerror("Error", f"No attendance records found for {sub}")

    # Window frame for subject chooser
    subject = Tk()
    subject.title("Attendance System")
    subject.geometry("780x550")  # Increased height to accommodate webcam button
    subject.resizable(0, 0)
    subject.configure(background="#1e3d59")
    
    titl = tk.Label(subject, bg="#1c1c1c", relief=RIDGE, bd=10, font=("Verdana", 30, "bold"))
    titl.pack(fill=X)
    
    titl = tk.Label(
        subject,
        text="Classroom Attendance System",
        bg="#1e3d59",
        fg="green",
        font=("Verdana", 25, "bold"),
    )
    titl.place(x=170, y=12)
    
    # Subject entry section
    subject_frame = Frame(subject, bg="#1e3d59")
    subject_frame.place(x=50, y=100)
    
    sub_label = tk.Label(
        subject_frame,
        text="Enter Subject",
        width=12,
        height=2,
        bg="#1e3d59",
        fg="yellow",
        bd=5,
        relief=RIDGE,
        font=("Verdana", 14),
    )
    sub_label.grid(row=0, column=0, padx=10)
    
    tx = tk.Entry(
        subject_frame,
        width=15,
        bd=5,
        bg="#1e3d59",
        fg="yellow",
        relief=RIDGE,
        font=("Verdana", 20, "bold"),
    )
    tx.grid(row=0, column=1, padx=20)
    
    # Image selection section
    image_frame = Frame(subject, bg="#1e3d59")
    image_frame.place(x=50, y=170)
    
    # Selected image info
    selected_img_label = tk.Label(
        image_frame,
        text="No images selected",
        bg="#1e3d59",
        fg="white",
        font=("Verdana", 12),
    )
    selected_img_label.grid(row=0, column=1, padx=10, sticky=W)
    
    # List of selected images
    list_frame = Frame(subject, bg="#1e3d59")
    list_frame.place(x=50, y=240)
    
    list_label = tk.Label(
        list_frame,
        text="Selected Images:",
        bg="#1e3d59",
        fg="yellow",
        font=("Verdana", 12),
    )
    list_label.grid(row=0, column=0, sticky=W, pady=5)
    
    # Listbox with scrollbar for image names
    list_container = Frame(list_frame, bg="#1e3d59")
    list_container.grid(row=1, column=0)
    
    scrollbar = Scrollbar(list_container)
    scrollbar.pack(side=RIGHT, fill=Y)
    
    image_listbox = Listbox(
        list_container,
        bg="#1e3d59",
        fg="white",
        width=40,
        height=5,
        font=("Verdana", 10),
        yscrollcommand=scrollbar.set
    )
    image_listbox.pack(side=LEFT)
    scrollbar.config(command=image_listbox.yview)
    
    # Action buttons
    button_frame = Frame(subject, bg="#1e3d59")
    button_frame.place(x=50, y=370)
    
    fill_a = tk.Button(
        button_frame,
        text="Select Images & \nFill Attendance",
        command=FillAttendance,
        bd=7,
        font=("Verdana", 14),
        bg="#1e3d59",
        fg="yellow",
        height=2,
        width=15,
        relief=RIDGE,
    )
    fill_a.grid(row=0, column=0, padx=10)
    
    # NEW WEBCAM BUTTON
    webcam_btn = tk.Button(
        button_frame,
        text="Webcam \nAttendance",
        command=WebcamAttendance,
        bd=7,
        font=("Verdana", 14),
        bg="#1e3d59",
        fg="white",
        height=2,
        width=15,
        relief=RIDGE,
    )
    webcam_btn.grid(row=0, column=1, padx=10)
    
    attf = tk.Button(
        button_frame,
        text="Check \nAttendance Sheets",
        command=Attf,
        bd=7,
        font=("Verdana", 14),
        bg="#1e3d59",
        fg="yellow",
        height=2,
        width=15,
        relief=RIDGE,
    )
    attf.grid(row=0, column=2, padx=10)
    
    # Notification area
    Notifica = tk.Label(
        subject,
        text="",
        bg="#1c1c1c",
        fg="yellow",
        width=40,
        height=2,
        font=("times", 15, "bold"),
    )
    Notifica.place(x=120, y=450)
    
    subject.mainloop()