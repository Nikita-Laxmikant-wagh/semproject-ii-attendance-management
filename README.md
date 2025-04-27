
# 📸 Multiple Face Recognition Attendance System

A desktop-based automated attendance system using **face recognition**. This project captures multiple classroom images, recognizes student faces using a trained model, and automatically marks attendance. Designed to work with large classrooms (up to 80 students) using uploaded or webcam-captured images.

---

## 🚀 Features

- **Desktop GUI** using Python's `Tkinter`
- **Face detection** with Haar Cascades
- **Face recognition** using LBPHFaceRecognizer
- Capture or upload up to **5 classroom images** per session
- Automatically detect and recognize **multiple student faces**
- Attendance is marked if a student appears in **any image**
- Attendance data saved to **CSV files**
- Maintain **student database and face training** records
- **Teacher login system** to access attendance features
- **Speech feedback** for system interactions
- Export **present and absent lists**

---

## 🧠 Tech Stack

| Component         | Technology         |
|------------------|--------------------|
| Language          | Python             |
| GUI               | Tkinter            |
| Face Detection    | Haar Cascade       |
| Face Recognition  | LBPH (OpenCV)      |
| Database          | SQLite / MySQL     |
| CSV Handling      | pandas / csv       |
| Speech Feedback   | pyttsx3            |
| Image Processing  | OpenCV             |

---

## 🏗️ Project Structure

```
📁 face_attendance_system/
├── images/                  # Saved classroom images
├── dataset/                 # Trained student images
├── trainer/                 # Face recognition model files
├── attendance/              # CSV attendance files
├── gui/                     # Tkinter UI files
├── haarcascade_frontalface_default.xml
├── train_model.py           # Training script
├── recognize_faces.py       # Face recognition and attendance marking
├── register_student.py      # New student registration
├── login.py                 # Login interface
├── main.py                  # Main GUI launcher
├── database.py              # SQLite/MySQL database handler
└── README.md                # You're here!
```

---

## 🔁 Project Workflow

1. **Register Students**: Run `register_student.py` to capture and save face data.
2. **Train Model**: Run `train_model.py` to train the recognizer with registered faces.
3. **Login**: Run `login.py` or directly `main.py` to access the GUI.
4. **Upload/Capture Images**: Capture or upload 5 classroom images via the GUI.
5. **Process Images**: `recognize_faces.py` detects and recognizes faces in the images.
6. **Mark Attendance**: Recognized students are marked present, data stored in DB and CSV.
7. **View Attendance**: GUI displays results, and CSV files are saved in `/attendance`.

---

## 📁 File Responsibilities

| File/Folder               | Responsibility |
|--------------------------|----------------|
| `main.py`                | Launches the main GUI |
| `login.py`               | Handles teacher login |
| `register_student.py`    | GUI for capturing new student faces |
| `train_model.py`         | Trains the LBPH face recognizer |
| `recognize_faces.py`     | Recognizes faces and marks attendance |
| `database.py`            | Handles database connections and queries |
| `haarcascade_frontalface_default.xml` | Haar cascade used for face detection |
| `/dataset/`              | Stores images for training |
| `/trainer/`              | Stores trained model (yml file) |
| `/images/`               | Stores classroom images |
| `/attendance/`           | Stores attendance CSV files |

---

## ⚙️ How To Run

### 1. Install dependencies
```bash
pip install opencv-python numpy pillow pyttsx3 pandas
```

### 2. Register Students
```bash
python register_student.py
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Start the App
```bash
python main.py
```

---

## 👨‍💻 Author

Developed by **Parag, Raj, Nikita, Tanvi **  
For project guidance or queries, feel free to reach out!
project mentor: Dr. U. M. Patil.

