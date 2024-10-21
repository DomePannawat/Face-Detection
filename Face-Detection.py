import cv2
import mediapipe as mp
import os
import tkinter as tk
from tkinter import messagebox, font
import threading

# Variables, face detection and face border drawing
face_detector = mp.solutions.face_detection
face_drawer = mp.solutions.drawing_utils

#  GUI Tkinter
class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.root.geometry("300x200")  
        self.root.config(bg="#282c34") 
        self.camera = None
        self.is_running = False

        
        self.custom_font = font.Font(family="Helvetica", size=12)

        # Camera start/stop button
        self.btn_start = tk.Button(root, text="Start Camera", command=self.start_camera, bg="#61afef", fg="white", font=self.custom_font)
        self.btn_start.pack(pady=10)

        self.btn_stop = tk.Button(root, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED, bg="#e06c75", fg="white", font=self.custom_font)
        self.btn_stop.pack(pady=10)

        # Program close button
        self.btn_exit = tk.Button(root, text="Exit", command=self.exit_app, bg="#c678dd", fg="white", font=self.custom_font)
        self.btn_exit.pack(pady=10)

        # Check or create a folder to store data
        self.folder = "face_data"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Add a label to display information
        self.label_info = tk.Label(root, text="", bg="#282c34", fg="white", font=self.custom_font)
        self.label_info.pack(pady=5)

    # Camera start function
    def start_camera(self):
        self.camera = cv2.VideoCapture(0)
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.label_info.config(text="Camera is opening...", font=self.custom_font)

        threading.Thread(target=self.detect_faces).start()

    # Face detection function
    def detect_faces(self):
        with face_detector.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
            while self.is_running:
                ret, frame = self.camera.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(frame_rgb)

                # Draw a face frame and display face detection data.
                if results.detections:
                    for idx, detection in enumerate(results.detections):
                        face_drawer.draw_detection(frame, detection)
                        nose_position = face_detector.get_key_point(detection, face_detector.FaceKeyPoint.NOSE_TIP)
                        cv2.putText(frame, f'Face {idx+1} Nose: {nose_position}', (50, 50 + idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    self.label_info.config(text=f"Face detection: {len(results.detections)} Face", font=self.custom_font)
                else:
                    self.label_info.config(text="No face found", font=self.custom_font)

                # Show results in a window
                cv2.imshow("Face Detection", frame)  

                # Save face data in the created folder.
                cv2.imwrite(os.path.join(self.folder, f"frame_{idx}.jpg"), frame)

                if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to stop.
                    self.stop_camera()
                    break

        self.camera.release()
        cv2.destroyAllWindows()

    # Camera stop function
    def stop_camera(self):
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.label_info.config(text="The camera is stopped.", font=self.custom_font)

    # Program close function
    def exit_app(self):
        if self.camera:
            self.camera.release()
        self.root.quit()

# Create a GUI
root = tk.Tk()
app = FaceApp(root)
root.mainloop()
