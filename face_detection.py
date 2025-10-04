
# Task 5: Face Detection and Recognition GUI
# Enhanced GUI version with tkinter interface

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import threading

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

class FaceDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎯 Face Detection & Recognition - Task 5")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')

        # Initialize face detection system
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        # Face recognition data
        self.known_faces_encodings = []
        self.known_faces_names = []
        self.face_database_path = "face_database/"

        if not os.path.exists(self.face_database_path):
            os.makedirs(self.face_database_path)

        # GUI variables
        self.current_image = None
        self.current_image_path = None
        self.webcam_active = False
        self.cap = None

        self.create_widgets()

        # Load existing face database
        if FACE_RECOGNITION_AVAILABLE:
            self.load_face_database()

    def create_widgets(self):
        """Create GUI widgets"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(fill='x', pady=10)

        title_label = tk.Label(
            title_frame,
            text="🎯 Face Detection & Recognition System",
            font=("Arial", 20, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title_label.pack()

        subtitle_label = tk.Label(
            title_frame,
            text="Task 5 - Codsoft AI Internship",
            font=("Arial", 12),
            bg='#2c3e50',
            fg='#bdc3c7'
        )
        subtitle_label.pack()

        # Main content frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='#34495e', width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)

        # Image controls
        img_frame = tk.LabelFrame(
            left_panel,
            text="📁 Image Processing",
            font=("Arial", 12, "bold"),
            bg='#34495e',
            fg='#ecf0f1',
            pady=10
        )
        img_frame.pack(fill='x', padx=10, pady=10)

        tk.Button(
            img_frame,
            text="📂 Load Image",
            font=("Arial", 10),
            bg='#3498db',
            fg='white',
            command=self.load_image,
            width=20
        ).pack(pady=5)

        tk.Button(
            img_frame,
            text="🔍 Detect Faces",
            font=("Arial", 10),
            bg='#e67e22',
            fg='white',
            command=self.detect_faces_in_image,
            width=20
        ).pack(pady=5)

        tk.Button(
            img_frame,
            text="💾 Save Result",
            font=("Arial", 10),
            bg='#27ae60',
            fg='white',
            command=self.save_result,
            width=20
        ).pack(pady=5)

        # Webcam controls
        webcam_frame = tk.LabelFrame(
            left_panel,
            text="🎥 Live Detection",
            font=("Arial", 12, "bold"),
            bg='#34495e',
            fg='#ecf0f1',
            pady=10
        )
        webcam_frame.pack(fill='x', padx=10, pady=10)

        self.webcam_btn = tk.Button(
            webcam_frame,
            text="▶️ Start Webcam",
            font=("Arial", 10),
            bg='#e74c3c',
            fg='white',
            command=self.toggle_webcam,
            width=20
        )
        self.webcam_btn.pack(pady=5)

        tk.Button(
            webcam_frame,
            text="📸 Capture Frame",
            font=("Arial", 10),
            bg='#9b59b6',
            fg='white',
            command=self.capture_frame,
            width=20
        ).pack(pady=5)

        # Face database controls
        if FACE_RECOGNITION_AVAILABLE:
            db_frame = tk.LabelFrame(
                left_panel,
                text="🗃️ Face Database",
                font=("Arial", 12, "bold"),
                bg='#34495e',
                fg='#ecf0f1',
                pady=10
            )
            db_frame.pack(fill='x', padx=10, pady=10)

            tk.Button(
                db_frame,
                text="➕ Add Face",
                font=("Arial", 10),
                bg='#1abc9c',
                fg='white',
                command=self.add_face_to_database,
                width=20
            ).pack(pady=5)

            tk.Button(
                db_frame,
                text="📋 View Database",
                font=("Arial", 10),
                bg='#f39c12',
                fg='white',
                command=self.show_database,
                width=20
            ).pack(pady=5)

            tk.Button(
                db_frame,
                text="🗑️ Clear Database",
                font=("Arial", 10),
                bg='#e74c3c',
                fg='white',
                command=self.clear_database,
                width=20
            ).pack(pady=5)

        # Settings frame
        settings_frame = tk.LabelFrame(
            left_panel,
            text="⚙️ Settings",
            font=("Arial", 12, "bold"),
            bg='#34495e',
            fg='#ecf0f1',
            pady=10
        )
        settings_frame.pack(fill='x', padx=10, pady=10)

        # Detection options
        self.detect_eyes_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            settings_frame,
            text="Detect Eyes",
            variable=self.detect_eyes_var,
            bg='#34495e',
            fg='#ecf0f1',
            selectcolor='#2c3e50'
        ).pack(anchor='w')

        self.detect_smiles_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            settings_frame,
            text="Detect Smiles",
            variable=self.detect_smiles_var,
            bg='#34495e',
            fg='#ecf0f1',
            selectcolor='#2c3e50'
        ).pack(anchor='w')

        # Right panel - Display
        right_panel = tk.Frame(main_frame, bg='#34495e')
        right_panel.pack(side='right', fill='both', expand=True)

        # Image display
        self.image_label = tk.Label(
            right_panel,
            text="🖼️ No Image Loaded\n\nClick 'Load Image' or 'Start Webcam'",
            font=("Arial", 14),
            bg='#2c3e50',
            fg='#bdc3c7',
            width=50,
            height=20
        )
        self.image_label.pack(fill='both', expand=True, padx=10, pady=10)

        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame,
            text="Ready - Face Detection System Initialized",
            font=("Arial", 10),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.status_label.pack(fill='x', padx=10, pady=5)

    def update_status(self, message):
        """Update status bar"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.config(text=f"[{timestamp}] {message}")

    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                # Load with OpenCV
                self.current_image = cv2.imread(file_path)
                self.current_image_path = file_path

                # Display image
                self.display_image(self.current_image)
                self.update_status(f"Image loaded: {os.path.basename(file_path)}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
                self.update_status("Failed to load image")

    def detect_faces_in_image(self):
        """Detect faces in the current image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return

        try:
            result_image = self.current_image.copy()
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

            # Draw face rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(result_image, 'Face', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Detect eyes if enabled
                if self.detect_eyes_var.get():
                    roi_gray = gray[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(result_image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

            # Face recognition if available
            recognized_faces = []
            if FACE_RECOGNITION_AVAILABLE and len(self.known_faces_encodings) > 0:
                recognized_faces = self.recognize_faces_in_image(self.current_image)

                for result in recognized_faces:
                    name = result['name']
                    confidence = result['confidence']
                    x, y, w, h = result['location']

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 3)

                    label = f"{name} ({confidence:.2f})" if confidence > 0 else name
                    cv2.putText(result_image, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Display result
            self.display_image(result_image)

            # Update status
            message = f"Detected {len(faces)} faces"
            if recognized_faces:
                recognized_names = [r['name'] for r in recognized_faces if r['name'] != 'Unknown']
                if recognized_names:
                    message += f", recognized: {', '.join(recognized_names)}"

            self.update_status(message)

        except Exception as e:
            messagebox.showerror("Error", f"Face detection failed: {e}")
            self.update_status("Face detection failed")

    def recognize_faces_in_image(self, image):
        """Recognize faces in image (helper function)"""
        if not FACE_RECOGNITION_AVAILABLE:
            return []

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            results = []
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding)
                name = "Unknown"
                confidence = 0

                if True in matches:
                    face_distances = face_recognition.face_distance(self.known_faces_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_faces_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]

                top, right, bottom, left = face_location
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (left, top, right-left, bottom-top)
                })

            return results
        except:
            return []

    def display_image(self, cv_image):
        """Display OpenCV image in tkinter label"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Resize image to fit display
            height, width = rgb_image.shape[:2]
            max_height, max_width = 400, 600

            if height > max_height or width > max_width:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                rgb_image = cv2.resize(rgb_image, (new_width, new_height))

            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)

            # Update label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference

        except Exception as e:
            print(f"Display error: {e}")

    def toggle_webcam(self):
        """Start/stop webcam"""
        if not self.webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()

    def start_webcam(self):
        """Start webcam detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return

            self.webcam_active = True
            self.webcam_btn.config(text="⏹️ Stop Webcam", bg='#e74c3c')
            self.update_status("Webcam started")

            # Start webcam loop
            self.webcam_loop()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {e}")

    def stop_webcam(self):
        """Stop webcam detection"""
        self.webcam_active = False
        if self.cap:
            self.cap.release()
        self.webcam_btn.config(text="▶️ Start Webcam", bg='#27ae60')
        self.update_status("Webcam stopped")

    def webcam_loop(self):
        """Webcam processing loop"""
        if self.webcam_active and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Process frame
                processed_frame = self.process_webcam_frame(frame)
                self.current_image = processed_frame.copy()

                # Display frame
                self.display_image(processed_frame)

            # Schedule next frame
            self.root.after(30, self.webcam_loop)

    def process_webcam_frame(self, frame):
        """Process webcam frame for face detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

            # Draw face rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Add statistics
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            return frame
        except:
            return frame

    def capture_frame(self):
        """Capture current webcam frame"""
        if self.current_image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_frame_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_image)
            messagebox.showinfo("Saved", f"Frame saved as {filename}")
            self.update_status(f"Frame captured: {filename}")
        else:
            messagebox.showwarning("No Frame", "No frame to capture")

    def save_result(self):
        """Save current processed image"""
        if self.current_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
            )
            if file_path:
                cv2.imwrite(file_path, self.current_image)
                messagebox.showinfo("Saved", f"Image saved as {os.path.basename(file_path)}")
                self.update_status(f"Image saved: {os.path.basename(file_path)}")
        else:
            messagebox.showwarning("No Image", "No image to save")

    def load_face_database(self):
        """Load face database"""
        encodings_file = os.path.join(self.face_database_path, "face_encodings.pkl")
        if os.path.exists(encodings_file):
            try:
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces_encodings = data.get('encodings', [])
                    self.known_faces_names = data.get('names', [])
                    self.update_status(f"Loaded {len(self.known_faces_names)} known faces")
            except:
                pass

    def add_face_to_database(self):
        """Add face to database"""
        if not FACE_RECOGNITION_AVAILABLE:
            messagebox.showwarning("Not Available", "Face recognition library not installed")
            return

        # Get image file
        file_path = filedialog.askopenfilename(
            title="Select face image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        # Get name
        name = tk.simpledialog.askstring("Name", "Enter person's name:")
        if not name:
            return

        try:
            # Load and encode face
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                self.known_faces_encodings.append(encodings[0])
                self.known_faces_names.append(name)

                # Save database
                self.save_face_database()

                messagebox.showinfo("Success", f"Added {name} to database")
                self.update_status(f"Added {name} to database")
            else:
                messagebox.showwarning("No Face", "No face detected in image")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to add face: {e}")

    def save_face_database(self):
        """Save face database"""
        encodings_file = os.path.join(self.face_database_path, "face_encodings.pkl")
        data = {
            'encodings': self.known_faces_encodings,
            'names': self.known_faces_names
        }
        with open(encodings_file, 'wb') as f:
            pickle.dump(data, f)

    def show_database(self):
        """Show face database info"""
        if len(self.known_faces_names) == 0:
            messagebox.showinfo("Database", "Face database is empty")
        else:
            names = "\n".join([f"{i+1}. {name}" for i, name in enumerate(self.known_faces_names)])
            messagebox.showinfo("Database", f"Known faces ({len(self.known_faces_names)}):\n\n{names}")

    def clear_database(self):
        """Clear face database"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the face database?"):
            self.known_faces_encodings = []
            self.known_faces_names = []
            self.save_face_database()
            messagebox.showinfo("Cleared", "Face database cleared")
            self.update_status("Face database cleared")

    def run(self):
        """Start the GUI"""
        try:
            self.root.mainloop()
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

# Import dialog for name input
import tkinter.simpledialog

if __name__ == "__main__":
    app = FaceDetectionGUI()
    app.run()
