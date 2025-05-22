import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import queue
import time
import tkinter as tk
from PIL import Image, ImageTk

# TTS Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
speech_queue = queue.Queue()

def tts_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
        speech_queue.task_done()

def speak(text):
    speech_queue.put(text)

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

class SquatTrackerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Virtual Squat Tracker")
        self.attributes('-fullscreen', True)
        self.configure(bg='black')

        # Input frame for Sets & Reps
        input_frame = tk.Frame(self, bg='black')
        input_frame.pack(pady=10)

        tk.Label(input_frame, text="Sets:", font=("Helvetica", 18), fg="white", bg="black").pack(side=tk.LEFT)
        self.sets_var = tk.IntVar(value=3)
        self.sets_entry = tk.Entry(input_frame, font=("Helvetica", 18), width=3, textvariable=self.sets_var)
        self.sets_entry.pack(side=tk.LEFT, padx=10)

        tk.Label(input_frame, text="Reps per Set:", font=("Helvetica", 18), fg="white", bg="black").pack(side=tk.LEFT)
        self.reps_var = tk.IntVar(value=5)
        self.reps_entry = tk.Entry(input_frame, font=("Helvetica", 18), width=3, textvariable=self.reps_var)
        self.reps_entry.pack(side=tk.LEFT, padx=10)

        # Video frame
        self.video_label = tk.Label(self)
        self.video_label.pack(pady=10)

        # Feedback label (voice + posture feedback)
        self.feedback_label = tk.Label(self, text="", font=("Helvetica", 30), fg="yellow", bg="black")
        self.feedback_label.pack()

        # Rep & Set info label
        self.info_label = tk.Label(self, text="", font=("Helvetica", 24), fg="white", bg="black")
        self.info_label.pack(pady=10)

        # Progress bar canvas
        self.bar_canvas = tk.Canvas(self, width=600, height=50, bg='gray20', highlightthickness=0)
        self.bar_canvas.pack(pady=10)

        # Buttons
        button_frame = tk.Frame(self, bg='black')
        button_frame.pack(pady=10)

        self.calibrate_btn = tk.Button(button_frame, text="Calibrate", font=("Helvetica", 18), command=self.calibrate_mode)
        self.calibrate_btn.pack(side=tk.LEFT, padx=20)

        self.start_btn = tk.Button(button_frame, text="Start Workout", font=("Helvetica", 18), command=self.start_workout, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=20)

        self.exit_btn = tk.Button(button_frame, text="Exit", font=("Helvetica", 18), command=self.close_app)
        self.exit_btn.pack(side=tk.LEFT, padx=20)

        # Camera & State variables
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.calibrated = False
        self.squat_threshold_angle = 90
        self.rep_count = 0
        self.current_set = 1
        self.stage = "up"
        self.feedback = ""
        self.rep_timestamp = 0
        self.rep_debounce_sec = 1.2
        self.resting = False
        self.rest_duration = 45
        self.rest_time_left = self.rest_duration

        self.update_id = None
        self.info_label.config(text="Enter sets and reps, then click Calibrate.")

    def close_app(self):
        self.cap.release()
        self.destroy()
        speech_queue.put(None)
        tts_thread.join()

    def calibrate_mode(self):
        self.calibrate_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)
        speak("Calibration mode started. Get into your lowest squat position.")
        self.feedback_label.config(text="Calibration mode: Get into lowest squat position.")

        self.calibrated = False
        start_time = time.time()
        calibrated_angle = None
        countdown = 10
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                h, w = frame.shape[:2]
                landmarks = results.pose_landmarks.landmark
                hip = [int(landmarks[24].x * w), int(landmarks[24].y * h)]
                knee = [int(landmarks[26].x * w), int(landmarks[26].y * h)]
                ankle = [int(landmarks[28].x * w), int(landmarks[28].y * h)]

                calibrated_angle = calculate_angle(hip, knee, ankle)

                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, f"Calibrating in {countdown}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Angle: {int(calibrated_angle)}", tuple(knee),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "No pose detected. Hold squat.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF

            elapsed = time.time() - start_time
            if elapsed > 1 and countdown > 0:
                countdown -= 1
                start_time = time.time()

            if countdown <= 0 or key == ord('q'):
                break

        cv2.destroyWindow("Calibration")
        if calibrated_angle is None:
            calibrated_angle = 90
            speak("Calibration failed. Using default squat depth of 90 degrees.")
        else:
            speak(f"Calibration complete. Squat depth set to {int(calibrated_angle)} degrees.")

        self.squat_threshold_angle = calibrated_angle
        self.calibrated = True
        self.feedback_label.config(text=f"Calibration done! Threshold angle: {int(calibrated_angle)}")
        self.info_label.config(text="Press 'Start Workout' to begin.")
        self.start_btn.config(state=tk.NORMAL)
        self.calibrate_btn.config(state=tk.NORMAL)

    def start_workout(self):
        try:
            self.total_sets_val = int(self.sets_entry.get())
            self.reps_per_set_val = int(self.reps_entry.get())
        except ValueError:
            speak("Please enter valid numbers for sets and reps.")
            return

        if self.total_sets_val <= 0 or self.reps_per_set_val <= 0:
            speak("Sets and reps must be greater than zero.")
            return

        if not self.calibrated:
            speak("Please calibrate before starting the workout.")
            return

        self.rep_count = 0
        self.current_set = 1
        self.stage = "up"
        self.resting = False
        self.rest_time_left = self.rest_duration
        self.feedback_label.config(text="Workout started. Squat down!")
        self.info_label.config(text=f"Set {self.current_set} of {self.total_sets_val} | Reps: {self.rep_count}/{self.reps_per_set_val}")
        self.calibrate_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)
        self.run_workout_loop()

    def run_workout_loop(self):
        if self.resting:
            self.show_rest_timer()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.run_workout_loop)
            return

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        posture_feedback = ""

        if results.pose_landmarks:
            h, w = frame.shape[:2]
            landmarks = results.pose_landmarks.landmark

            required_indices = [0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30]  # head, shoulders, hips, knees, ankles
            visible = all(landmarks[i].visibility > 0.6 for i in required_indices)

            if not visible:
               self.feedback = "Full body not visible. Move back or adjust camera."
               if (time.time() - getattr(self, 'last_feedback_time', 0)) > 3:
                    speak("Please ensure your full body is visible to the camera.")
                    self.last_feedback_time = time.time()     
            else:  
               hip = [int(landmarks[24].x * w), int(landmarks[24].y * h)]
               knee = [int(landmarks[26].x * w), int(landmarks[26].y * h)]
               ankle = [int(landmarks[28].x * w), int(landmarks[28].y * h)]

            angle = calculate_angle(hip, knee, ankle)
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, f"Angle: {int(angle)}", tuple(knee),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            current_time = time.time()
            if angle < self.squat_threshold_angle and self.stage == "up" and (current_time - self.rep_timestamp) > self.rep_debounce_sec:
                self.stage = "down"
                self.rep_timestamp = current_time
            elif angle > self.squat_threshold_angle and self.stage == "down" and (current_time - self.rep_timestamp) > self.rep_debounce_sec:
                self.stage = "up"
                self.rep_count += 1
                self.rep_timestamp = current_time
                

            # Posture feedback like a trainer
            angle_diff = angle - self.squat_threshold_angle
            if angle_diff > 25:
                posture_feedback = "Try to go a little deeper."
            elif angle_diff < -20:
                posture_feedback = "Careful! You're going too low."
            else:
                posture_feedback = "Nice form, keep going!"

            if posture_feedback != self.feedback and (current_time - getattr(self, 'last_feedback_time', 0)) > 3:
                speak(posture_feedback)
                self.last_feedback_time = current_time
            self.feedback = posture_feedback

            # Set & rep tracking
            if self.rep_count >= self.reps_per_set_val:
                if self.current_set >= self.total_sets_val:
                    self.end_workout()
                    speak("Workout Complete! Good job!")
                    self.feedback_label.config(text="Workout Complete! Good job!")
                    return
                else:
                    self.resting = True
                    self.rest_time_left = self.rest_duration
                    speak(f"Set {self.current_set} complete. Take a break.")
                    self.current_set += 1
                    self.rep_count = 0

        else:
            self.feedback = "No pose detected. Make sure you're in view."
            if (time.time() - getattr(self, 'last_feedback_time', 0)) > 5:
                speak("I can't see you. Please stay in the camera frame.")
                self.last_feedback_time = time.time()

        # Convert frame to ImageTk
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.feedback_label.config(text=self.feedback)
        self.info_label.config(text=f"Set {self.current_set} of {self.total_sets_val} | Reps: {self.rep_count}/{self.reps_per_set_val}")
        self.draw_progress_bar()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close_app()
            return

        self.update_id = self.after(10, self.run_workout_loop)

    def draw_progress_bar(self):
        self.bar_canvas.delete("all")
        bar_width = 600
        bar_height = 40
        reps_ratio = self.rep_count / self.reps_per_set_val if self.reps_per_set_val else 0
        fill_width = int(bar_width * reps_ratio)
        self.bar_canvas.create_rectangle(0, 0, bar_width, bar_height, fill="gray30", outline="")
        self.bar_canvas.create_rectangle(0, 0, fill_width, bar_height, fill="limegreen", outline="")
        self.bar_canvas.create_text(bar_width // 2, bar_height // 2,
                                   text=f"Set {self.current_set}/{self.total_sets_val} - Reps: {self.rep_count}/{self.reps_per_set_val}",
                                   fill="white", font=("Helvetica", 16))

    def show_rest_timer(self):
        if self.rest_time_left > 0:
            self.feedback_label.config(text=f"Resting... Next set starts in {self.rest_time_left} seconds")
            if self.rest_time_left == 5:
                speak("Get ready for next set.")
            self.rest_time_left -= 1
            self.update_id = self.after(1000, self.show_rest_timer)
        else:
            self.resting = False
            self.feedback_label.config(text="Start your squat!")
            self.info_label.config(text=f"Set {self.current_set} of {self.total_sets_val} | Reps: {self.rep_count}/{self.reps_per_set_val}")
            self.run_workout_loop()

    def end_workout(self):
        self.feedback_label.config(text="Workout complete! Well done!")
        self.info_label.config(text="")
        self.calibrate_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.DISABLED)

    def on_closing(self):
        if self.update_id:
            self.after_cancel(self.update_id)
        self.cap.release()
        self.destroy()
        speech_queue.put(None)
        tts_thread.join()

if __name__ == "__main__":
    app = SquatTrackerApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


