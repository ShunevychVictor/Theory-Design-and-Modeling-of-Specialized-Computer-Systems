import os
import sys
import cv2
import time
import csv
import threading
import psutil
import numpy as np
import imutils
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

CSV_FILE = "activity_log.csv"

# Створення файлу журналу, якщо він відсутній
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Start Time", "End Time", "Duration (s)"])

# Спільні ресурси та синхронізація
frame_to_show = np.zeros((480, 640, 3), dtype=np.uint8)
lock = threading.Lock()

# Змінна чутливості (поріг площі контуру)
min_contour_area = 10000


def get_camera_capture():
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"✅ Камеру підключено під індексом: {idx}")
                return cap
            cap.release()
    print("❌ Не вдалося відкрити камеру. Перевірте дозволи в системі.")
    return None


def capture_and_detect_motion():
    global frame_to_show, min_contour_area

    video = get_camera_capture()
    if video is None:
        return

    time.sleep(1.0)
    static_back = None
    motion_start_time = None

    while True:
        check, frame = video.read()
        if not check or frame is None:
            time.sleep(0.02)
            continue

        frame = imutils.resize(frame, width=640)
        current_time = datetime.now()
        timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # Попередня обробка зображення
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if static_back is None:
            static_back = gray.copy().astype("float")
            continue

        # Побудова фонової моделі та визначення різниці
        cv2.accumulateWeighted(gray, static_back, 0.05)
        diff_frame = cv2.absdiff(gray, cv2.convertScaleAbs(static_back))
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Отримання актуального значення чутливості під блокуванням
        with lock:
            threshold_area = min_contour_area

        motion = False
        for contour in cnts:
            if cv2.contourArea(contour) < threshold_area:
                continue
            motion = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Фіксація подій у CSV-журнал
        if motion:
            if motion_start_time is None:
                motion_start_time = current_time
        else:
            if motion_start_time is not None:
                motion_end_time = current_time
                duration = round((motion_end_time - motion_start_time).total_seconds(), 2)
                with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        motion_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        motion_end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        duration
                    ])
                motion_start_time = None

        # Візуалізація інформації
        cv2.putText(frame, f"Motion: {'YES' if motion else 'NO'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if motion else (0, 255, 0), 2)
        cv2.putText(frame, timestamp_str, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {threshold_area} px", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        with lock:
            frame_to_show = frame.copy()

        time.sleep(0.03)

    video.release()


def generate():
    global frame_to_show
    while True:
        with lock:
            frame = frame_to_show.copy()
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# API: Оновлення порогу чутливості від повзунка
@app.route('/set_sensitivity', methods=['POST'])
def set_sensitivity():
    global min_contour_area
    data = request.get_json()
    if data and "threshold" in data:
        with lock:
            min_contour_area = int(data["threshold"])
        return jsonify({"status": "success", "threshold": min_contour_area})
    return jsonify({"status": "error"}), 400


# API: Відправка системних метрик CPU/RAM
@app.route('/system_metrics')
def system_metrics():
    return jsonify({
        "time": datetime.now().strftime("%H:%M:%S"),
        "cpu": psutil.cpu_percent(interval=None),
        "ram": psutil.virtual_memory().percent
    })


if __name__ == '__main__':
    psutil.cpu_percent(interval=None)
    threading.Thread(target=capture_and_detect_motion, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)