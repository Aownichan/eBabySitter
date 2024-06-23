import cv2
import threading
import queue

# Thread-safe queue for frames
frame_queue = queue.Queue()
# Signal to control the thread's loop
stop_signal = threading.Event()

def detect_faces_in_thread(face_cascade):
    while not stop_signal.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('Camera', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_signal.set()

def main():
    haarcascade_path = '/home/aown/Desktop/eBabySitter/server/data/haarcascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haarcascade_path)

    if face_cascade.empty():
        raise IOError("Failed to load haarcascade_frontalface_default.xml. Check the path and OpenCV installation.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    # Start the face detection thread
    face_detection_thread = threading.Thread(target=detect_faces_in_thread, args=(face_cascade,))
    face_detection_thread.daemon = True
    face_detection_thread.start()

    while not stop_signal.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            stop_signal.set()
            break

        # Put frame in queue for processing
        frame_queue.put(frame)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_detection_thread.join()

if __name__ == "__main__":
    main()