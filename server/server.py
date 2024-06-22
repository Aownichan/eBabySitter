import cv2

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    return frame

def main():
    # Load the pre-trained Haar cascades classifier for face detection
    # Path to the manually downloaded Haarcascade file
    haarcascade_path = '/home/aown/Desktop/eBabySitter/server/data/haarcascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haarcascade_path)

    if face_cascade.empty():
        raise IOError("Failed to load haarcascade_frontalface_default.xml. Check the path and OpenCV installation.")

    # Open the default camera (usually the built-in webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    # Define the IP address and port for the localhost connection
    ip_address = "127.0.0.1"
    port = 8888

    # Create a VideoCapture object for localhost connection
    cap_send = cv2.VideoCapture(f"udp://{ip_address}:{port}")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Detect faces and draw bounding boxes
        frame = detect_faces(frame, face_cascade)

        # Display the resulting frame
        cv2.imshow('Camera', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
