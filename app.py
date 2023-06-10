"""Streamlit App"""

import cv2

# Initialize webcam video capture
cap = cv2.VideoCapture(0)

# Set camera to capture at 1080p 30fps
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

if __name__ == "__main__":
    while True:
        # Read webcam footage frame by frame and display it
        ret, frame = cap.read()
        cv2.imshow('Webcam Footage', frame)

        # Exit on pressing 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
