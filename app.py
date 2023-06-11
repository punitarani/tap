"""Streamlit App"""

import math
import time

import cv2
import mediapipe as mp


def get_finger_tip_coordinates(hand_landmarks):
    """Returns the coordinates of the thumb and index fingertips"""
    thumb_tip = [hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y]
    index_tip = [hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y]
    return {
        "thumb": thumb_tip,
        "index": index_tip
    }


def get_finger_distance(finger_tip_coordinates):
    """Returns the distance between the thumb and index fingertips"""
    # Calculate the Euclidean distance between the thumb and index fingertips
    return math.sqrt(
        (finger_tip_coordinates["thumb"][0] - finger_tip_coordinates["index"][0]) ** 2 +
        (finger_tip_coordinates["thumb"][1] - finger_tip_coordinates["index"][1]) ** 2
    )


if __name__ == "__main__":
    # Initialize webcam video capture
    cap = cv2.VideoCapture(0)

    # Set camera to capture at 1080p 30fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize mediapipe hand detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    while True:
        # Get the latest frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to RGB (MediaPipe uses RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(frame_rgb)

        # Draw the hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate the distance between the thumb and index fingertips
                distance = get_finger_distance(get_finger_tip_coordinates(hand_landmarks))

                # Check if the distance is below a threshold to consider the fingers touching
                if distance < 0.03:
                    print(f"{time.time()} Index and thumb fingers are touching")

        # Display the frame
        cv2.imshow("Hand Tracking", frame)

        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
