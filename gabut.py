import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

show_text = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (480, 360))


    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    show_text = ""

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            thumb_tip = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_mcp = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_finger_tip = handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = handLms.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = handLms.landmark[mp_hands.HandLandmark.PINKY_TIP]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            distance_thumb_index = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

            if distance_thumb_index > 40 and handLms.landmark[mp_hands.HandLandmark.WRIST].y > 0.5:
                show_text = "Aku Aku"

            elif (distance_thumb_index < 30 and
                  all(handLms.landmark[finger].y > index_finger_mcp.y for finger in 
                  [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP])):
                show_text = "Cinta Cinta"

            elif (index_finger_tip.y < index_finger_mcp.y and
                  all(handLms.landmark[finger].y > index_finger_mcp.y for finger in
                  [mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, 
                   mp_hands.HandLandmark.PINKY_TIP])):
                show_text = "Kamu Kamu"

    if show_text:
        cv2.putText(frame, show_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
