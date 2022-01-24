import cv2
import mediapipe as mp
from time import time
import numpy as np
from Header import load_model

TIME_PER_HAND = 5

FPS_UPDATE_INTERVAL = 1  # seconds

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
fingers = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five'}
prevTime = 0
currTime = time()
last_update_time = time()
fps = 0

# load model
model = load_model('finalized_model.sav')

# Finger timer vars
with open('data.txt', 'a') as file:
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        handLandMarks = results.multi_hand_landmarks
        hand_present = handLandMarks
        h, w, c = img.shape

        if hand_present:
            hand_coords = []
            for handLandMarks in handLandMarks:  # hand
                for id, landmark in enumerate(handLandMarks.landmark):  # point on hand
                    centx, centy = int(landmark.x * w), int(landmark.y * h)
                    hand_coords.append([id, centx, centy])
                    # if id % 4 == 0 and id != 0:
                    #     cv2.circle(img, (centx, centy), 25, (255, 255, 0))
                mpDraw.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)  # draws 1 hand at a time
            hand_coords = np.array(hand_coords)
            print(int(np.average(model.predict(hand_coords))))
        elapsed_time = round(time() - last_update_time, 1)

        currTime = time()
        if elapsed_time >= FPS_UPDATE_INTERVAL:
            last_update_time = time()
            fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # cv2.putText(img, str(write_string), (int(w - 160), int(h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        # cv2.putText(img, str(num_fingers), (int(w * .5), 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        # cv2.putText(img, str(time_elapsed % time_per_hand), (w - 60, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        # image, text, pos, font, font size, color, thickness
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
