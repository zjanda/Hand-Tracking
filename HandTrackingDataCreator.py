import cv2
import mediapipe as mp
from time import time
import numpy as np

write = False
# write = True
TIME_PER_HAND = 10

if write:
    with open('data.txt', 'w') as file:
        pass

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

# Finger timer vars
start = time()
cur_time = time()
multiplier = 1
time_elapsed = int(cur_time - start)
num_fingers = multiplier // TIME_PER_HAND
with open('data.txt', 'a') as file:
    while num_fingers <= 5:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        handLandMarks = results.multi_hand_landmarks
        hand_present = handLandMarks
        h, w, c = img.shape
        top = .45
        bot = 1 - top
        TH_TOPLEFT = (int(top * w), int(top * h))
        TH_BOTRIGHT = (int(bot * w), int(bot * h))


        def in_threshold(centx, centy):
            return TH_TOPLEFT[0] < centx < TH_BOTRIGHT[0] and TH_TOPLEFT[1] < centy < TH_BOTRIGHT[1]


        if hand_present:
            for handLandMarks in handLandMarks:  # hand
                centx_list = []
                centy_list = []
                id_list = []
                newline_list = []
                hand = (id_list, centx_list, centy_list, [num_fingers] * 21)
                for id, landmark in enumerate(handLandMarks.landmark):  # point on hand
                    centx, centy = int(landmark.x * w), int(landmark.y * h)

                    centx_list.append(centx)
                    centy_list.append(centy)
                    id_list.append(id)

                    if id % 4 == 0 and id != 0:
                        cv2.circle(img, (centx, centy), 25, (255, 255, 0))
                        # if id == 8 and in_threshold(centx, centy):
                        #     cv2.putText(img, "DETECTED", (int(.4 * w), 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                mpDraw.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)  # draws 1 hand at a time

                np_list = np.array(hand).T
                if write: np.savetxt(file, np_list, newline='\n')

        elapsed_time = round(time() - last_update_time, 1)

        currTime = time()
        if elapsed_time >= FPS_UPDATE_INTERVAL:
            last_update_time = time()
            fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # Finger timer

        timer = int(cur_time - start)
        if timer >= 1:
            time_elapsed = timer + multiplier
            start = time()
            multiplier += 1
        cur_time = time()
        num_fingers = multiplier // TIME_PER_HAND

        write_string = 'w:1' if write else 'w:0'

        cv2.putText(img, str(write_string), (int(w - 160), int(h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.putText(img, str(num_fingers), (int(w * .5), 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.putText(img, str(time_elapsed % TIME_PER_HAND), (w - 60, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        # Draw threshold
        # cv2.rectangle(img, TH_TOPLEFT, TH_BOTRIGHT, (0, 0, 0), 10)
        # image, text, pos, font, font size, color, thickness
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
