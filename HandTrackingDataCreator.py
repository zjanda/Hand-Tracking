from time import time

import mediapipe as mp

import helpers
from helpers import *

write = False
TIME_PER_HAND = 20


def Draw():
    global last_update_time
    global prevTime
    global start
    global cur_time
    global seconds_passed
    global start_up
    global fps
    global num_fingers
    h, w, c = img.shape

    if start_up == 0:
        # FPS
        elapsed_time = round(time() - last_update_time, 1)
        currTime = time()
        if elapsed_time >= FPS_UPDATE_INTERVAL:
            last_update_time = time()
            fps = 1 / (currTime - prevTime)
        prevTime = currTime

        ########################################################################################################
        # Put text to image
        ########################################################################################################
        fontSize = 3

        # Show text for indicating whether writing to disk or not
        write_string = 'w:1' if write else 'w:0'
        position = (int(w - 160), int(h - 10))
        cv2.putText(img, str(write_string), position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 255), 3)

        # Finger timer
        timer = int(cur_time - start)
        if timer >= 1:
            start = time()
            seconds_passed += 1
        cur_time = time()
        num_fingers = seconds_passed // TIME_PER_HAND

        # Show text for number of fingers
        position = (int(w * .5), 24 * fontSize)
        cv2.putText(img, str(num_fingers), position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 255), 3)

        # Show text of timer for each set of fingers
        tph = str(seconds_passed % TIME_PER_HAND)  # time per hand
        position = (w - 20 * fontSize - (len(tph) - 1) * 20 * fontSize, 24 * fontSize)
        cv2.putText(img, tph, position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 255), 3)

        # Show text for FPS
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
    else:
        # wait to display and begin time calculations. This is done to reduce data imbalance (0 was much less)
        start_up -= 1
        fps = 0

    # Draw threshold window for hand positioning.
    # Purpose: model will need unreasonably more data if hand position is not restricted.
    DrawRegion(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


if write:
    PromptOverwrite()
    write = helpers.write

setWriteFalse()  # to not overwrite data on accident

if write:
    with open('data.txt', 'w'):
        pass

FPS_UPDATE_INTERVAL = 1  # seconds

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
fingers = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five'}
# FPS vars
# TODO: Make FPS class
prevTime = 0
currTime = time()
last_update_time = time()
fps = 0

# Finger timer vars
# TODO: Make finger timer class
start = time()
cur_time = time()
seconds_passed = 1
time_elapsed = int(cur_time - start)
num_fingers = seconds_passed // TIME_PER_HAND
start_up = 100

with open('data.txt', 'a') as file:
    while num_fingers <= 5:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        handLandMarks = results.multi_hand_landmarks
        hand_present = handLandMarks

        if hand_present and start_up == 0:
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
                mpDraw.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)  # draws 1 hand at a time

                np_list = np.array(hand).T
                if write: np.savetxt(file, np_list, newline='\n')

        else:
            if start_up != 0:
                start = time()
                cur_time = time()
                seconds_passed = 1
                time_elapsed = int(cur_time - start)
                # num_fingers = seconds_passed // TIME_PER_HAND
        Draw()
