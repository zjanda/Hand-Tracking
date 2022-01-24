import cv2
import mediapipe as mp
from time import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        handLandMarks = results.multi_hand_landmarks
        hand_present = handLandMarks
        if hand_present:
            for handLandMarks in handLandMarks:
                if draw:
                    for id, landmark in enumerate(handLandMarks.landmark):
                        h, w, c = img.shape
                        centx, centy = int(landmark.x * w), int(landmark.y * h)
                        print(id, centx, centy)
                        if id % 4 == 0 and id != 0:
                            cv2.circle(img, (centx, centy), 25, (255, 255, cv2.FILLED))
                    self.mpDraw.draw_landmarks(img, handLandMarks,
                                               self.mpHands.HAND_CONNECTIONS)  # draws 1 hand at a time

        return img


def main():
    FPS_UPDATE_INTERVAL = 1  # seconds
    prevTime = 0
    last_update_time = time()
    fps = 0

    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        elapsed_time = round(time() - last_update_time, 1)

        currTime = time()
        if elapsed_time >= FPS_UPDATE_INTERVAL:
            last_update_time = time()
            fps = 1 / (currTime - prevTime)
        prevTime = currTime
        # image, text, pos, font, thickness, color,
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
