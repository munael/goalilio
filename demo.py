import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def check_motion(p1, p2):
    # return sp.spatial.distance.cdist(p1, p2) > 2
    return (p1 != p2).any()

def threshold(frame):
    x = frame[..., 1] <= 25
    y = frame[..., 2] >= 240

    return (x & y).astype(np.uint8) * 255

def pre(frame):
    frame = cv.GaussianBlur(frame, ksize=(3, 3), sigmaX=10)
    frame = cv.erode(frame, np.ones((3, 3)))
    frame = cv.dilate(frame, np.ones((5, 5)))

    return frame


video_path = 'project_cv_2.mov'

cap = cv.VideoCapture(video_path)

video = []
video_hsv = []
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))

zs = np.empty((int(cap.get(4)), int(cap.get(3))))
counter = 0
hist = []
while True:
    ret, frame = cap.read()
    counter += 1
    if not ret:
        break

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    thsh = threshold(hsv_frame)

    thsh = pre(thsh)

    circles = cv.HoughCircles(thsh[350:700, 250:750],
                              cv.cv.CV_HOUGH_GRADIENT, 1,
                              param1=5, param2=10,
                              minDist=200, minRadius=13, maxRadius=20)

    thsh = np.array([thsh, zs, zs]).transpose(1, 2, 0).copy()

    if circles is not None:
        circles = circles[0]
        circles[:, 0] += 250
        circles[:, 1] += 350
        hist += [circles[0][:2]]
        for c in circles[:1]:
            cv.circle(thsh, tuple(c[:2]), c[-1], (0, 0, 255), 3)

        # subs = cv.BackgroundSubtractorMOG2()
    if len(hist) > 1:
        if (check_motion(hist[-1], hist[-2])):
            # x1, y1 = hist[-3]
            # x2, y2 = hist[-1]
            p1 = hist[-2]
            p2 = hist[-1]
            p1 = p2 + (p2 - p1)*4
            cv.line(thsh, tuple(p2), tuple(p1), (0, 255, 0), 2)
            cv.line(frame, tuple(p2), tuple(p1), (150, 0, 240), 2)


    video += [frame]
    video_hsv += [hsv_frame]

    cv.imshow('Source', frame)
    cv.imshow('Thresholded', thsh)

    if cv.waitKey(1) == 27:
        break

cap.release()

cv.destroyAllWindows()

video     = np.array(video    [:-2])

video_hsv = np.array(video_hsv[:-2])

print(hsv_frame.shape)
