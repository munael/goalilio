# In[ ]:
import cv2 as cv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numba


video_path = '/home/daw/Desktop/goalilio/goals2.bin.mp4'

frames = []
hsv_vid = []

# In[ ]:
cap = cv.VideoCapture(video_path)

bgr_vid = threshed = None

print('we\'re here!')

def bgr2hsv(video):
    print(video.shape)

    # video = video / 255

    V = np.maximum.reduce(video, axis=-1)

    mn = np.minimum.reduce(video, axis=-1)
    
    S = ((V - mn) / (V)) * (V != 0)
    
    Vi = np.argmax(video, axis=-1)
    x = (V - mn)

    Hr = 60*(video[..., 1] - video[..., 0]) / x
    Hg = 120 + 60*(video[..., 0] - video[..., 2]) / x
    Hb = 240 + 60*(video[..., 2] - video[..., 1]) / x

    H = np.array([Hb, Hg, Hr]).transpose(1, 2, 3, 0)

    # Vi = Vi.reshape((Vi.shape[0], Vi.shape[1], Vi.shape[2], 1))

    
    # print(Vi.max(), Vi.min())
    print(Vi.shape, H.shape)

    sh = H.shape
    H = H.reshape(-1, 3)
    Vi = Vi.reshape(-1, 1)
    H = H[Vi]

    # h = input()

    H = Vi.choose(H)

    rv = np.array([H, S, V]).transpose(3, 0, 1, 2)

    return rv

@numba.jit
def cvtColor(video, last, dst, *args, **kwds):
    if kwds.has_key('dst'):
        rv = kwds['dst']
    else:
        rv = np.empty(video.shape[:-1] + (last,))
    
    for i, frame in enumerate(video):
        cv.cvtColor(frame, dst=rv[i], *args, **(kwds))
    
    return rv

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
axs[0].set_title('Original')
axs[1].set_title('segmented')

plt.ion()

def main():
    global frames, cap, bgr_vid, hsv_vid, threshed
    # In[ ]:
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        frame = frame[100:750, 200:850]
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frames += [frame]
        hsv_vid += [hsv_frame]
        axs[0].imshow(frame)
        axs[1].imshow(hsv_frame)
        plt.pause(0.05)


    # print(len(frames))
    cap.release()

    # In[ ]:
    bgr_vid = np.array(frames[:-2])
    hsv_vid = np.array(hsv_vid[:-2])


    # In[ ]:
    print('Got HSV')

    # In[ ]:
    th_s = hsv_vid[..., 1] <= 15
    th_v = hsv_vid[..., 2] >= 240
    threshed = th_s & th_v
    threshed = threshed * 255

main()