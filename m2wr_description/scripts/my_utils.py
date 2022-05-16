import cv2 as cv
import pyrealsense2 as rs
import realsense_depth

def realsence_show(img, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    centure_x = abs(p1[0] + p2[0]) // 2
    centure_y = abs(p1[1] + p2[1]) // 2
    centure = (centure_x, centure_y)
    cv.rectangle(img, p1, p2, color, thickness=lw, lineType=cv.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv.rectangle(img, p1, p2, color, -1, cv.LINE_AA)  # filled
        cv.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv.LINE_AA)
        cv.circle(img, centure, 4, (0, 255, 0), thickness=-1)

    return centure
