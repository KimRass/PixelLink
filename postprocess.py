import cv2


_, out = cv2.connectedComponents(image=temp.astype("uint8"), connectivity=4)
show_image(out)