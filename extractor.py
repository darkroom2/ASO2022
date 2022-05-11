import cv2 as cv
from matplotlib import pyplot as plt


def main():
    # data_dirs = [
    #     'output/view_1',
    #     'output/view_2'
    # ]

    # TODO: Process all the models views and save its 'features' somewhere for later use in neural network (?) or
    #  other classifier.

    # Test script, not ready for processing

    img1 = cv.imread('output/view_1/chair/train/chair_0892.png', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('output/view_1/sofa/train/sofa_0013.png', cv.IMREAD_GRAYSCALE)

    # Initiate SIFT detector
    sift = cv.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3)
    plt.show()


if __name__ == '__main__':
    main()
