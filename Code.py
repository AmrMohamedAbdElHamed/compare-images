import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
"""(640, 480)
(1024, 683)
(1110, 800)
(968, 1600)
(1052, 800)
(1035, 690)
(533, 800)"""
sift = cv2.SIFT_create()
image_path_jpeg_A = glob.glob('assignment data/*a.jpeg', )
image_path_jpeg_B = glob.glob('assignment data/*b.jpeg')
image_path_png_C = glob.glob('assignment data/*c.png')
bf = cv2.BFMatcher()
images = []
image_a = np.zeros((2, 2))
image_list_a = []
image_b = np.zeros((2, 2))
image_list_b = []
image_c = np.zeros((2, 2))
image_list_c = []
size = (640, 480)

for img in image_path_jpeg_A:
    image_a = cv2.imread(img, 0)
    resized_image = cv2.resize(image_a, size)
    image_list_a.append(resized_image)

for img in image_path_jpeg_B:
    image_b = cv2.imread(img, 0)
    resized_image = cv2.resize(image_b, size)
    image_list_b.append(resized_image)

for img in image_path_png_C:
    image_c = cv2.imread(img, 0)
    resized_image = cv2.resize(image_c, size)
    image_list_c.append(resized_image)

for i in range(len(image_path_jpeg_A)):
    ratio_matches = []
    ratio_matches_a_b = []
    matches_threshold  = []
    if i == 55:
        kp_a, des_a = sift.detectAndCompute(image_list_a[i], None)
        kp_b, des_b = sift.detectAndCompute(image_list_b[i], None)
        kp_c, des_c = sift.detectAndCompute(image_list_c[0], None)
        matches_a_b = bf.match(des_a, des_b)
        matches_a_c = bf.match(des_a, des_c)
        matches_b_c = bf.match(des_b, des_c)

        matches_a_b = sorted(matches_a_b, key=lambda x: x.distance)
        src_pts_a_b = np.float32([kp_a[m.queryIdx].pt for m in matches_a_b]).reshape(-1, 1, 2)
        dst_pts_a_b = np.float32([kp_b[m.trainIdx].pt for m in matches_a_b]).reshape(-1, 1, 2)
        M_a_b, mask_a_b = cv2.findHomography(src_pts_a_b, dst_pts_a_b, cv2.RANSAC, 20)
        matchesMask_a_b = mask_a_b.ravel().tolist()
        out_img_Ransac_a_b = cv2.drawMatches(image_list_a[i], kp_a, image_list_b[i], kp_b, matches_a_b, None,
                                   matchColor=(70, 255, 255),
                                   matchesMask=matchesMask_a_b,
                                   flags=2)
        for m in matches_a_b:
            j, k = m.queryIdx, m.trainIdx
            if matches_a_b[j].distance < 0.7 * matches_a_b[k].distance:
                ratio_matches_a_b.append(m)
        img_matches = cv2.drawMatches(image_list_a[i], kp_a, image_list_b[i], kp_b, ratio_matches_a_b, None,
                                      matchColor=(70, 255, 255),
                                      matchesMask=matchesMask_a_b,
                                      flags=2)

        matches_a_c = sorted(matches_a_c, key=lambda x: x.distance)
        src_pts_a_c = np.float32([kp_a[m.queryIdx].pt for m in matches_a_c]).reshape(-1, 1, 2)
        dst_pts_a_c = np.float32([kp_c[m.trainIdx].pt for m in matches_a_c]).reshape(-1, 1, 2)
        M_a_c, mask_a_c = cv2.findHomography(src_pts_a_c, dst_pts_a_c, cv2.RANSAC, 20)
        matchesMask_a_c = mask_a_c.ravel().tolist()
        out_img_Ransac_a_c = cv2.drawMatches(image_list_a[i], kp_a, image_list_c[0], kp_c, matches_a_c, None,
                                             matchColor=(70, 255, 255),
                                             matchesMask=matchesMask_a_c,
                                             flags=2)

        matches_b_c = sorted(matches_b_c, key=lambda x: x.distance)
        src_pts_b_c = np.float32([kp_b[m.queryIdx].pt for m in matches_b_c]).reshape(-1, 1, 2)
        dst_pts_b_c = np.float32([kp_c[m.trainIdx].pt for m in matches_b_c]).reshape(-1, 1, 2)
        M_b_c, mask_b_c = cv2.findHomography(src_pts_b_c, dst_pts_b_c, cv2.RANSAC, 20)
        matchesMask_b_c = mask_b_c.ravel().tolist()
        out_img_Ransac_b_c = cv2.drawMatches(image_list_b[i], kp_b, image_list_c[0], kp_c, matches_b_c, None,
                                             matchColor=(70, 255, 255),
                                             matchesMask=matchesMask_b_c,
                                             flags=2)
        out_img1 = cv2.drawMatches(image_list_a[i], kp_a, image_list_b[i], kp_b, matches_a_b[0:100], flags=2, outImg=None)
        out_img2 = cv2.drawMatches(image_list_a[i], kp_a, image_list_c[0], kp_c, matches_a_c[0:100], flags=2, outImg=None)
        out_img3 = cv2.drawMatches(image_list_b[i], kp_b, image_list_c[0], kp_c, matches_b_c[0:100], flags=2, outImg=None)

        # plt.imshow(out_img1)
        # plt.show()
        # plt.imshow(out_img2)
        # plt.show()
        # plt.imshow(out_img3)
        # plt.show()
        plt.imshow(out_img_Ransac_a_b)
        plt.show()
        # plt.imshow(out_img_Ransac_a_c)
        # plt.show()
        # plt.imshow(out_img_Ransac_b_c)
        # plt.show()
    else:
        kp_a, des_a = sift.detectAndCompute(image_list_a[i], None)
        kp_b, des_b = sift.detectAndCompute(image_list_b[i], None)
        #matches = bf.match(des_a, des_b)
        matches_D = bf.knnMatch(des_a, des_b, 2)
        for m, n in matches_D:
            if m.distance < 0.9 * n.distance:
                ratio_matches.append(m)


        img_matches = cv2.drawMatches(image_list_a[i], kp_a, image_list_b[i], kp_b, ratio_matches, None,
                                      matchColor=(255, 0, 170),
                                      matchesMask=None,
                                      flags=2)
        for m in ratio_matches:
            if m.distance < 200:
                matches_threshold.append(m)
        out_img_threshold = cv2.drawMatches(image_list_a[i], kp_a, image_list_b[i], kp_b, matches_threshold, None,
                                         matchColor=(0, 0, 255),
                                         matchesMask=None,
                                         flags=2)
        matches = sorted(matches_threshold, key=lambda x: x.distance)
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)
        matchesMask = mask.ravel().tolist()
        out_img0 = cv2.drawMatches(image_list_a[i], kp_a, image_list_b[i], kp_b, matches[0:100], flags=2, outImg=None)
        out_img_Ransac = cv2.drawMatches(image_list_a[i], kp_a, image_list_b[i], kp_b, matches, None,
                                   matchColor=(70, 255, 255),
                                   matchesMask=matchesMask,
                                   flags=2)




        # plt.imshow(img_matches)
        # plt.show()
        plt.imshow(out_img_Ransac)
        plt.show()



# for i in range(len(images) - 1):
#     for j in range(len(images) - 1):
#         kp1, des1 = sift.detectAndCompute(images[i], None)
#         kp2, des2 = sift.detectAndCompute(images[j], None)
#         matches = bf.match(des1, des2)
#         matches = sorted(matches, key=lambda x: x.distance)
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)
#         matchesMask = mask.ravel().tolist()
#
#         # out_img = cv2.drawMatches(images[i], kp1, images[j],
#         # kp2, matches[0:100], flags = 2,
#         # outImg = None
#         out_img = cv2.drawMatches(images[i], kp1, images[j], kp2, matches, None,
#                               matchColor=(0, 255, 0),
#                               matchesMask=matchesMask,
#                               flags=2)
#         plt.imshow(out_img)
#         plt.show()
