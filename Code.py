import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

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
# size = (640, 480)

for img in image_path_jpeg_A:
    image_a = cv2.imread(img, 0)
    # resized_image = cv2.resize(image_a, size)
    image_list_a.append(image_a)

for img in image_path_jpeg_B:
    image_b = cv2.imread(img, 0)
    # resized_image = cv2.resize(image_b, size)
    image_list_b.append(image_b)

for img in image_path_png_C:
    image_c = cv2.imread(img, 0)
    # resized_image = cv2.resize(image_c, size)
    image_list_c.append(image_c)

for i in range(len(image_path_jpeg_A)):
    ratio_matches = []
    ratio_matches_a_b = []
    ratio_matches_a_c = []
    ratio_matches_b_c = []
    matches_threshold_a_b = []
    matches_threshold_a_c = []
    matches_threshold_b_c = []
    matches_crosscheck_a_b = []
    matches_crosscheck_a_c = []
    matches_crosscheck_b_c = []
    matches_threshold = []
    matches_crosscheck = []
    if i == 3:
        kp_a, des_a = sift.detectAndCompute(image_list_a[i], None)
        kp_b, des_b = sift.detectAndCompute(image_list_b[i], None)
        kp_c, des_c = sift.detectAndCompute(image_list_c[0], None)

        # ------------------------------------------start David Lowe’s Ratio for a and b--------------------------------
        matches_D = bf.knnMatch(des_a, des_b, 2)
        for m, n in matches_D:
            if m.distance < 0.95 * n.distance:
                ratio_matches_a_b.append(m)
        # ------------------------------------------end David Lowe’s Ratio for a and b----------------------------------

        # ------------------------------------------start David Lowe’s Ratio for a and c--------------------------------
        matches_D = bf.knnMatch(des_a, des_c, 2)
        for m, n in matches_D:
            if m.distance < 0.95 * n.distance:
                ratio_matches_a_c.append(m)
        # ------------------------------------------end David Lowe’s Ratio for a and c----------------------------------

        # ------------------------------------------start David Lowe’s Ratio for b and c--------------------------------
        matches_D = bf.knnMatch(des_b, des_c, 2)
        for m, n in matches_D:
            if m.distance < 0.95 * n.distance:
                ratio_matches_b_c.append(m)
        # ------------------------------------------end David Lowe’s Ratio for b and c----------------------------------

        # -------------------------------------------------start Threshold a and b----------------------------------------------
        sum_of_distance = 0.0
        for m in ratio_matches_a_b:
            sum_of_distance = sum_of_distance + m.distance
        mean_of_distance = sum_of_distance / len(ratio_matches_a_b)
        # print(len(ratio_matches_a_b))
        # print(mean_of_distance)
        for m in ratio_matches_a_b:
            if m.distance < mean_of_distance:
                matches_threshold_a_b.append(m)
        # -------------------------------------------------end Threshold a and b------------------------------------------------

        # -------------------------------------------------start Threshold a and c----------------------------------------------
        sum_of_distance = 0.0
        for m in ratio_matches_a_c:
            sum_of_distance = sum_of_distance + m.distance
        mean_of_distance = sum_of_distance / len(ratio_matches_a_c)
        # print(len(ratio_matches_a_c))
        # print(mean_of_distance)
        for m in ratio_matches_a_c:
            if m.distance < mean_of_distance:
                matches_threshold_a_c.append(m)
        # -------------------------------------------------end Threshold a and c------------------------------------------------

        # -------------------------------------------------start Threshold b and c----------------------------------------------
        sum_of_distance = 0.0
        for m in ratio_matches_b_c:
            sum_of_distance = sum_of_distance + m.distance
        mean_of_distance = sum_of_distance / len(ratio_matches_b_c)
        # print(len(ratio_matches_b_c))
        # print(mean_of_distance)
        for m in ratio_matches_b_c:
            if m.distance < mean_of_distance:
                matches_threshold_b_c.append(m)
        # -------------------------------------------------end Threshold c and b------------------------------------------------

        # -----------------------------------------------start manual cross check a and b---------------------------------------
        first_image_matches_a_b = [None] * len(kp_a)
        second_image_matches_a_b = [None] * len(kp_b)
        for match in matches_threshold_a_b:
            if first_image_matches_a_b[match.queryIdx] is None:
                first_image_matches_a_b[match.queryIdx] = match
        for match in matches_threshold_a_b:
            if second_image_matches_a_b[match.trainIdx] is None:
                second_image_matches_a_b[match.trainIdx] = match
        for match in matches_threshold_a_b:
            if match.queryIdx == second_image_matches_a_b[match.trainIdx].queryIdx and match.trainIdx == \
                    first_image_matches_a_b[match.queryIdx].trainIdx:
                matches_crosscheck_a_b.append(match)
        # ----------------------------------------------end  manual cross check a and b-----------------------------------------

        # -----------------------------------------------start manual cross check a and c---------------------------------------
        first_image_matches_a_c = [None] * len(kp_a)
        second_image_matches_a_c = [None] * len(kp_c)
        for match in matches_threshold_a_c:
            if first_image_matches_a_c[match.queryIdx] is None:
                first_image_matches_a_c[match.queryIdx] = match
        for match in matches_threshold_a_c:
            if second_image_matches_a_c[match.trainIdx] is None:
                second_image_matches_a_c[match.trainIdx] = match
        for match in matches_threshold_a_c:
            if match.queryIdx == second_image_matches_a_c[match.trainIdx].queryIdx and match.trainIdx == \
                    first_image_matches_a_c[match.queryIdx].trainIdx:
                matches_crosscheck_a_c.append(match)
        # ----------------------------------------------end  manual cross check a and c-----------------------------------------

        # -----------------------------------------------start manual cross check b and c---------------------------------------
        first_image_matches_b_c = [None] * len(kp_b)
        second_image_matches_b_c = [None] * len(kp_c)
        for match in matches_threshold_b_c:
            if first_image_matches_b_c[match.queryIdx] is None:
                first_image_matches_b_c[match.queryIdx] = match
        for match in matches_threshold_b_c:
            if second_image_matches_b_c[match.trainIdx] is None:
                second_image_matches_b_c[match.trainIdx] = match
        for match in matches_threshold_b_c:
            if match.queryIdx == second_image_matches_b_c[match.trainIdx].queryIdx and match.trainIdx == \
                    first_image_matches_b_c[match.queryIdx].trainIdx:
                matches_crosscheck_b_c.append(match)
        # ----------------------------------------------end  manual cross check b and c-----------------------------------------

        # -------------------------------------------------start Ransac a and b-------------------------------------------------
        matches_a_b = sorted(matches_crosscheck_a_b, key=lambda x: x.distance)
        src_pts_a_b = np.float32([kp_a[m.queryIdx].pt for m in matches_a_b]).reshape(-1, 1, 2)
        dst_pts_a_b = np.float32([kp_b[m.trainIdx].pt for m in matches_a_b]).reshape(-1, 1, 2)
        M_a_b, mask_a_b = cv2.findHomography(src_pts_a_b, dst_pts_a_b, cv2.RANSAC, 20)
        matchesMask_a_b = mask_a_b.ravel().tolist()
        out_img_Ransac_a_b = cv2.drawMatches(image_list_a[i], kp_a, image_list_b[i], kp_b, matches_a_b, None,
                                             matchColor=(70, 255, 255),
                                             matchesMask=matchesMask_a_b,
                                             flags=2)
        # -------------------------------------------------end Ransac a and b ---------------------------------------------------

        # -------------------------------------------------start Ransac a and c-------------------------------------------------
        matches_a_c = sorted(matches_crosscheck_a_c, key=lambda x: x.distance)
        src_pts_a_c = np.float32([kp_a[m.queryIdx].pt for m in matches_a_c]).reshape(-1, 1, 2)
        dst_pts_a_c = np.float32([kp_c[m.trainIdx].pt for m in matches_a_c]).reshape(-1, 1, 2)
        M_a_c, mask_a_c = cv2.findHomography(src_pts_a_c, dst_pts_a_c, cv2.RANSAC, 20)
        matchesMask_a_c = mask_a_c.ravel().tolist()
        out_img_Ransac_a_c = cv2.drawMatches(image_list_a[i], kp_a, image_list_c[0], kp_c, matches_a_c, None,
                                             matchColor=(70, 255, 255),
                                             matchesMask=matchesMask_a_c,
                                             flags=2)
        # -------------------------------------------------end Ransac a and c---------------------------------------------------

        # -------------------------------------------------start Ransac b and c-------------------------------------------------
        matches_b_c = sorted(matches_crosscheck_b_c, key=lambda x: x.distance)
        src_pts_b_c = np.float32([kp_b[m.queryIdx].pt for m in matches_b_c]).reshape(-1, 1, 2)
        dst_pts_b_c = np.float32([kp_c[m.trainIdx].pt for m in matches_b_c]).reshape(-1, 1, 2)
        M_b_c, mask_b_c = cv2.findHomography(src_pts_b_c, dst_pts_b_c, cv2.RANSAC, 20)
        matchesMask_b_c = mask_b_c.ravel().tolist()
        out_img_Ransac_b_c = cv2.drawMatches(image_list_b[i], kp_b, image_list_c[0], kp_c, matches_b_c, None,
                                             matchColor=(70, 255, 255),
                                             matchesMask=matchesMask_b_c,
                                             flags=2)

        # -------------------------------------------------end Ransac b and c---------------------------------------------------
        score = len(matches_a_b) / min(len(kp_a), len(kp_b))
        formatted_score = f"{score:.2f}"
        if score > 0.111:
            is_match = "Match"
        else:
            is_match = "Not Match"
        print(score)
        plt.title(f"Score: {formatted_score}, Result: {is_match}")
        plt.imshow(out_img_Ransac_a_b)
        plt.show()

        score = len(matches_a_c) / min(len(kp_a), len(kp_b))
        formatted_score = f"{score:.2f}"
        if score > 0.111:
            is_match = "Match"
        else:
            is_match = "Not Match"
        print(score)
        plt.title(f"Score: {formatted_score}, Result: {is_match}")
        plt.imshow(out_img_Ransac_a_c)
        plt.show()

        score = len(matches_b_c) / min(len(kp_a), len(kp_b))
        formatted_score = f"{score:.2f}"
        if score > 0.111:
            is_match = "Match"
        else:
            is_match = "Not Match"
        print(score)
        plt.title(f"Score: {formatted_score}, Result: {is_match}")
        plt.imshow(out_img_Ransac_b_c)
        plt.show()
    else:
        kp_a, des_a = sift.detectAndCompute(image_list_a[i], None)
        kp_b, des_b = sift.detectAndCompute(image_list_b[i], None)
        # matches = bf.match(des_a, des_b)
        # ------------------------------------------start David Lowe’s Ratio--------------------------------------------
        matches_D = bf.knnMatch(des_a, des_b, 2)
        for m, n in matches_D:
            if m.distance < 0.95 * n.distance:
                ratio_matches.append(m)
        img_matches = cv2.drawMatches(image_list_a[i], kp_a, image_list_b[i], kp_b, ratio_matches, None,
                                      matchColor=(255, 0, 170),
                                      matchesMask=None,
                                      flags=2)
        # --------------------------------------------end David Lowe’s Ratio--------------------------------------------

        # -------------------------------------------------start Threshold----------------------------------------------
        sum_of_distance = 0.0
        for m in ratio_matches:
            sum_of_distance = sum_of_distance + m.distance
        mean_of_distance = sum_of_distance / len(ratio_matches)
        # print(len(ratio_matches))
        # print(mean_of_distance)
        for m in ratio_matches:
            if m.distance < mean_of_distance:
                matches_threshold.append(m)
        # -------------------------------------------------end Threshold------------------------------------------------

        # -----------------------------------------------start manual cross check---------------------------------------
        first_image_matches = [None] * len(kp_a)
        second_image_matches = [None] * len(kp_b)
        for match in matches_threshold:
            if first_image_matches[match.queryIdx] is None:
                first_image_matches[match.queryIdx] = match
        for match in matches_threshold:
            if second_image_matches[match.trainIdx] is None:
                second_image_matches[match.trainIdx] = match
        for match in matches_threshold:
            if match.queryIdx == second_image_matches[match.trainIdx].queryIdx and match.trainIdx == \
                    first_image_matches[match.queryIdx].trainIdx:
                matches_crosscheck.append(match)
        # ----------------------------------------------end  manual cross check-----------------------------------------

        # -------------------------------------------------start Ransac-------------------------------------------------
        matches = sorted(matches_crosscheck, key=lambda x: x.distance)
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 40)
        matchesMask = mask.ravel().tolist()
        out_img0 = cv2.drawMatches(image_list_a[i], kp_a, image_list_b[i], kp_b, matches, flags=2, outImg=None)
        out_img_Ransac = cv2.drawMatches(image_list_a[i], kp_a, image_list_b[i], kp_b, matches, None,
                                         matchColor=(70, 255, 255),
                                         matchesMask=matchesMask,
                                         flags=2)
        final_match = out_img_Ransac
        # -------------------------------------------------end Ransac---------------------------------------------------
        score = len(matches) / min(len(kp_a), len(kp_b))
        formatted_score = f"{score:.2f}"
        if score > 0.111:
            is_match = "Match"
        else:
            is_match = "Not Match"
        print(score)
        plt.title(f"Score: {formatted_score}, Result: {is_match}")
        plt.imshow(final_match)
        plt.show()



print("Note --start code that take one image and loop on other image--")






# ---------------------------------start code that take one image and loop on other image-------------------------------
sift = cv2.SIFT_create()
image_path_jpeg = glob.glob('assignment data/*.jpeg')
image_path_png = glob.glob('assignment data/*.png')
image_path = image_path_jpeg + image_path_png
bf = cv2.BFMatcher()
images = []
size = (640, 480)

for img in image_path:
    image = cv2.imread(img, 0)
    resized_image = cv2.resize(image, size)
    images.append(resized_image)

for i in range(len(images) - 1):
    for j in range(len(images) - 1):
        ratio_matches = []
        ratio_matches_a_b = []
        matches_threshold = []
        matches_crosscheck = []
        kp_a, des_a = sift.detectAndCompute(images[i], None)
        kp_b, des_b = sift.detectAndCompute(images[j], None)
        # matches = bf.match(des_a, des_b)
        # ------------------------------------------start David Lowe’s Ratio--------------------------------------------
        matches_D = bf.knnMatch(des_a, des_b, 2)
        for m, n in matches_D:
            if m.distance < 0.85 * n.distance:
                ratio_matches.append(m)
        img_matches = cv2.drawMatches(images[i], kp_a, images[j], kp_b, ratio_matches, None,
                                      matchColor=(255, 0, 170),
                                      matchesMask=None,
                                      flags=2)
        # --------------------------------------------end David Lowe’s Ratio--------------------------------------------

        # -------------------------------------------------start Threshold----------------------------------------------
        sum_of_distance = 0.0
        for m in ratio_matches:
            sum_of_distance = sum_of_distance + m.distance
        mean_of_distance = sum_of_distance / len(ratio_matches)
        # print(len(ratio_matches))
        # print(mean_of_distance)
        for m in ratio_matches:
            if m.distance < mean_of_distance:
                matches_threshold.append(m)
        out_img_threshold = cv2.drawMatches(images[i], kp_a, images[j], kp_b, matches_threshold, None,
                                            matchColor=(0, 0, 255),
                                            matchesMask=None,
                                            flags=2)
        # -------------------------------------------------end Threshold------------------------------------------------

        # -----------------------------------------------start manual cross check---------------------------------------
        first_image_matches = [None] * len(kp_a)
        second_image_matches = [None] * len(kp_b)
        for match in matches_threshold:
            if first_image_matches[match.queryIdx] is None:
                first_image_matches[match.queryIdx] = match
        for match in matches_threshold:
            if second_image_matches[match.trainIdx] is None:
                second_image_matches[match.trainIdx] = match
        for match in matches_threshold:
            if match.queryIdx == second_image_matches[match.trainIdx].queryIdx and match.trainIdx == \
                    first_image_matches[match.queryIdx].trainIdx:
                matches_crosscheck.append(match)
        out_img_crosscheck = cv2.drawMatches(images[i], kp_a, images[j], kp_b, matches_crosscheck, None,
                                             matchColor=(70, 255, 255),
                                             matchesMask=None,
                                             flags=2)
        # ----------------------------------------------end  manual cross check-----------------------------------------

        # -------------------------------------------------start Ransac-------------------------------------------------
        matches = sorted(matches_crosscheck, key=lambda x: x.distance)
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        if len(matches) >= 4:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0)
            matchesMask = mask.ravel().tolist()
            out_img0 = cv2.drawMatches(images[i], kp_a, images[j], kp_b, matches, flags=2, outImg=None)
            out_img_Ransac = cv2.drawMatches(images[i], kp_a, images[j], kp_b, matches, None,
                                             matchColor=(70, 255, 255),
                                             matchesMask=matchesMask,
                                             flags=2)
            final_match = out_img_Ransac
            score = len(matches) / min(len(kp_a), len(kp_b))
            formatted_score = f"{score:.2f}"
            if float(formatted_score) >= 0.06:
                is_match = "Match"
            else:
                is_match = "Not Match"
            print(score)
            plt.title(f"Score: {formatted_score}, Result: {is_match}")
            plt.imshow(final_match)
            plt.show()
        else:
            pass
        # -------------------------------------------------end Ransac---------------------------------------------------


        # -----------------------------------------------end second code -----------------------------------------------
