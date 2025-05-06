true_points = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0], 
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0]
])

# Пример координат
pred_points_res1 = np.array([
    [-1, -5],
    [3, -5],
    [2, -1], 
    [0.0, 0.0],
    [2, 0],
    [4, -2]
])

pred_points_res2 = np.array([
    [0,-3],
    [4,-3],
    [2, -1], 
    [0.0, 0.0],
    [2,1],
    [2,-2]
])

pred_points_res3 = np.array([
    [1,-5],
    [4,-4],
    [2, -1], 
    [0.0, 0.0],
    [2,1],
    [2,-3]
])


ma = euclidean_matching_accuracy(pred_points_res1, true_points)
print(f"pred_points_res1: {ma:.3f} pixels")

ma = euclidean_matching_accuracy(pred_points_res2, true_points)
print(f"pred_points_res2: {ma:.3f} pixels")

ma = euclidean_matching_accuracy(pred_points_res3, true_points)
print(f"pred_points_res2: {ma:.3f} pixels")

# ------------------------------------------------------------------

coordanates_res1 = [
    (-1, -5), # 1
    (3, -5),  # 2
    (2, -1), # 3
    (0, 0), # 4
    (2, 0), # 5
    (4, -2) # 6
]

coordanates_res2 = [
    (0,-3), # 1
    (4,-3),  # 2
    (2, -1), # 3
    (0, 0), # 4
    (2,1), # 5
    (2,-2) # 6
]

coordanates_res3 = [
    (1,-5), # 1
    (4,-4),  # 2
    (2, -1), # 3
    (0, 0), # 4
    (2, 1), # 5
    (2,-3) # 6
]

for i in range (len(file_names)):
    shift_image(f'pics/test1/resized/{file_names[i]}', coordanates_res3[i][0], coordanates_res3[i][1], f'pics/test1/shifted_fragments_res3/{file_names[i]}')

overlay_images_from_folder('pics/test1/shifted_fragments_res3', 'pics/test1/combined_fragments_res3.png')


#--------------------------------------------------------------------------


img1 = cv2.imread('pics/test2/combined_fragments_resized.png')
img2 = cv2.imread('pics/test2/combined_fragments_res1.png')

tsi_glcm, tsi_lbp = compute_texture_similarity_components(img1, img2)
print(f"TSI (GLCM): {tsi_glcm:.4f}")
print(f"TSI (LBP):  {tsi_lbp:.4f}")


img1 = cv2.imread('pics/test2/combined_fragments_resized.png')
img2 = cv2.imread('pics/test2/combined_fragments_res2.png')

tsi_glcm, tsi_lbp = compute_texture_similarity_components(img1, img2)
print(f"TSI (GLCM): {tsi_glcm:.4f}")
print(f"TSI (LBP):  {tsi_lbp:.4f}")


img1 = cv2.imread('pics/test2/combined_fragments_resized.png')
img2 = cv2.imread('pics/test2/combined_fragments_res3.png')

tsi_glcm, tsi_lbp = compute_texture_similarity_components(img1, img2)
print(f"TSI (GLCM): {tsi_glcm:.4f}")
print(f"TSI (LBP):  {tsi_lbp:.4f}")