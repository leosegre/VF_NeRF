import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import json
import open3d as o3d
from nerfstudio.utils import poses as pose_utils
import torch
import copy
import random
import os
import cv2
from scipy import spatial
import random
import math
from scipy.spatial.transform import Rotation as R


def rigid_transform_3D(A, B, scale=False):
    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.dot(np.transpose(BB), AA)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("Reflection detected")
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)


    t = np.dot(-R , centroid_B.T) + centroid_A.T

    return R, t
#Kabsch Algorithm
def compute_transformation(source,target):
    #Normalization
    number = len(source)
    #the centroid of source points
    cs = np.zeros((3,1))
    #the centroid of target points
    ct = copy.deepcopy(cs)
    cs[0] = np.mean(source[:][0]);cs[1]=np.mean(source[:][1]);cs[2]=np.mean(source[:][2])
    ct[0] = np.mean(target[:][0]);ct[1]=np.mean(target[:][1]);ct[2]=np.mean(target[:][2])
    #covariance matrix
    cov = np.zeros((3,3))
    #translate the centroids of both models to the origin of the coordinate system (0,0,0)
    #subtract from each point coordinates the coordinates of its corresponding centroid
    for i in range(number):
        sources = source[i].reshape(-1,1)-cs
        targets = target[i].reshape(-1,1)-ct
        cov = cov + np.dot(sources,np.transpose(targets))
    #SVD (singular values decomposition)
    u,w,v = np.linalg.svd(cov)
    #rotation matrix
    R = np.dot(u,np.transpose(v))
    # special reflection case
    if np.linalg.det(R) < 0:
        # print("Reflection detected")
        v *= -1
        R = np.dot(u,np.transpose(v))
    #Transformation vector
    T = ct - np.dot(R,cs)
    return R, T

#compute the transformed points from source to target based on the R/T found in Kabsch Algorithm
def _transform(source,R,T):
    points = []
    for point in source:
        points.append(np.dot(R,point.reshape(-1,1)+T))
    return points

#compute the root mean square error between source and target
def compute_rmse(source,target,R,T):
    rmse = 0
    number = len(target)
    points = _transform(source,R,T)
    for i in range(number):
        error = target[i].reshape(-1,1)-points[i]
        rmse = rmse + math.sqrt(error[0][0]**2+error[1][0]**2+error[2][0]**2)
    return rmse

def draw_registrations(source, target, transformation = None, recolor = False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if(recolor): # recolor the points
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    if(transformation is not None): # transforma source to targets
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def pc2array(pointcloud):
    return np.asarray(pointcloud.points)

def registration_RANSAC(source,target,source_feature,target_feature,ransac_n=3,max_iteration=64,max_validation=100):
    #the intention of RANSAC is to get the optimal transformation between the source and target point cloud
    s = pc2array(source) #(4760,3)
    t = pc2array(target)
    #source features (33,4760)
    sf = np.transpose(source_feature.data)
    tf = np.transpose(target_feature.data)
    #create a KD tree
    tree = spatial.KDTree(tf)
    corres_stock = tree.query(sf)[1]
    for i in range(max_iteration):
        #take ransac_n points randomly
        idx = [random.randint(0,s.shape[0]-1) for j in range(ransac_n)]
        corres_idx = corres_stock[idx]
        source_point = s[idx,...]
        target_point = t[corres_idx,...]
        #estimate transformation
        #use Kabsch Algorithm
        R, T = compute_transformation(source_point,target_point)
        # R, T = rigid_transform_3D(source_point, target_point)
        #calculate rmse for all points
        source_point = s
        target_point = t[corres_stock,...]
        rmse = compute_rmse(source_point,target_point,R,T)
        #compare rmse and optimal rmse and then store the smaller one as optimal values
        if not i:
            opt_rmse = rmse
            opt_R = R
            opt_T = T
        else:
            if rmse < opt_rmse:
                opt_rmse = rmse
                opt_R = R
                opt_T = T
    return opt_R, opt_T

#used for downsampling
# voxel_size = 0.05
#this is to get the fpfh features, just call the library
def get_fpfh(cp, voxel_size=0.05):
    cp = cp.voxel_down_sample(voxel_size)
    cp.estimate_normals()
    return cp, o3d.pipelines.registration.compute_fpfh_feature(cp, o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100))

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    # new_fpfh = np.concatenate((pcd_fpfh.data, np.asarray(pcd_down.colors).T), axis=0)
    # pcd_fpfh.data = new_fpfh
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    # import ipdb; ipdb.set_trace();
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.90),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold),
            # o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(1.0)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.99999))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold,
            iteration_number=64,
        ))
    return result


def sample_points_uniformly(point_cloud, num_samples):
    # Sample points uniformly from the point cloud
    indices = np.random.choice(np.arange(len(point_cloud.points)), num_samples, replace=False)
    sampled_points = np.asarray(point_cloud.points)[indices]
    sampled_point_cloud = o3d.geometry.PointCloud()
    sampled_point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
    return sampled_point_cloud

def extract_rotation_translation_3d(registration_matrix):
    # Extract rotation and translation using scipy's Rotation class
    rotation_matrix = registration_matrix[:3, :3]
    translation = registration_matrix[:3, 3]

    # Convert rotation matrix to a rotation object
    rotation = R.from_matrix(rotation_matrix)

    # Extract euler angles
    rotation_euler = rotation.as_euler('xyz', degrees=True)  # Change 'xyz' to match your convention

    return rotation_euler, translation


def main(path_A, path_B, output_path, max_iterations, num_samples):
    # Read point cloud data from ply files
    A = o3d.io.read_point_cloud(path_A)
    B = o3d.io.read_point_cloud(path_B)
    seed = 42
    o3d.utility.random.seed(seed)
    random.seed(seed)

    # A_sampled = sample_points_uniformly(A, num_samples)
    # B_sampled = sample_points_uniformly(B, num_samples)

    # A_points = np.asarray(A_sampled.points)
    # B_points = np.asarray(B_sampled.points)

    # Call FGR function
    # Run Fast Global Registration (FGR) to register the point clouds
    voxel_size = 0.05  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(copy.deepcopy(A),
                                                                                         copy.deepcopy(B), voxel_size)
    # with open(os.path.join(os.path.dirname(output_path), "features.json"), 'w') as json_file:
    #     json.dump({"features": source_fpfh.data.tolist()}, json_file, indent=4)

    # if we want to use RANSAC registration, get_fpfh features should be acquired firstly
    # r1, f1 = get_fpfh(source)
    # r2, f2 = get_fpfh(target)
    # rot, trans = registration_RANSAC(r1, r2, f1, f2, max_iteration=1000)
    # transformation matrix is formed by R, T based on np.hstack and np.vstack(corporate two matrices by rows)
    # Notice we need add the last row [0 0 0 1] to make it homogeneous
    # print(rot)
    # print(trans)
    # transformation = np.vstack((np.hstack((np.float64(rot), np.float64(trans))), np.array([0, 0, 0, 1])))

    # result = execute_fast_global_registration(source_down, target_down,
    #                                           source_fpfh, target_fpfh,
    #                                           voxel_size)

    for i in range(1):
        o3d.utility.random.seed(i)
        random.seed(i)

        result = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
        if i == 0:
            best_result = result
        elif result.inlier_rmse < best_result.inlier_rmse:
            best_result = result


    # threshold = 0.02
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source_down, target_down, threshold, result.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    # # colored pointcloud registration
    # # This is implementation of following paper
    # # J. Park, Q.-Y. Zhou, V. Koltun,
    # # Colored Point Cloud Registration Revisited, ICCV 2017
    # voxel_radius = [0.04, 0.02, 0.01]
    # max_iter = [50, 30, 14]
    # # current_transformation = np.identity(4)
    # current_transformation = result.transformation
    # print("3. Colored point cloud registration")
    # for scale in range(3):
    #     iter = max_iter[scale]
    #     radius = voxel_radius[scale]
    #     print([iter, radius, scale])
    #
    #     print("3-1. Downsample with a voxel size %.2f" % radius)
    #     source_down = source.voxel_down_sample(radius)
    #     target_down = target.voxel_down_sample(radius)
    #
    #     print("3-2. Estimate normal.")
    #     source_down.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    #     target_down.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    #
    #     print("3-3. Applying colored point cloud registration")
    #     result_icp = o3d.pipelines.registration.registration_colored_icp(
    #         source_down, target_down, radius, current_transformation,
    #         o3d.pipelines.registration.TransformationEstimationForColoredICP(),
    #         o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
    #                                                           relative_rmse=1e-6,
    #                                                           max_iteration=iter))
    #     # current_transformation = result_icp.transformation

    # Convert Open3D registration result to dictionary

    registration_matrix = best_result.transformation
#     registration_matrix = np.array(
#         [
#             [
#                 0.3965458100925794,
#                 0.3633671281320545,
#                 0.8430395770767456,
#                 0.24326184072891058
#             ],
#             [
#                 0.820538567917696,
#                 -0.5520994074298269,
#                 -0.14799568576703948,
#                 -0.2540083146368506
#             ],
#             [
#                 0.41166480745756573,
#                 0.7504335115499323,
#                 -0.51708941575711,
#                 -0.15832928590143291
#             ],
#             [
#                 0.0,
#                 0.0,
#                 0.0,
#                 1.0
#             ]
#         ]
# )
    print(registration_matrix)
    # rotation_degrees, translation = extract_rotation_translation_3d(registration_matrix)
    # print("Rotation (degrees) around x, y, z axes:", rotation_degrees)
    # print("Translation (x, y, z):", translation)
    # print("Det:", np.linalg.det(registration_matrix[:3, :3]))

    # if np.linalg.det(registration_matrix[:3, :3]) < 0:
    #     # registration_matrix[1, :] *= -1
    #     registration_matrix[0:3, 1:3] *= -1
    #     registration_matrix = registration_matrix[np.array([1, 0, 2, 3]), :]
    #     registration_matrix[2, :] *= -1
    # print(registration_matrix)
    # rotation_degrees, translation = extract_rotation_translation_3d(registration_matrix)
    # print("Rotation (degrees) around x, y, z axes:", rotation_degrees)
    # print("Translation (x, y, z):", translation)
    # print("Det:", np.linalg.det(registration_matrix[:3, :3]))

    result_dict = {
        "t0_matrix": registration_matrix.tolist(),
    }

    # Save results to JSON file
    with open(output_path, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)

    transformed_A = copy.deepcopy(source).transform(registration_matrix)
    # transformed_B = copy.deepcopy(B).transform(registration_matrix)
    output_path = Path(output_path)
    output_path_A = str(output_path.parent / "Transformed_A.ply")
    # output_path_B = str(output_path.parent / "Transformed_B.ply")
    o3d.io.write_point_cloud(output_path_A, transformed_A)
    # o3d.io.write_point_cloud(output_path_B, transformed_B)
    # o3d.visualization.draw_geometries([transformed_A, B])


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ICP with ply point clouds')
    parser.add_argument('path_A', type=str, help='Path to ply file A')
    parser.add_argument('path_B', type=str, help='Path to ply file B')
    parser.add_argument('output_path', type=str, help='Output path for JSON file')
    parser.add_argument('--max_iterations', type=int, default=20, help='Maximum iterations (default: 20)')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of samples from each PC (default: 100000)')
    args = parser.parse_args()

    # Call main function with provided arguments
    main(args.path_A, args.path_B, args.output_path, args.max_iterations, args.num_samples)
