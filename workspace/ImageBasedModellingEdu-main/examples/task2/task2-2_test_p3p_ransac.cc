//
// Created by sway on 2021/1/19.
//
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <sstream>

#include <cmath>
#include <random>
#include <vector>
#include <set>

#include <assert.h>

#include "sfm/ransac_pose_p3p.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <unsupported/Eigen/Polynomials>


std::vector<std::pair<double, double>> SolveKneipEquation(
    const Eigen::Vector3d &F1, const Eigen::Vector3d &F2,
    const Eigen::Vector3d &P1, const Eigen::Vector3d &P2,
    const Eigen::Vector3d &F3_new_camera, const Eigen::Vector3d &P3_new_world
) {
    //
    // build Kneip equation:
    //    
    Eigen::Matrix<double, 5, 1> kneip_equation;

    const double p_x = P3_new_world.x();
    const double p_y = P3_new_world.y();

    const double theta_x = F3_new_camera.x() / F3_new_camera.z();
    const double theta_y = F3_new_camera.y() / F3_new_camera.z();

    const double d_12 = (P2 - P1).norm();

    const double b = F1.dot(F2) / F1.cross(F2).norm();

    const double p_x_2 = p_x*p_x;
    const double p_x_3 = p_x_2*p_x;
    const double p_x_4 = p_x_2*p_x_2;

    const double p_y_2 = p_y*p_y;
    const double p_y_3 = p_y_2*p_y;
    const double p_y_4 = p_y_2*p_y_2;

    const double theta_x_2 = theta_x*theta_x;
    const double theta_y_2 = theta_y*theta_y;

    const double d_12_2 = (P2 - P1).squaredNorm();

    const double b_2 = b*b;

    kneip_equation(4,0) = - theta_y_2 * p_y_4 - p_y_4 * theta_x_2 - p_y_4;

    kneip_equation(3,0) =   2.0 * p_y_3 * d_12 * b
                          + 2.0 * theta_y_2 * p_y_3 * d_12 * b
                          - 2.0 * theta_y * p_y_3 * theta_x * d_12;

    kneip_equation(2,0) = - theta_y_2 * p_y_2 * p_x_2
                          - theta_y_2 * p_y_2 * d_12_2 * b_2
                          - theta_y_2 * p_y_2 * d_12_2
                          + theta_y_2 * p_y_4
                          + p_y_4 * theta_x_2
                          + 2.0 * p_x * p_y_2 * d_12
                          + 2.0 * theta_x * theta_y * p_x * p_y_2 * d_12 * b
                          - p_y_2 * p_x_2 * theta_x_2
                          + 2.0 * p_x * p_y_2 * theta_y_2 * d_12
                          - p_y_2 * d_12_2 * b_2
                          - 2.0 * p_x_2 * p_y_2;

    kneip_equation(1,0) =   2.0 * p_x_2 * p_y * d_12 * b
                          + 2.0 * theta_y * p_y_3 * theta_x * d_12
                          - 2.0 * theta_y_2 * p_y_3 * d_12 * b
                          - 2.0 * p_x * p_y * d_12_2 * b;

    kneip_equation(0,0) = - 2.0 * theta_y * p_y_2 * theta_x * p_x * d_12 * b
                          + theta_y_2 * p_y_2 * d_12_2
                          + 2.0 * p_x_3 * d_12
                          - p_x_2 * d_12_2
                          + theta_y_2 * p_y_2 * p_x_2
                          - p_x_4
                          - 2.0 * theta_y_2 * p_y_2 * p_x * d_12
                          + p_y_2 * theta_x_2 * p_x_2
                          + theta_y_2 * p_y_2 * d_12_2 * b_2;

    //
    // solve using QR decomposition:
    //
    Eigen::PolynomialSolver<double, 4> psolve( kneip_equation );
    std::vector<std::pair<double, double>> results;

    const auto &roots = psolve.roots();
    for (size_t i = 0; i < roots.size(); ++i) {
        const double cos_theta = roots(i).real();
        const double cot_alpha = (theta_x*p_x + cos_theta*p_y*theta_y - d_12*b*theta_y) / (theta_x*cos_theta*p_y - p_x*theta_y + d_12*theta_y);

        results.emplace_back(cos_theta, cot_alpha);
    }

    return std::move(results);
}

/**
 * compute 4 candidate camera pose usin Kneip's P3P
 * @param p1
 * @param p2
 * @param p3
 * @param f1
 * @param f2
 * @param f3
 * @param solutions
 */
std::vector<Eigen::Matrix4d> PoseP3PKneip(
    const std::vector<Eigen::Vector3d> &Ps,
    const std::vector<Eigen::Vector3d> &fs
) {
    //
    // format as Eigen:
    //
    const Eigen::Vector3d &P1 = Ps.at(0);
    const Eigen::Vector3d &P2 = Ps.at(1);
    const Eigen::Vector3d &P3 = Ps.at(2);

    const Eigen::Vector3d &F1 = fs.at(0);
    const Eigen::Vector3d &F2 = fs.at(1);
    const Eigen::Vector3d &F3 = fs.at(2);

    //
    // build new world frame:
    //
    Eigen::Matrix3d R_new_world = Eigen::Matrix3d::Identity();
    R_new_world.block<3, 1>(0, 0) = (P2 - P1).normalized();
    R_new_world.block<3, 1>(0, 2) = R_new_world.col(0).cross(P3 - P1).normalized();
    R_new_world.block<3, 1>(0, 1) = R_new_world.col(2).cross(R_new_world.col(0));

    //
    // build new camera frame:
    //
    Eigen::Matrix3d R_new_camera = Eigen::Matrix3d::Identity();
    R_new_camera.block<3, 1>(0, 0) = F1.normalized();
    R_new_camera.block<3, 1>(0, 2) = F1.cross(F2).normalized();
    R_new_camera.block<3, 1>(0, 1) = R_new_camera.col(2).cross(R_new_camera.col(0));

    //
    // project f3 to new camera frame:
    //
    Eigen::Vector3d P3_new_world = R_new_world.transpose()*(P3 - P1);
    Eigen::Vector3d F3_new_camera = R_new_camera.transpose()*F3;

    //
    // get candidate poses:
    //
    auto result = SolveKneipEquation(
        F1, F2, P1, P2, F3_new_camera, P3_new_world
    );

    //
    // construct candidate poses:
    //
    std::vector<Eigen::Matrix4d> poses;

    const double d_12 = (P2 - P1).norm();
    const double b = F1.dot(F2) / F1.cross(F2).norm();
    for (size_t i = 0; i < result.size(); ++i) {
        const auto &params = result.at(i);

        // TODO: handle the sign of theta
        const double &cos_theta = params.first;
        const double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        const double &cot_alpha = params.second;

        const double sin_alpha = std::sqrt(1.0 / (cot_alpha*cot_alpha + 1.0));
        const double cos_alpha = (cot_alpha < 0.0 ? -1.0 : 1.0) * std::sqrt(1.0 - sin_alpha*sin_alpha);

        //
        // 1. rotation around x, new world frame:
        //
        Eigen::Matrix3d R_theta = Eigen::Matrix3d::Identity();
        R_theta(1, 1) = cos_theta; R_theta(1, 2) = -sin_theta;
        R_theta(2, 1) = sin_theta; R_theta(2, 2) =  cos_theta;

        //
        // 2. rotation around z, new world frame:
        //
        Eigen::Matrix3d R_alpha = Eigen::Matrix3d::Identity();
        R_alpha(0, 0) = -cos_alpha; R_alpha(0, 1) =  sin_alpha;
        R_alpha(1, 0) = -sin_alpha; R_alpha(1, 1) = -cos_alpha;

        //
        // camera pose:
        //
        // 1. orientation:
        Eigen::Matrix3d R_camera = R_new_world*R_theta*R_alpha*R_new_camera.transpose();
        // 2. position:
        const double cp1 = d_12*(sin_alpha*b + cos_alpha);
        Eigen::Vector3d t_camera(cp1*cos_alpha, cp1*sin_alpha, 0.0);
        t_camera = R_new_world*R_theta*t_camera + P1;

        //
        // follow output specification:
        //
        Eigen::Matrix4d T_camera = Eigen::Matrix4d::Identity();
        T_camera.block<3, 3>(0, 0) = R_camera;
        T_camera.block<3, 1>(0, 3) = t_camera;

        poses.push_back(T_camera);
    }

    return std::move(poses);
}

int GetNumInliers(
    const sfm::Correspondences2D3D &corrs,
    const Eigen::Matrix3d &K,
    const Eigen::Matrix4d &T_camera,
    const double &threshold
) {
    int num_inliers = 0;

    const Eigen::Matrix3d R_camera = T_camera.block<3, 3>(0, 0);
    const Eigen::Vector3d t_camera = T_camera.block<3, 1>(0, 3);

    for (const auto &corr: corrs) {
        Eigen::Map<const Eigen::Vector3d> P(corr.p3d);
        Eigen::Map<const Eigen::Vector2d> p(corr.p2d);

        Eigen::Vector3d p_homo = K*R_camera.transpose()*(P - t_camera);
        Eigen::Vector2d p_observed(
            p_homo.x() / p_homo.z(),
            p_homo.y() / p_homo.z()
        );

        if ( (p - p_observed).norm() < threshold ) ++num_inliers;
    }

    return num_inliers;
}

int main(int argc, char* argv[]){
    // 相机内参矩阵
    math::Matrix<double, 3, 3>k_matrix;
    k_matrix.fill(0.0);
    k_matrix[0] = 0.972222;
    k_matrix[2] = 0.0; // cx =0 图像上的点已经进行了归一化（图像中心为原点，图像尺寸较长的边归一化为1）
    k_matrix[4] = 0.972222;
    k_matrix[5] = 0.0; // cy=0  图像上的点已经进行了归一化（图像中心为原点，图像尺寸较长的边归一化为1）
    k_matrix[8] = 1.0;

    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
    K(0, 0) = k_matrix[0]; K(0, 2) = k_matrix[2];
    K(1, 1) = k_matrix[4]; K(1, 2) = k_matrix[5];
    K(2, 2) = k_matrix[8];
    Eigen::Matrix3d K_inv = K.inverse();

    // 从文件中读取3D-2D对应点，2D点已经进行归一化
    sfm::Correspondences2D3D corrs;
    std::ifstream fin("./correspondence2D3D.txt");
    assert(fin.is_open());
    std::string line;
    int line_id = 0;
    int n_pts = 0;
    while(getline(fin, line)){
        std::stringstream  stream(line);
        if(line_id==0){
            stream>>n_pts;
            //std::cout<<"n_pts: "<<n_pts<<std::endl;
            line_id++;
            continue;
        }
        sfm::Correspondence2D3D corr;
        stream>>corr.p3d[0]>>corr.p3d[1]>>corr.p3d[2]>>corr.p2d[0]>>corr.p2d[1];
        corrs.push_back(corr);
        //std::cout<<corr.p3d[0]<<" "<<corr.p3d[1]<<" "
        //<<corr.p3d[2]<<" "<<corr.p2d[0]<<" "<<corr.p2d[1]<<std::endl;
    }

    // Ransac中止条件，内点阈判断
    sfm::RansacPoseP3P::Options pose_p3p_opts;

    // Ransac估计相机姿态
    // sfm::RansacPoseP3P::Result ransac_result;
    // sfm::RansacPoseP3P ransac(pose_p3p_opts);
    // ransac.estimate(corrs, k_matrix, &ransac_result);

    //
    // here implement RANSAC as a simple procedure:
    // 
    // at least 3 correspondences are needed:
    if ( corrs.size() < 3 ) {
        throw std::invalid_argument("At least 3 correspondences are needed. Check input data.");
    } 

    //
    // RANSAC:
    //

    // result:
    int num_inliers_optimal = 0;
    Eigen::Matrix4d T_camera_optimal;

    // init random correspondence generator:
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, corrs.size() - 1);
    pose_p3p_opts.max_iterations = 10000;
    for (int i = 0; i < pose_p3p_opts.max_iterations; ++i) {
        //
        // generate candidate:
        //
        std::set<int> indices;
        while( indices.size() < 3 ){
            int idx = distrib(gen);
            indices.insert(idx);
        }

        //
        // solve Kneip P3P:
        //
        std::vector<Eigen::Vector3d> Ps;
        std::vector<Eigen::Vector3d> fs;
        for (const int &index: indices) {
            Eigen::Map<const Eigen::Vector3d> P(corrs.at(index).p3d);
            Eigen::Map<const Eigen::Vector2d> f(corrs.at(index).p2d);

            Ps.push_back(P);
            fs.push_back(K_inv*Eigen::Vector3d(f.x(), f.y(), 1.0));
        }

        std::vector<Eigen::Matrix4d> poses = PoseP3PKneip(Ps, fs);

        //
        // evaluate:
        //
        for (const auto &T: poses) {
            int num_inliers = GetNumInliers(corrs, K, T, pose_p3p_opts.threshold);

            if ( num_inliers > num_inliers_optimal ) {
                num_inliers_optimal = num_inliers;
                T_camera_optimal = T;
            }
        }
    }

    std::cout << "2D-3D correspondences inliers: " << (100 * num_inliers_optimal / corrs.size())<<std::endl;
    std::cout << "Estimated pose: " << std::endl;
    std::cout << T_camera_optimal.inverse().block<3, 4>(0, 0) << std::endl;

    std::cout<<"The result pose should be:"<<std::endl;
    std::cout<<
    "0.99896 0.0341342 -0.0302263 -0.292601\n"
    "-0.0339703 0.999405 0.0059176 -4.6632\n"
    "0.0304104 -0.00488465 0.999526 -0.0283862\n"<<std::endl;

    return EXIT_SUCCESS;
}

