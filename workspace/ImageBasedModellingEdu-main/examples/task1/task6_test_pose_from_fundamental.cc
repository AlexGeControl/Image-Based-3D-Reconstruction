//
// Created by caoqi on 2018/9/5.
//

#include <cmath>
#include <math/matrix.h>
#include <math/matrix_svd.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

typedef math::Matrix<double, 3, 3> FundamentalMatrix;
typedef math::Matrix<double, 3, 3> EssentialMatrix;

//用于测试相机姿态的正确性
Eigen::Vector3d p1(0.18012331426143646, -0.156584024429321290, 0.0);
Eigen::Vector3d p2(0.20826430618762970, -0.035404585301876068, 0.0);

/*第一个相机的内外参数*/
double f1 = 0.972222208;
/*第二个相机的内外参数*/
double f2 = 0.972222208;

/**
 * \description 对匹配点进行三角化得到空间三维点
 * @param p1 -- 第一幅图像中的特征点
 * @param p2 -- 第二幅图像中的特征点
 * @param K1 -- 第一幅图像的内参数矩阵
 * @param R1 -- 第一幅图像的旋转矩阵
 * @param t1 -- 第一幅图像的平移向量
 * @param K2 -- 第二幅图像的内参数矩阵
 * @param R2 -- 第二幅图像的旋转矩阵
 * @param t2 -- 第二幅图像的平移向量
 * @return 三维点
 */
Eigen::Vector3d triangulation(
    const Eigen::Matrix3d &K1, const Eigen::Matrix3d &K2,
    const Eigen::Matrix4d &T
) {
    // 
    // a. build projection matrices:
    //
    Eigen::Matrix4d P1 = Eigen::Matrix4d::Identity();
    P1.block<3, 4>(0, 0) = K1 * P1.block<3, 4>(0, 0);

    Eigen::Matrix4d P2 = T;
    P2.block<3, 4>(0, 0) = K2 * P2.block<3, 4>(0, 0);

    // 
    // b. construct linear systems:
    //
    Eigen::Matrix4d A;
    A.block<1, 4>(0, 0) = P1.block<1, 4>(0, 0) - p1(0) * P1.block<1, 4>(2, 0);
    A.block<1, 4>(1, 0) = P1.block<1, 4>(1, 0) - p1(1) * P1.block<1, 4>(2, 0);
    A.block<1, 4>(2, 0) = P2.block<1, 4>(0, 0) - p2(0) * P2.block<1, 4>(2, 0);
    A.block<1, 4>(3, 0) = P2.block<1, 4>(1, 0) - p2(1) * P2.block<1, 4>(2, 0);

    // 
    // c. solve point 3d Position:
    //
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);

    Eigen::VectorXd X_homo = svd.matrixV().col(3);
    Eigen::Vector3d X(
        X_homo(0) / X_homo(3),
        X_homo(1) / X_homo(3),
        X_homo(2) / X_homo(3)
    );

    return X;
}
/**
 * \description 判断相机姿态是否正确，方法是计算三维点在两个相机中的坐标，要求其z坐标大于0，即
 * 三维点同时位于两个相机的前方
 * @param match
 * @param pose1
 * @param pose2
 * @return
 */
bool is_correct_pose(
    const Eigen::Matrix3d &K1, const Eigen::Matrix3d &K2,
    const Eigen::Matrix4d &T
) {
    Eigen::Vector3d P = triangulation(K1, K2, T);
    Eigen::Vector3d P_cam = T.block<3, 3>(0, 0) * P + T.block<3, 1>(0, 3);

    return P.z() > 0.0 && P_cam.z() > 0.0;
}

bool calc_cam_poses(
    FundamentalMatrix const &F, 
    const double f1, const double f2, 
    Eigen::Matrix3d& R, Eigen::Vector3d& t
) {
    //
    // a. calculate essential matrix:
    //
    Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
    K1(0, 0) = K1(1, 1) = f1;

    Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
    K2(0, 0) = K2(1, 1) = f2;

    Eigen::Matrix3d E;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            E(i, j) = F(i, j);
        }
    }
    E = K2.transpose() * E * K1;

    std::cout << "EssentialMatrix result is " << std::endl 
              << E << std::endl;
    std::cout << "EssentialMatrix should be: \n"
              << "-0.00490744 -0.0146139 0.34281\n"
              << "0.0212215 -0.000748851 -0.0271105\n"
              << "-0.342111 0.0315182 -0.00552454\n";

    //
    // b. generate camera pose proposals:
    //
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        E, 
        Eigen::ComputeFullU | Eigen::ComputeFullV
    );

    Eigen::Matrix3d R_z = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).matrix();

    Eigen::Vector3d t1 = -svd.matrixU().col(2);
    Eigen::Vector3d t2 =  svd.matrixU().col(2);
    Eigen::Matrix3d R1 =  svd.matrixU() * R_z.transpose() * svd.matrixV().transpose();
    if ( R1.determinant() < 0.0 ) {
        R1 = -R1;
    }
    Eigen::Matrix3d R2 =  svd.matrixU() * R_z * svd.matrixV().transpose();
    if ( R2.determinant() < 0.0 ) {
        R2 = -R2;
    }
    std::vector<Eigen::Matrix4d> candidate_poses;

    Eigen::Matrix4d T;

    T.block<3, 3>(0, 0) = R1; T.block<3, 1>(0, 3) = t1;
    candidate_poses.push_back(T);
    T.block<3, 3>(0, 0) = R1; T.block<3, 1>(0, 3) = t2;
    candidate_poses.push_back(T);
    T.block<3, 3>(0, 0) = R2; T.block<3, 1>(0, 3) = t1;
    candidate_poses.push_back(T);
    T.block<3, 3>(0, 0) = R2; T.block<3, 1>(0, 3) = t2;
    candidate_poses.push_back(T);

    //
    // c. find correct pose:
    // 
    for (const auto &pose: candidate_poses) {
        if ( is_correct_pose(K1, K2, pose) ) {
            R = pose.block<3, 3>(0, 0);
            t = pose.block<3, 1>(0, 3);

            std::cout << "Valid camera pose: " << std::endl;
        } else {
            std::cout << "INVALID camera pose: " << std::endl;
        }

        std::cout << "\tR:" << std::endl;
        std::cout << pose.block<3, 3>(0, 0) << std::endl;
        std::cout << "\tt:" << std::endl;
        std::cout << pose.block<3, 1>(0, 3) << std::endl;
        std::cout << std::endl;
    }

    return false;
}


int main(int argc, char* argv[]){
    FundamentalMatrix F;
    F[0] = -0.0051918668202215884;
    F[1] = -0.015460923969578466;
    F[2] = 0.35260470328319654;
    F[3] = 0.022451443619913483;
    F[4] = -0.00079225386526248181;
    F[5] = -0.027885130552744289;
    F[6] = -0.35188558059920161;
    F[7] = 0.032418724757766811;
    F[8] = -0.005524537443406155;


    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    if(
        calc_cam_poses(F, f1, f2, R, t)
    ) {
        std::cout<<"Correct pose found!"<<std::endl;
        std::cout<<"R: "<<R<<std::endl;
        std::cout<<"t: "<<t<<std::endl;
    }

    std::cout<<"Result should be: \n";
    std::cout<<"R: \n"
             << "0.999827 -0.0119578 0.0142419\n"
             << "0.0122145 0.999762 -0.0180719\n"
             << "-0.0140224 0.0182427 0.999735\n";
    std::cout<<"t: \n"
             <<"0.0796625 0.99498 0.0605768\n";


    return 0;
}