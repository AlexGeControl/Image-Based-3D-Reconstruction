//
// Created by caoqi on 2018/9/5.
//
#include <iostream>
#include "math/vector.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>


class Camera{

public:

    // constructor
    Camera(){
        // 采用归一化坐标，不考虑图像尺寸
        c_[0]=c_[1] = 0.0;
    }

    // 相机投影过程
    Eigen::Vector2d projection(const Eigen::Vector3d &P){
        //
        // a. project to normalized plane:
        // 
        Eigen::Vector3d P_normalized;

        Eigen::Matrix3d R = Eigen::Matrix<
            double, 
            Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
        >::Map(
            R_,
            3, 3
        );
        Eigen::Vector3d t = Eigen::Matrix<
            double, 
            Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
        >::Map(
            t_,
            3, 1
        );

        P_normalized = R*P + t;

        // b. calculate distortion factor:
        Eigen::Vector2d p_normalized = Eigen::Vector2d(
            P_normalized.x() / P_normalized.z(), 
            P_normalized.y() / P_normalized.z()
        );

        double r_2 = p_normalized.squaredNorm();
        double r_4 = r_2 * r_2;
        double distortion_factor = 1.0 + dist_[0] * r_2 + dist_[1] * r_4;

        // c. project to image plane:
        Eigen::Vector2d p;
        Eigen::Vector2d c = Eigen::Matrix<
            double, 
            Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
        >::Map(
            c_,
            2, 1
        );

        p = f_ * distortion_factor * p_normalized + c;

        return p;
    }

    // 相机在世界坐标中的位置 -R^T*t
    math::Vec3d pos_in_world(){
        math::Vec3d pos;
        pos[0] = R_[0]* t_[0] + R_[3]* t_[1] + R_[6]* t_[2];
        pos[1] = R_[1]* t_[0] + R_[4]* t_[1] + R_[7]* t_[2];
        pos[2] = R_[2]* t_[0] + R_[5]* t_[1] + R_[8]* t_[2];
        return -pos;
    }

    // 相机在世界坐标中的方向
    math::Vec3d dir_in_world(){

        math::Vec3d  dir (R_[6], R_[7],R_[8]);
        return dir;
    }
public:

    // 焦距f
    double f_;

    // 径向畸变系数k1, k2
    double dist_[2];

    // 中心点坐标u0, v0
    double c_[2];

    // 旋转矩阵
    /*
     * [ R_[0], R_[1], R_[2] ]
     * [ R_[3], R_[4], R_[5] ]
     * [ R_[6], R_[7], R_[8] ]
     */
    double R_[9];

    // 平移向量
    double t_[3];
};

int main(int argc, char* argv[]){


    Camera cam;

    //焦距
    cam.f_ = 0.920227;

    // 径向畸变系数
    cam.dist_[0] = -0.106599; cam.dist_[1] = 0.104385;

    // 平移向量
    cam.t_[0] = 0.0814358; cam.t_[1] =  0.937498;   cam.t_[2] = -0.0887441;

    // 旋转矩阵
    cam.R_[0] = 0.999796 ; cam.R_[1] = -0.0127375;  cam.R_[2] =  0.0156807;
    cam.R_[3] = 0.0128557; cam.R_[4] =  0.999894 ;  cam.R_[5] = -0.0073718;
    cam.R_[6] = -0.0155846; cam.R_[7] = 0.00757181; cam.R_[8] = 0.999854;

    // 三维点坐标
    Eigen::Vector3d p3d(1.36939, -1.17123, 7.04869);

    /*计算相机的投影点*/
    Eigen::Vector2d p2d = cam.projection(p3d);
    std::cout<<"projection coord:\n "<<p2d<<std::endl;
    std::cout<<"result should be:\n 0.208188 -0.035398\n\n";

    /*计算相机在世界坐标系中的位置*/
    math::Vec3d pos = cam.pos_in_world();
    std::cout<<"cam position in world is:\n "<< pos<<std::endl;
    std::cout<<"result should be: \n -0.0948544 -0.935689 0.0943652\n\n";

    /*计算相机在世界坐标系中的方向*/
    math::Vec3d dir = cam.dir_in_world();
    std::cout<<"cam direction in world is:\n "<<dir<<std::endl;
    std::cout<<"result should be: \n -0.0155846 0.00757181 0.999854\n";
}