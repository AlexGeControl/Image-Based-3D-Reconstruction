//
// Created by caoqi on 2018/8/31.
//

//3D:  1.36939, -1.17123, 7.04869
//obs: 0.180123 -0.156584

#include "sfm/bundle_adjustment.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <sophus/so3.hpp>

/*
 * This function computes the Jacobian entries for the given camera and
 * 3D point pair that leads to one observation.
 *
 * The camera block 'cam_x_ptr' and 'cam_y_ptr' is:
 * - ID 0: Derivative of focal length f
 * - ID 1-2: Derivative of distortion parameters k0, k1
 * - ID 3-5: Derivative of translation t0, t1, t2
 * - ID 6-8: Derivative of rotation w0, w1, w2
 *
 * The 3D point block 'point_x_ptr' and 'point_y_ptr' is:
 * - ID 0-2: Derivative in x, y, and z direction.
 *
 * The function that leads to the observation is given as follows:
 *
 *   u = f * D(x,y) * x  (image observation x coordinate)
 *   v = f * D(x,y) * y  (image observation y coordinate)
 *
 * with the following definitions:
 *
 *   xc = R0 * X + t0  (homogeneous projection)
 *   yc = R1 * X + t1  (homogeneous projection)
 *   zc = R2 * X + t2  (homogeneous projection)
 *   x = xc / zc  (central projection)
 *   y = yc / zc  (central projection)
 *   D(x, y) = 1 + k0 (x^2 + y^2) + k1 (x^2 + y^2)^2  (distortion)
 */

 /**
  * /description 给定一个相机参数和一个三维点坐标，求解雅各比矩阵，即公式中的df(theta)/dtheta
  * @param cam       相机参数
  * @param point     三维点坐标
  * @param J_cam     重投影残差相对于相机参数的偏导数, 
  *                  相机有9个参数： 
  *                     [0] 焦距f; 
  *                     [1-2] 径向畸变系数k1, k2; 
  *                     [3-5] 平移向量 t1, t2, t3
  *                     [6-8] 旋转矩阵（角轴向量）
  * @param J_point   重投影残差相对于三维点坐标的偏导数
  */
void jacobian(
    sfm::ba::Camera const& cam, sfm::ba::Point3D const& point,
    Eigen::MatrixXd &J_cam,
    Eigen::MatrixXd &J_point
) {
    // parse camera intrinsics:
    const double f = cam.focal_length;
    const double k0 = cam.distortion[0];
    const double k1 = cam.distortion[1];
    // parse camera extrinsics:
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(cam.rotation);
    Eigen::Map<const Eigen::Vector3d> t(cam.translation);
    Eigen::Map<const Eigen::Vector3d> P(point.pos);

    //
    // 1. calculate intermediate results:
    //
    // 1.a. to world frame:
    Eigen::Vector3d X = R*P + t;

    const double x = X.x();
    const double y = X.y();
    const double z = X.z();

    const double x_2 = x*x;
    const double y_2 = y*y;
    const double z_2 = z*z;
    const double z_3 = z*z_2;
    // 1.b. to normalized plane:
    Eigen::Vector2d p_normalized(
        x / z,
        y / z
    );
    // 1.c. distortion:
    const double r_2 = p_normalized.squaredNorm();
    const double r_4 = r_2*r_2;
    const double d = 1.0 + (k0 + k1*r_2)*r_2;
    // 1.d. Jacobian with respect to X:
    Eigen::MatrixXd J_X(2, 3);
    
    J_X(0, 0) = f*x*(2*k1*x*(x_2/z_2 + y_2/z_2)/z_2 + 2*x*(k0 + k1*(x_2/z_2 + y_2/z_2))/z_2)/z + f*((k0 + k1*(x_2/z_2 + y_2/z_2))*(x_2/z_2 + y_2/z_2) + 1)/z;
    J_X(0, 1) = f*x*(2*k1*y*(x_2/z_2 + y_2/z_2)/z_2 + 2*y*(k0 + k1*(x_2/z_2 + y_2/z_2))/z_2)/z;
    J_X(0, 2) = f*x*(k1*(-2*x_2/z_3 - 2*y_2/z_3)*(x_2/z_2 + y_2/z_2) + (k0 + k1*(x_2/z_2 + y_2/z_2))*(-2*x_2/z_3 - 2*y_2/z_3))/z - f*x*((k0 + k1*(x_2/z_2 + y_2/z_2))*(x_2/z_2 + y_2/z_2) + 1)/z_2;
    J_X(1, 0) = f*y*(2*k1*x*(x_2/z_2 + y_2/z_2)/z_2 + 2*x*(k0 + k1*(x_2/z_2 + y_2/z_2))/z_2)/z;
    J_X(1, 1) = f*y*(2*k1*y*(x_2/z_2 + y_2/z_2)/z_2 + 2*y*(k0 + k1*(x_2/z_2 + y_2/z_2))/z_2)/z + f*((k0 + k1*(x_2/z_2 + y_2/z_2))*(x_2/z_2 + y_2/z_2) + 1)/z;
    J_X(1, 2) = f*y*(k1*(-2*x_2/z_3 - 2*y_2/z_3)*(x_2/z_2 + y_2/z_2) + (k0 + k1*(x_2/z_2 + y_2/z_2))*(-2*x_2/z_3 - 2*y_2/z_3))/z - f*y*((k0 + k1*(x_2/z_2 + y_2/z_2))*(x_2/z_2 + y_2/z_2) + 1)/z_2;

    //
    // 2. Jacobian with respect to camera params:
    //
    // 2.1. f:
    J_cam.block<2, 1>(0, 0) = d*p_normalized;
    // 2.2. k0:
    J_cam.block<2, 1>(0, 1) = f*r_2*p_normalized;
    // 2.3. k1:
    J_cam.block<2, 1>(0, 2) = f*r_4*p_normalized;
    // 2.4. t:
    J_cam.block<2, 3>(0, 3) = J_X;
    // 2.5. R, left perturbation:
    J_cam.block<2, 3>(0, 6) = -J_X*Sophus::SO3d::hat(R*P);

    //
    // 3. Jacobian with respect to landmark position:
    //
    J_point = J_X*R;
}


int main(int argc, char*argv[])
{
    sfm::ba::Camera cam;
    cam.focal_length  =  0.919654;
    cam.distortion[0] = -0.108298;
    cam.distortion[1] =  0.103775;

    cam.rotation[0] = 0.999999;
    cam.rotation[1] = -0.000676196;
    cam.rotation[2] = -0.0013484;
    cam.rotation[3] = 0.000663243;
    cam.rotation[4] = 0.999949;
    cam.rotation[5] = -0.0104095;
    cam.rotation[6] = 0.00135482;
    cam.rotation[7] = 0.0104087;
    cam.rotation[8] = 0.999949;

    cam.translation[0]=0.00278292;
    cam.translation[1]=0.0587996;
    cam.translation[2]=-0.127624;

    sfm::ba::Point3D pt3D;
    pt3D.pos[0]= 1.36939;
    pt3D.pos[1]= -1.17123;
    pt3D.pos[2]= 7.04869;

    double cam_x_ptr[9]={0};
    double cam_y_ptr[9]={0};
    double point_x_ptr[3]={0};
    double point_y_ptr[3]={0};

    Eigen::MatrixXd J_cam(2, 9), J_point(2, 3);
    jacobian(
        cam, pt3D,
        J_cam, J_point
    );

    // 
    // display result:
    //
    std::cout<<"Result is :"<<std::endl;
    std::cout<<"cam_x_ptr: ";
    for(int i=0; i<9; i++){
        std::cout<<J_cam(0, i)<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"cam_y_ptr: ";
    for(int i=0; i<9; i++){
        std::cout<<J_cam(1, i)<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"point_x_ptr: ";
    std::cout<<J_point(0, 0)<<" "<<J_point(0, 1)<<" "<<J_point(0, 2)<<std::endl;

    std::cout<<"point_y_ptr: ";
    std::cout<<J_point(1, 0)<<" "<<J_point(1, 1)<<" "<<J_point(1, 2)<<std::endl;

    //
    // display ground truth:
    //
    std::cout<<"\nResult should be :\n"
       <<"cam_x_ptr: 0.195942 0.0123983 0.000847141 0.131188 0.000847456 -0.0257388 0.0260453 0.95832 0.164303\n"
       <<"cam_y_ptr: -0.170272 -0.010774 -0.000736159 0.000847456 0.131426 0.0223669 -0.952795 -0.0244697 0.179883\n"
       <<"point_x_ptr: 0.131153 0.000490796 -0.0259232\n"
       <<"point_y_ptr: 0.000964926 0.131652 0.0209965\n";

    return EXIT_SUCCESS;
}
