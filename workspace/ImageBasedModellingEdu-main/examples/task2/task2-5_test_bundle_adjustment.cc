#include <cassert>
#include <fstream>
#include <vector>

#include <features/matching.h>
#include <sfm/ransac_fundamental.h>
#include <core/image_exif.h>

#include "math/matrix.h"
#include "math/vector.h"

#include "core/image_io.h"
#include "core/image.h"
#include "core/image_tools.h"

#include "sfm/camera_pose.h"
#include "sfm/fundamental.h"

#include "sfm/feature_set.h"
#include "sfm/correspondence.h"
#include "sfm/bundle_adjustment.h"
#include "sfm/correspondence.h"

#include "sfm/camera_database.h"
#include "sfm/extract_focal_length.h"

#include "sfm/triangulate.h"

#include "sfm/ba_conjugate_gradient.h"
#include "sfm/bundle_adjustment.h"
#include "sfm/ba_sparse_matrix.h"
#include "sfm/ba_dense_vector.h"
#include "sfm/ba_linear_solver.h"
#include "sfm/ba_sparse_matrix.h"
#include "sfm/ba_dense_vector.h"
#include "sfm/ba_cholesky.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <sophus/so3.hpp>

typedef sfm::ba::SparseMatrix<double> SparseMatrixType;
typedef sfm::ba::DenseVector<double> DenseVectorType;

//global variables
std::vector<sfm::ba::Camera> cameras;
std::vector<sfm::ba::Point3D> points;
std::vector<sfm::ba::Observation> observations;

#define TRUST_REGION_RADIUS_INIT (1000)
#define TRUST_REGION_RADIUS_DECREMENT (1.0 / 10.0)
#define TRUST_REGION_RADIUS_GAIN (10.0)

// lm 算法最多迭代次数
const int lm_max_iterations = 100;
// mean square error
double initial_mse = 0.0;
double final_mse = 0.0;
int num_lm_iterations = 0;
int num_lm_successful_iterations = 0;
int num_lm_unsuccessful_iterations = 0;

// lm 算法终止条件
double lm_mse_threshold = 1e-16;
double lm_delta_threshold = 1e-8;

// 信赖域大小
double trust_region_radius = 1000;
int cg_max_iterations =1000;
//相机参数个数
int camera_block_dim  = 9;

const int num_cam_params = 9;

#define MAX_PIXELS 1000000

void convert_sift_discriptors(features::Sift::Descriptors const&sift_descrs,
                              util::AlignedMemory<math::Vec128f, 16> *aligned_descr)
{
    aligned_descr->resize(sift_descrs.size());
    float * data_ptr=aligned_descr->data()->begin();
    for(int i=0; i<sift_descrs.size(); ++i, data_ptr+=128)
    {
        features::Sift::Descriptor const& descr = sift_descrs[i];
        std::copy(descr.data.begin(), descr.data.end(), data_ptr);
    }

}

float extract_focal_len(const std::string& img_name)
{
    std::string exif_str;
    core::image::load_jpg_file(img_name.c_str(), &exif_str);
    core::image::ExifInfo exif = core::image::exif_extract(exif_str.c_str(), exif_str.size(), false);
    sfm::FocalLengthEstimate fl = sfm::extract_focal_length(exif);
    std::cout <<"Focal length: " <<fl.first << " " << fl.second << std::endl;
    return fl.first;
}

sfm::Correspondences2D2D sift_feature_matching(
    sfm::FeatureSet &feat1, 
    sfm::FeatureSet&feat2
) {

    /* 1.0 特征匹配*/
    // 进行数据转换
    util::AlignedMemory<math::Vec128f, 16> aligned_descrs1, aligned_descrs2;
    convert_sift_discriptors(feat1.sift_descriptors, &aligned_descrs1);
    convert_sift_discriptors(feat2.sift_descriptors, &aligned_descrs2);

    // 特征匹配参数设置
    features::Matching::Options matching_opts;
    matching_opts.descriptor_length = 128;
    matching_opts.distance_threshold = 1.0f;
    matching_opts.lowe_ratio_threshold = 0.8f;

    // 特征匹配
    features::Matching::Result matching_result;
    features::Matching::twoway_match(matching_opts, aligned_descrs1.data()->begin()
            , feat1.sift_descriptors.size()
            ,aligned_descrs2.data()->begin()
            , feat2.sift_descriptors.size(),&matching_result);
    // 去除不一致的匹配对
    features::Matching::remove_inconsistent_matches(&matching_result);
    int n_consitent_matches = features::Matching::count_consistent_matches(matching_result);
    std::cout << "Consistent Sift Matches: "
              << n_consitent_matches
              << std::endl;

    /*2.0 利用本征矩阵对数据进行*/
    // 进行特征点坐标归一化，归一化之后坐标中心位于(0,0), 范围[-0.5, 0.5]。坐标归一化有助于
    // 保持计算的稳定性
    feat1.normalize_feature_positions();
    feat2.normalize_feature_positions();

    sfm::Correspondences2D2D corr_all;
    std::vector<int> const & m12 = matching_result.matches_1_2;
    for(int i=0; i<m12.size(); i++)
    {
        if(m12[i]<0)continue;

        sfm::Correspondence2D2D c2d2d;
        c2d2d.p1[0] = feat1.positions[i][0];
        c2d2d.p1[1] = feat1.positions[i][1];
        c2d2d.p2[0] = feat2.positions[m12[i]][0];
        c2d2d.p2[1] = feat2.positions[m12[i]][1];

        corr_all.push_back(c2d2d);
    }
    /* RANSAC 估计本征矩阵, 并对特征匹配对进行筛选*/
    sfm::RansacFundamental::Options ransac_fundamental_opts;
    ransac_fundamental_opts.max_iterations =1000;
    ransac_fundamental_opts.verbose_output = true;
    sfm::RansacFundamental ransac_fundamental(ransac_fundamental_opts);
    sfm::RansacFundamental::Result ransac_fundamental_result;
    ransac_fundamental.estimate(corr_all, &ransac_fundamental_result);
    // 根据估计的Fundamental矩阵对特征匹配对进行筛选
    sfm::Correspondences2D2D corr_f;
    for(int i=0; i<ransac_fundamental_result.inliers.size(); ++i)
    {
        int inlier_id = ransac_fundamental_result.inliers[i];
        corr_f.push_back(corr_all[inlier_id]);
    }

    std::cout<<"F: "<<ransac_fundamental_result.fundamental<<std::endl;
    return corr_f;
}

Eigen::Matrix3d GetFundamentalMatrix(
    const sfm::Correspondences2D2D &matches,
    Eigen::Matrix3d &F
) {
    //
    // solve fundamental matrix using least square estimator:
    //
    const int N = matches.size();

    // init:
    Eigen::MatrixXd A(N, 9);

    for (int i = 0; i < N; ++i) {
        // parse correspondence:
        const auto &match = matches.at(i);
        Eigen::Map<const Eigen::Vector2d> p1(match.p1);
        Eigen::Map<const Eigen::Vector2d> p2(match.p2);

        // represent F in column major order:
        A(i, 0) = p1.x()*p2.x();
        A(i, 1) = p1.x()*p2.y();
        A(i, 2) = p1.x();
        A(i, 3) = p1.y()*p2.x();
        A(i, 4) = p1.y()*p2.y();
        A(i, 5) = p1.y();
        A(i, 6) = p2.x();
        A(i, 7) = p2.y();
        A(i, 8) = 1.0;
    }

    // solve F, unconstrained:
    Eigen::JacobiSVD<Eigen::MatrixXd> solver_unconstrained(A, Eigen::ComputeFullV);
    Eigen::VectorXd F_unconstrained_column_major = solver_unconstrained.matrixV().col(8);

    // refine F, apply eigen value constraint:
    Eigen::Matrix3d F_constrained;

    F_constrained.block<3, 1>(0, 0) = F_unconstrained_column_major.block<3, 1>(0, 0);
    F_constrained.block<3, 1>(0, 1) = F_unconstrained_column_major.block<3, 1>(3, 0);
    F_constrained.block<3, 1>(0, 2) = F_unconstrained_column_major.block<3, 1>(6, 0);

    Eigen::JacobiSVD<Eigen::MatrixXd> solver_constrained(F_constrained, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Vector3d sigma = solver_constrained.singularValues();
    sigma(2) = 0.0;

    // finally:
    F = solver_constrained.matrixU()*sigma.asDiagonal()*solver_constrained.matrixV().transpose();
    F = 1.0 / F(2, 2) * F;
}

void GetCandidateCameraPoses(
    const Eigen::Matrix3d &F,
    const Eigen::Matrix3d &K1, const Eigen::Matrix3d &K2,
    std::vector<Eigen::Matrix4d> &camera_poses
) {
    //
    // build essential matrix:
    //
    Eigen::Matrix3d E = K2.transpose()*F*K1.transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> solver(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    //
    // generate candidate poses:
    //
    const Eigen::Matrix3d &U = solver.matrixU();
    const Eigen::Matrix3d &V_T = solver.matrixV().transpose();
    const Eigen::Matrix3d &R_Z = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    const Eigen::Matrix3d R1 = U*R_Z*V_T;
    const Eigen::Matrix3d R2 = U*R_Z.transpose()*V_T;
    const Eigen::Vector3d t1 =  U.col(2);
    const Eigen::Vector3d t2 = -U.col(2);

    Eigen::Matrix4d T;
    // candidate 01:
    T.block<3, 3>(0, 0) = R1; T.block<3, 1>(0, 3) = t1;
    camera_poses.push_back(T);
    // candidate 02:
    T.block<3, 3>(0, 0) = R1; T.block<3, 1>(0, 3) = t2;
    camera_poses.push_back(T);
    // candidate 03:
    T.block<3, 3>(0, 0) = R2; T.block<3, 1>(0, 3) = t1;
    camera_poses.push_back(T);
    // candidate 04:
    T.block<3, 3>(0, 0) = R2; T.block<3, 1>(0, 3) = t2;
    camera_poses.push_back(T);
}

Eigen::Vector3d Triangulate(
    const sfm::Correspondence2D2D &match,
    const Eigen::Matrix3d &K1, const Eigen::Matrix3d &K2,
    const Eigen::Matrix4d &camera_pose    
) {
    //
    // parse correspondence:
    //
    Eigen::Map<const Eigen::Vector2d> p1(match.p1);
    Eigen::Map<const Eigen::Vector2d> p2(match.p2);

    //
    // build projection matrix:
    //
    Eigen::MatrixXd T1 = Eigen::MatrixXd::Zero(3, 4);
    T1.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    T1 = K1*T1;

    Eigen::MatrixXd T2 = Eigen::MatrixXd::Zero(3, 4);
    T2 = camera_pose.block<3, 4>(0, 0);
    T2 = K2*T2;

    //
    // triangulation:
    //
    Eigen::Matrix4d A;

    A.block<1, 4>(0, 0) = T1.block<1, 4>(0, 0) - p1.x()*T1.block<1, 4>(2, 0);
    A.block<1, 4>(1, 0) = T1.block<1, 4>(1, 0) - p1.y()*T1.block<1, 4>(2, 0);
    A.block<1, 4>(2, 0) = T2.block<1, 4>(0, 0) - p2.x()*T2.block<1, 4>(2, 0);
    A.block<1, 4>(3, 0) = T2.block<1, 4>(1, 0) - p2.y()*T2.block<1, 4>(2, 0);

    Eigen::JacobiSVD<Eigen::Matrix4d> solver(A, Eigen::ComputeFullV);

    Eigen::VectorXd X_homo = solver.matrixV().col(3);

    Eigen::Vector3d X(
        X_homo(0) / X_homo(3),
        X_homo(1) / X_homo(3),
        X_homo(2) / X_homo(3)
    );

    return X;
}

bool IsValidCameraPose(
    const sfm::Correspondence2D2D &match,
    const Eigen::Matrix3d &K1, const Eigen::Matrix3d &K2,
    const Eigen::Matrix4d &camera_pose
) {
    Eigen::Vector3d X1 = Triangulate(match, K1, K2, camera_pose);
    Eigen::Vector3d X2 = camera_pose.block<3, 3>(0, 0)*X1 + camera_pose.block<3, 1>(0, 3);

    return X1.z() > 0.0 && X2.z() > 0.0;
}

bool CalcCamPose(
    const sfm::Correspondences2D2D &matches,
    const Eigen::Matrix3d &K1, const Eigen::Matrix3d &K2,
    Eigen::Matrix4d &camera_pose
) {    
    // estimate fundamental matrix:
    Eigen::Matrix3d F;
    GetFundamentalMatrix(matches, F);

    // get candidate poses:
    std::vector<Eigen::Matrix4d> Ts;
    GetCandidateCameraPoses(F, K1, K2, Ts);
    
    // indentify the only consistent pose:
    bool found_pose = false;
    for (const auto &T: Ts) {
        if (
            IsValidCameraPose(
                matches.at(0),
                K1, K2,
                T
            )
        ) {
            camera_pose = T;
            std::cout << "Estimated Pose" << std::endl;
            std::cout << T.block<3, 4>(0, 0) << std::endl;
            found_pose = true;
            break;
        }
    }

    return found_pose;
}

/**
 * /descrition 将角轴法转化成旋转矩阵
 * @param r 角轴向量
 * @param m 旋转矩阵
 */
void rodrigues_to_matrix (double const* r, double* m)
{
    /* Obtain angle from vector length. */
    double a = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    /* Precompute sine and cosine terms. */
    double ct = (a == 0.0) ? 0.5f : (1.0f - std::cos(a)) / (2.0 * a);
    double st = (a == 0.0) ? 1.0 : std::sin(a) / a;
    /* R = I + st * K + ct * K^2 (with cross product matrix K). */
    m[0] = 1.0 - (r[1] * r[1] + r[2] * r[2]) * ct;
    m[1] = r[0] * r[1] * ct - r[2] * st;
    m[2] = r[2] * r[0] * ct + r[1] * st;
    m[3] = r[0] * r[1] * ct + r[2] * st;
    m[4] = 1.0f - (r[2] * r[2] + r[0] * r[0]) * ct;
    m[5] = r[1] * r[2] * ct - r[0] * st;
    m[6] = r[2] * r[0] * ct - r[1] * st;
    m[7] = r[1] * r[2] * ct + r[0] * st;
    m[8] = 1.0 - (r[0] * r[0] + r[1] * r[1]) * ct;
}

/**
 * \description 根据求解得到的增量，对相机参数进行更新
 * @param cam
 * @param update
 * @param out
 */
void update_camera (sfm::ba::Camera const& cam,
                                 double const* update, sfm::ba::Camera* out)
{
    out->focal_length = cam.focal_length   + update[0];
    out->distortion[0] = cam.distortion[0] + update[1];
    out->distortion[1] = cam.distortion[1] + update[2];

    out->translation[0] = cam.translation[0] + update[3];
    out->translation[1] = cam.translation[1] + update[4];
    out->translation[2] = cam.translation[2] + update[5];

    double rot_orig[9];
    std::copy(cam.rotation, cam.rotation + 9, rot_orig);
    double rot_update[9];
    rodrigues_to_matrix(update + 6, rot_update);
    math::matrix_multiply(rot_update, 3, 3, rot_orig, 3, out->rotation);
}

/**
 * \description 根据求解的增量，对三维点坐标进行更新
 * @param pt
 * @param update
 * @param out
 */
void update_point (sfm::ba::Point3D const& pt,
                                double const* update, sfm::ba::Point3D* out)
{
    out->pos[0] = pt.pos[0] + update[0];
    out->pos[1] = pt.pos[1] + update[1];
    out->pos[2] = pt.pos[2] + update[2];
}

/**
 * /descripition 根据求得的delta_x, 更新相机参数和三维点
 * @param delta_x
 * @param cameras
 * @param points
 */
void
update_parameters (
    DenseVectorType const& delta_x, 
    std::vector<sfm::ba::Camera> &cameras, 
    std::vector<sfm::ba::Point3D> &points
) {
    /* Update cameras. */
    std::size_t total_camera_params = 0;
    for (std::size_t i = 0; i < cameras.size(); ++i){
        update_camera(
            cameras.at(i),
            delta_x.data() + num_cam_params * i,
            &cameras.at(i)
        );
        total_camera_params = cameras.size() * num_cam_params;
    }

    /* Update points. */
    for (std::size_t i = 0; i < points.size(); ++i) {
        update_point(
            points.at(i),
            delta_x.data() + total_camera_params + i * 3,
            &points.at(i)
        );
    }
}

/**
 * \description 对像素进行径向畸变
 * @param p_normalized
 * @param y
 * @param dist
 */
void radial_distort(
    Eigen::Vector2d &p_normalized, 
    const double* dist
) {
    const double r_squared = p_normalized.squaredNorm();
    const double d = 1.0 + r_squared * (dist[0] + dist[1] * r_squared);

    p_normalized = d * p_normalized;
}

/**
 * \description 计算重投影误差
 * @param vector_f
 * @param delta_x
 * @param cameras
 * @param points
 * @param observations
 */
void compute_reprojection_errors (
    DenseVectorType* vector_f, 
    DenseVectorType const* delta_x, 
    std::vector<sfm::ba::Camera> &cameras, 
    std::vector<sfm::ba::Point3D> &points,
    std::vector<sfm::ba::Observation> &observations
) {
    if (vector_f->size() != observations.size() * 2)
        vector_f->resize(observations.size() * 2);

#pragma omp parallel for
    for (std::size_t i = 0; i < observations.size(); ++i)
    {
        sfm::ba::Observation const& obs = observations.at(i);
        sfm::ba::Point3D const& p3d = points.at(obs.point_id);
        sfm::ba::Camera const& cam = cameras.at(obs.camera_id);

        const double* flen = &cam.focal_length; // 相机焦距
        const double* dist = cam.distortion;    // 径向畸变系数
        const double* rot = cam.rotation;       // 相机旋转矩阵
        const double* trans = cam.translation;  // 相机平移向量
        const double* point = p3d.pos;          // 三维点坐标

        sfm::ba::Point3D new_point;
        sfm::ba::Camera new_camera;

        // 如果delta_x 不为空，则先利用delta_x对相机和结构进行更新，然后再计算重投影误差
        if (delta_x != nullptr)
        {
            std::size_t cam_id = obs.camera_id * num_cam_params;
            std::size_t pt_id = obs.point_id * 3;

            update_camera(cam, delta_x->data() + cam_id, &new_camera);
            flen = &new_camera.focal_length;
            dist = new_camera.distortion;
            rot = new_camera.rotation;
            trans = new_camera.translation;
            pt_id += cameras.size() * num_cam_params;

            update_point(p3d, delta_x->data() + pt_id, &new_point);
            point = new_point.pos;
        }

        /* project landmark onto normalized plane */
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(rot);
        Eigen::Map<const Eigen::Vector3d> t(trans);
        Eigen::Map<const Eigen::Vector3d> P(point);

        Eigen::Vector3d X = R*P + t;
        Eigen::Vector2d p_normalized(
            X.x() / X.z(),
            X.y() / X.z()
        );

        /* Distort reprojections. */
        radial_distort(p_normalized, dist);

        /* Compute reprojection error. */
        Eigen::Map<const Eigen::Vector2d> p_observed(obs.pos);
        Eigen::Vector2d error = (*flen)*p_normalized - p_observed;

        vector_f->at(i * 2 + 0) = error.x();
        vector_f->at(i * 2 + 1) = error.y();
    }
}

/**
 * \description 计算均方误差
 * @param vector_f
 * @return
 */
double compute_mse (DenseVectorType const& vector_f) {
    double mse = 0.0;
    for (std::size_t i = 0; i < vector_f.size(); ++i)
        mse += vector_f[i] * vector_f[i];
    return mse / static_cast<double>(vector_f.size() / 2);
}

/**
 * get reprojection error Jacobians with respect to given camera params and given landmark position
 * @param cam
 * @param point
 * @param J_cam
 * @param J_point
 */
void GetReprojErrorJacobians(
    sfm::ba::Camera const& cam, sfm::ba::Point3D const& point,
    Eigen::MatrixXd &J_cam, Eigen::MatrixXd &J_point
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

/**
 * get bundle adjustment error Jacobians with respect to all camera params and all landmark positions
 * @param cam
 * @param point
 * @param J_cams
 * @param J_points
 */
void GetBAErrorJacobians(
    Eigen::SparseMatrix<double> &J_cams,
    Eigen::SparseMatrix<double> &J_points
) {
    //
    // determine BA problem size:
    //
    const int C = static_cast<int>(cameras.size());
    const int P = static_cast<int>(points.size());
    const int N = static_cast<int>(observations.size());

    // 
    // coeffs container for J_cams & J_points:
    // 
    std::vector<Eigen::Triplet<double>> J_cams_coeffs;
    std::vector<Eigen::Triplet<double>> J_points_coeffs;

    //
    // get Jacobians for bundle adjustment error:
    //
    Eigen::MatrixXd J_cam(2, 9), J_point(2, 3);
    for (int i = 0; i < N; ++i) {
        //
        // parse observation:
        //
        const sfm::ba::Observation &obs = observations.at(i);
        const sfm::ba::Camera &cam = cameras.at(obs.camera_id);
        const sfm::ba::Point3D &point = points.at(obs.point_id);

        //
        // calculate Jacobians for current observation's reprojection error:
        //
        GetReprojErrorJacobians(
            cam, point,
            J_cam, J_point
        );

        //
        // set coeffs for Jacobians:
        //
        for (int j = 0; j < 2; ++j) {
            int J_row_index = 2*i + j;

            // 1. camera params:
            for (int k = 0; k < 9; ++k) {
                int J_cams_col_index = 9*obs.camera_id + k;
                J_cams_coeffs.emplace_back(
                    J_row_index, J_cams_col_index, 
                    J_cam(j, k)
                );
            }

            // 2. landmark positions:
            for (int l = 0; l < 3; ++l) {
                int J_points_col_index = 3*obs.point_id + l;
                J_points_coeffs.emplace_back(
                    J_row_index, J_points_col_index, 
                    J_point(j, l)
                );
            }
        }
    }

    //
    // finally:
    //
    J_cams.resize(2*N, 9*C);
    J_cams.setFromTriplets(
        J_cams_coeffs.begin(), J_cams_coeffs.end()
    );

    J_points.resize(2*N, 3*P);
    J_points.setFromTriplets(
        J_points_coeffs.begin(), J_points_coeffs.end()
    );
}

sfm::ba::LinearSolver::Status SolveSchur(
    Eigen::SparseMatrix<double> &J_cams,
    Eigen::SparseMatrix<double> &J_points,
    DenseVectorType const& values,
    DenseVectorType* delta_x
) {
    //
    // parse residuals:
    //
    Eigen::SparseVector<double> e(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        e.coeffRef(i) = -values.at(i);
    }

    //
    // build system:
    // 
    Eigen::SparseMatrix<double> J_cc = J_cams.transpose()*J_cams;
    Eigen::SparseMatrix<double> J_cp = J_cams.transpose()*J_points;
    Eigen::SparseMatrix<double> J_pc = J_cp.transpose();
    Eigen::SparseMatrix<double> J_pp = J_points.transpose()*J_points;

    Eigen::SparseVector<double> b_c = J_cams.transpose()*e;
    Eigen::SparseVector<double> b_p = J_points.transpose()*e;

    //
    // add regulation:
    //
    const double regulation_strength = 1.0 + 1.0 / trust_region_radius;

    const int C = static_cast<int>(cameras.size());
    for (int i = 0; i < 9*C; ++i) { J_cc.coeffRef(i, i) *= regulation_strength; }

    const int P = static_cast<int>(points.size());
    for (int i = 0; i < 3*P; ++i) { J_pp.coeffRef(i, i) *= regulation_strength; }

    //
    // J_pp's inverse:
    //
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> J_pp_inv_solver; 
    J_pp_inv_solver.compute(J_pp); 

    Eigen::SparseMatrix<double> I(3*P, 3*P);
    I.setIdentity();

    Eigen::SparseMatrix<double> J_pp_inv = J_pp_inv_solver.solve(I);

    //
    // solve update for camera params:
    //
    Eigen::SparseMatrix<double> A_cam = J_cc - J_cp*J_pp_inv*J_pc;
    Eigen::SparseVector<double> b_cam = b_c - J_cp*J_pp_inv*b_p;

    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> cg_cam;
    cg_cam.compute(A_cam);
    Eigen::VectorXd delta_x_cam = cg_cam.solve(b_cam);

    //
    // solve update for landmark positions:
    //
    Eigen::SparseVector<double> b_point = b_p - J_pc*delta_x_cam;
    Eigen::VectorXd delta_x_point = J_pp_inv * b_point;

    // set output status:
    sfm::ba::LinearSolver::Status status;
    status.num_cg_iterations = 1;
    status.success = (
        J_pp_inv_solver.info() == Eigen::Success && 
        cg_cam.info() == Eigen::Success
    );

    // set output:
    const int N = 9*C + 3*P;
    if (N != static_cast<int>(delta_x->size()) )
        delta_x->resize(N, 0.0);

    for (int i = 0; i < 9*C; ++i)
        delta_x->at(i) = delta_x_cam(i);
    for (int i = 0; i < 3*P; ++i)
        delta_x->at(9*C + i) = delta_x_point(i);

    return status;
}

/**
 * /description  LM 算法流程
 * @param cameras
 * @param points
 * @param observations
 */
void lm_optimization(
    std::vector<sfm::ba::Camera> &cameras,
    std::vector<sfm::ba::Point3D> &points,
    std::vector<sfm::ba::Observation> &observations
) {
    //
    // init:
    // 
    // 计算重投影误差向量
    DenseVectorType F, F_new;
    compute_reprojection_errors(&F, nullptr, cameras, points, observations);// todo F 是误差向量
    // 计算初始的均方误差
    double current_mse = compute_mse(F);
    initial_mse = current_mse;
    final_mse = current_mse;

    // 设置共轭梯度法的相关参数
    trust_region_radius = TRUST_REGION_RADIUS_INIT;

    /* Levenberg-Marquard 算法. */
    for (int lm_iter = 0; ; ++lm_iter) {
        // 判断终止条件，均方误差小于一定阈值
        if (current_mse < lm_mse_threshold) {
            std::cout << "BA: Satisfied MSE threshold." << std::endl;
            break;
        }

        // 1. get Jacobians:
        // SparseMatrixType Jc, Jp;
        // analytic_jacobian(&Jc, &Jp);

        Eigen::SparseMatrix<double> J_cams, J_points;
        GetBAErrorJacobians(J_cams, J_points);

        //2.0 预置共轭梯梯度法对正规方程进行求解*/
        DenseVectorType delta_x;
        
        // sfm::ba::LinearSolver::Status cg_status = my_solve_schur(Jc, Jp, F, &delta_x);
        sfm::ba::LinearSolver::Status cg_status = SolveSchur(J_cams, J_points, F, &delta_x);

        //3.0 根据计算得到的偏移量，重新计算冲投影误差和均方误差，用于判断终止条件和更新条件.
        double new_mse, delta_mse, delta_mse_ratio = 1.0;

        // 正规方程求解成功的情况下
        if (cg_status.success) {
            /*重新计算相机和三维点，计算重投影误差，注意原始的相机参数没有被更新*/
            compute_reprojection_errors(&F_new, &delta_x, cameras, points, observations);
            /* 计算新的残差值 */
            new_mse = compute_mse(F_new);
            /* 均方误差的绝对变化值和相对变化率*/
            delta_mse = current_mse - new_mse;
            delta_mse_ratio = 1.0 - new_mse / current_mse;
        }
        // 正规方程求解失败的情况下
        else {
            new_mse = current_mse;
            delta_mse = 0.0;
        }

        // new_mse < current_mse表示残差值减少
        bool successful_iteration = delta_mse > 0.0;

        /*
         * 如果正规方程求解成功，则更新相机参数和三维点坐标，并且增大信赖域的尺寸，使得求解方式
         * 趋近于高斯牛顿法
         */
        if (successful_iteration) {
            std::cout << "BA: #" << std::setw(2) << std::left << lm_iter
                  << " success" << std::right
                  << ", MSE " << std::setw(11) << current_mse
                  << " -> " << std::setw(11) << new_mse
                  << ", CG " << std::setw(3) << cg_status.num_cg_iterations
                  << ", TRR " << trust_region_radius
                  << ", MSE Ratio: "<<delta_mse_ratio
                  << std::endl;

            num_lm_iterations += 1;
            num_lm_successful_iterations += 1;

            /* 对相机参数和三点坐标进行更新 */
            update_parameters(delta_x, cameras, points);

            std::swap(F, F_new);
            current_mse = new_mse;

            if (delta_mse_ratio < lm_delta_threshold) {
                std::cout << "BA: Satisfied delta mse ratio threshold of "
                          << lm_delta_threshold << std::endl;
                break;
            }

            // 增大信赖域大小
            trust_region_radius *= TRUST_REGION_RADIUS_GAIN;
        }
        else {
            std::cout << "BA: #" << std::setw(2) << std::left << lm_iter
                  << " failure" << std::right
                  << ", MSE " << std::setw(11) << current_mse
                  << ",    " << std::setw(11) << " "
                  << " CG " << std::setw(3) << cg_status.num_cg_iterations
                  << ", TRR " << trust_region_radius
                  << std::endl;

            num_lm_iterations += 1;
            num_lm_unsuccessful_iterations += 1;
            // 求解失败的减小信赖域尺寸
            trust_region_radius *= TRUST_REGION_RADIUS_DECREMENT;
        }

        /* 判断是否超过最大的迭代次数. */
        if (lm_iter + 1 >= lm_max_iterations) {
            std::cout << "BA: Reached maximum LM iterations of "
                  << lm_max_iterations << std::endl;
            break;
        }
    }

    final_mse = current_mse;
}

int main(int argc, char *argv[]) {
    if (argc != 3)
    {
        std::cerr << "Syntax: " << argv[0] << " <img1> <img2>" << std::endl;
        return 1;
    }

    //
    // 0. load images:
    // 
    std::string fname1(argv[1]);
    std::string fname2(argv[2]);
    std::cout << "Loading " << fname1 << "..." << std::endl;
    core::ByteImage::Ptr img1 = core::image::load_file(fname1);
    std::cout << "Loading " << fname2 << "..." << std::endl;
    core::ByteImage::Ptr img2 = core::image::load_file(fname2);
    
    // normalize image size:
    while(img1->get_pixel_amount() > MAX_PIXELS){
        img1=core::image::rescale_half_size<uint8_t>(img1);
    }
    while(img2->get_pixel_amount() > MAX_PIXELS){
        img2=core::image::rescale_half_size<uint8_t>(img2);
    }

    // extract focal lengths from EXIF header:
    float f1 = extract_focal_len(argv[1]);
    float f2 = extract_focal_len(argv[2]);

    Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
    K1(0, 0) = K1(1, 1) = f1;
    Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
    K2(0, 0) = K2(1, 1) = f2;

    std::cout<<"focal length: f1 "<<f1<<" f2: "<<f2<<std::endl;

    //
    // 1. SIFT detection and matching:
    //
    sfm::FeatureSet::Options feature_set_opts;
    feature_set_opts.feature_types = sfm::FeatureSet::FEATURE_SIFT;
    feature_set_opts.sift_opts.verbose_output = true;

    // detection & description:
    sfm::FeatureSet feat1(feature_set_opts);
    feat1.compute_features(img1);
    sfm::FeatureSet feat2(feature_set_opts);
    feat2.compute_features(img2);

    std::cout << "Image 1 (" << img1->width() << "x" << img1->height() << ") "
              << feat1.sift_descriptors.size() << " descriptors." << std::endl;
    std::cout << "Image 2 (" << img2->width() << "x" << img2->height() << ") "
              << feat2.sift_descriptors.size() << " descriptors." << std::endl;

    // matching:
    sfm::Correspondences2D2D corrs = sift_feature_matching(feat1, feat2);
    std::cout<<"Number of Matching pairs is "<< corrs.size()<<std::endl;

    std::cout<<"correspondence: "<<std::endl;
    for(int i=0; i< 3; i++) {
        std::cout << corrs[i].p1[0] << ", " <<corrs[i].p1[1] << " <--> "
                  << corrs[i].p2[0] << ", " <<corrs[i].p2[1] << std::endl;
    }

    if( corrs.size()<8 ){
        std::cerr<< "Number of matching pairs should not be less than 8." <<std::endl;
        return EXIT_FAILURE;
    }

    //
    // 2. relative pose estimation, coarse estimation:
    //
    Eigen::Matrix4d camera_pose = Eigen::Matrix4d::Identity();
    if( !CalcCamPose(corrs, K1, K2, camera_pose) )
    {
        std::cerr<<"Error to find corrent camera poses!"<<std::endl;
        return EXIT_FAILURE;
    }
    
    //
    // 3. landmark position estimation, coarse estimation:
    //
    std::vector<Eigen::Vector3d> landmarks;
    for (int i=0; i < corrs.size(); i++) {
        Eigen::Vector3d X = Triangulate(corrs.at(i), K1, K2, camera_pose);

        if (
            MATH_ISNAN(X.x()) || MATH_ISINF(X.x()) ||
            MATH_ISNAN(X.y()) || MATH_ISINF(X.y()) ||
            MATH_ISNAN(X.z()) || MATH_ISINF(X.z())
        ) {
            continue;
        }

        landmarks.push_back(X);
    }
    std::cout << "Successful triangulation:  "<< landmarks.size() << " points" << std::endl;

    //
    // 4. bundle adjustment for refined estimation:
    // 
    // 4.a. cameras:
    cameras.resize(2);

    cameras[0].focal_length = f1;

    cameras[0].distortion[0] = cameras[0].distortion[1] = 0.0;

    cameras[0].rotation[0] = 1.0; cameras[0].rotation[1] = 0.0; cameras[0].rotation[2] = 0.0;
    cameras[0].rotation[3] = 0.0; cameras[0].rotation[4] = 1.0; cameras[0].rotation[5] = 0.0;
    cameras[0].rotation[6] = 0.0; cameras[0].rotation[7] = 0.0; cameras[0].rotation[8] = 1.0;

    cameras[0].translation[0] = 0.0; cameras[0].translation[1] = 0.0; cameras[0].translation[2] = 0.0;

    cameras[0].is_constant = true;

    cameras[1].focal_length = f2;

    cameras[1].distortion[0] = cameras[1].distortion[1] = 0.0;

    cameras[1].rotation[0] = camera_pose(0, 0); cameras[1].rotation[1] = camera_pose(0, 1); cameras[1].rotation[2] = camera_pose(0, 2);
    cameras[1].rotation[3] = camera_pose(1, 0); cameras[1].rotation[4] = camera_pose(1, 1); cameras[1].rotation[5] = camera_pose(1, 2);
    cameras[1].rotation[6] = camera_pose(2, 0); cameras[1].rotation[7] = camera_pose(2, 1); cameras[1].rotation[8] = camera_pose(2, 2);

    cameras[1].translation[0] = camera_pose(0, 3); cameras[1].translation[1] = camera_pose(1, 3); cameras[1].translation[2] = camera_pose(2, 3);

    // 4.b. points:
    for(int i=0; i < landmarks.size(); i++)
    {
        sfm::ba::Point3D point;
        point.pos[0] = landmarks.at(i).x();
        point.pos[1] = landmarks.at(i).y();
        point.pos[2] = landmarks.at(i).z();
        points.push_back(point);
    }

    assert( points.size() == corrs.size() );

    // 4.c. observations:
    for (int i = 0; i < corrs.size(); ++i)
    {
        sfm::ba::Observation observation_1;
        observation_1.camera_id = 0;
        observation_1.point_id = i;
        std::copy(corrs[i].p1, corrs[i].p1+2, observation_1.pos);


        sfm::ba::Observation observation_2;
        observation_2.camera_id = 1;
        observation_2.point_id = i;
        std::copy(corrs[i].p2, corrs[i].p2+2, observation_2.pos);

        observations.push_back(observation_1);
        observations.push_back(observation_2);
    }

    // do optimization:
    lm_optimization(
        cameras, points, observations
    );

    // 将优化后的结果重新赋值
    std::vector<sfm::CameraPose> new_cam_poses(2);
    std::vector<math::Vec2f> radial_distortion(2);
    std::vector<math::Vec3f> new_pts_3d(points.size());
    for(int i=0; i<cameras.size(); i++) {
        std::copy(cameras[i].translation, cameras[i].translation + 3, new_cam_poses[i].t.begin());
        std::copy(cameras[i].rotation, cameras[i].rotation + 9, new_cam_poses[i].R.begin());
        radial_distortion[i]=math::Vec2f(cameras[i].distortion[0], cameras[i].distortion[1]);
        new_cam_poses[i].set_k_matrix(cameras[i].focal_length, 0.0, 0.0);
    }
    for(int i=0; i<new_pts_3d.size(); i++) {
        std::copy(points[i].pos, points[i].pos+3, new_pts_3d[i].begin());
    }

    // 输出优化信息
    std::cout << "Params after BA: " << std::endl;
    std::cout << "  f: " << new_cam_poses[0].get_focal_length()<<std::endl;
    std::cout << "  distortion: " << radial_distortion[0][0]<<", "<<radial_distortion[0][1]<<std::endl;
    std::cout << "  R: " << std::endl 
              << new_cam_poses[0].R<<std::endl;
    std::cout << "  t: " << std::endl
              << new_cam_poses[0] .t<<std::endl;

    // 输出优化信息
    std::cout << "Params after BA: " << std::endl;
    std::cout << "  f: " << new_cam_poses[1].get_focal_length()<<std::endl;
    std::cout << "  distortion: " << radial_distortion[1][0]<<", "<<radial_distortion[1][1]<<std::endl;
    std::cout << "  R: " << std::endl 
              << new_cam_poses[1].R<<std::endl;
    std::cout << "  t: " << std::endl 
              << new_cam_poses[1] .t<<std::endl;

    std::cout<<"points 3d: "<<std::endl;
    for(int i=0; i< 2; i++) {
        std::cout<<points[i].pos[0]<<", "<<points[i].pos[1]<<", "<<points[i].pos[2]<<std::endl;
    }

    return EXIT_SUCCESS;
}
