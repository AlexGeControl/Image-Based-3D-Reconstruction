/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <cmath>
#include <random>

#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <util/system.h>

#include <sfm/ransac_fundamental.h>
#include "math/functions.h"
#include "sfm/fundamental.h"
#include "sfm/correspondence.h"
#include "math/matrix_svd.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>


typedef math::Matrix<double, 3, 3> FundamentalMatrix;

constexpr int N = 8;

constexpr int U = 0;
constexpr int V = 1;

/**
 * \description 用于RANSAC采样成功所需要的采样次数
 * @param p -- 内点的概率
 * @param K --拟合模型需要的样本个数，对应基础矩阵num_samples=8
 * @param z  -- 预期的采样成功的概率
 *                          log(1-z)
 *       需要的采样次数 M = -----------
 *                          log(1-p^K)
 * Example: For p = 50%, z = 99%, n = 8: M = log(0.001) / log(0.99609) = 1176.
 * 需要采样1176次从而保证RANSAC的成功率不低于0.99.
 * @return
 */
int calc_ransac_iterations(
    double p,
    int K,
    double z = 0.99
){
    double prob_all_inliers = std::pow(p, K);

    int M = static_cast<int>(std::log(1.0 - z) / std::log(1.0 - prob_all_inliers)) + 1;

    return M;
}

/**
 * \description 给定基础矩阵和一对匹配点，计算匹配点的sampson 距离，用于判断匹配点是否是内点,
 * 计算公式如下：
 *              SD = (x'Fx)^2 / ( (Fx)_1^2 + (Fx)_2^2 + (x'F)_1^2 + (x'F)_2^2 )
 * @param F-- 基础矩阵
 * @param m-- 匹配对
 * @return
 */
double calc_sampson_distance(
    FundamentalMatrix const& F, 
    sfm::Correspondence2D2D const& m
) {

    double p2_F_p1 = 0.0;
    p2_F_p1 += m.p2[0] * (m.p1[0] * F[0] + m.p1[1] * F[1] + F[2]);
    p2_F_p1 += m.p2[1] * (m.p1[0] * F[3] + m.p1[1] * F[4] + F[5]);
    p2_F_p1 +=     1.0 * (m.p1[0] * F[6] + m.p1[1] * F[7] + F[8]);
    p2_F_p1 *= p2_F_p1;

    double sum = 1.0e-6;
    sum += math::fastpow(m.p1[0] * F[0] + m.p1[1] * F[1] + F[2], 2);
    sum += math::fastpow(m.p1[0] * F[3] + m.p1[1] * F[4] + F[5], 2);
    sum += math::fastpow(m.p2[0] * F[0] + m.p2[1] * F[3] + F[6], 2);
    sum += math::fastpow(m.p2[0] * F[1] + m.p2[1] * F[4] + F[7], 2);

    return p2_F_p1 / sum;
}
/**
 * \description 8点发估计相机基础矩阵
 * @param pset1 -- 第一个视角的特征点
 * @param pset2 -- 第二个视角的特征点
 * @return 估计的基础矩阵
 */
void calc_fundamental_8_point (
    math::Matrix<double, 3, 8> const& pset1, 
    math::Matrix<double, 3, 8> const& pset2,
    FundamentalMatrix &F
) {
    //
    // a. build linear systems to get unconstrained F:
    //
    Eigen::MatrixXd A(N, 9);
    for (int n = 0; n < N; ++n) {
        A(n, 0) = pset2(U, n) * pset1(U, n);
        A(n, 1) = pset2(U, n) * pset1(V, n);
        A(n, 2) = pset2(U, n);
        A(n, 3) = pset2(V, n) * pset1(U, n);
        A(n, 4) = pset2(V, n) * pset1(V, n);
        A(n, 5) = pset2(V, n);     
        A(n, 6) = pset1(U, n);
        A(n, 7) = pset1(V, n);
        A(n, 8) = 1.0;
    }

    // TODO: figure out why SAES would fail here
    Eigen::JacobiSVD<Eigen::MatrixXd> saes(A, Eigen::ComputeFullV);
    Eigen::MatrixXd F_unconstrained = Eigen::Matrix<
        double, 
        Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
    >::Map(
        saes.matrixV().col(8).data(),
        3, 3
    );

    //
    // b. apply singular value constraint on unconstrained F:
    //
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(F_unconstrained, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::VectorXd S = svd.singularValues();
    S(2) = 0.0;

    Eigen::MatrixXd F_constrained = svd.matrixU() * (S.asDiagonal()) * svd.matrixV().transpose();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            F(i, j) = F_constrained(i, j);
        }
    }
}

/**
 * \description 利用最小二乘法计算基础矩阵
 * @param matches--输入的匹配对 大于8对
 * @param F --基础矩阵
 */
void calc_fundamental_least_squares(
    sfm::Correspondences2D2D const &matches, 
    FundamentalMatrix& F
) {
    const int M = static_cast<int>(matches.size());

    if (M < 8)
        throw std::invalid_argument("At least 8 points required");

    //
    // a. build linear systems to get unconstrained F:
    //
    Eigen::MatrixXd A(M, 9);
    for (int m = 0; m < M; ++m) {
        const auto &p = matches[m];

        A(m, 0) = p.p2[U] * p.p1[U];
        A(m, 1) = p.p2[U] * p.p1[V];
        A(m, 2) = p.p2[U];
        A(m, 3) = p.p2[V] * p.p1[U];
        A(m, 4) = p.p2[V] * p.p1[V];
        A(m, 5) = p.p2[V];     
        A(m, 6) = p.p1[U];
        A(m, 7) = p.p1[V];
        A(m, 8) = 1.0;
    }


    // TODO: figure out why SAES would fail here
    Eigen::JacobiSVD<Eigen::MatrixXd> saes(A, Eigen::ComputeFullV);
    Eigen::MatrixXd F_unconstrained = Eigen::Matrix<
        double, 
        Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
    >::Map(
        saes.matrixV().col(8).data(),
        3, 3
    );

    //
    // b. apply singular value constraint on unconstrained F:
    //
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(F_unconstrained, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::VectorXd S = svd.singularValues();
    S(2) = 0.0;

    Eigen::MatrixXd F_constrained = svd.matrixU() * (S.asDiagonal()) * svd.matrixV().transpose();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            F(i, j) = F_constrained(i, j);
        }
    }
}
/**
 * \description 给定匹配对和基础矩阵，计算内点的个数
 * @param matches
 * @param F
 * @return
 */
std::vector<int> find_inliers(
    sfm::Correspondences2D2D const &matches, 
    const FundamentalMatrix &F, 
    const double &thresh
){
    const double squared_thresh = thresh* thresh;

    std::vector<int> inliers;

    const int N = static_cast<int>(matches.size());

    for (int n = 0; n < N; ++n) {
        const auto &m = matches.at(n);

        if (calc_sampson_distance(F, m) < squared_thresh) {
            inliers.push_back(n);
        }
    }

    return std::move(inliers);
}


int main(int argc, char *argv[]){
    /** 加载归一化后的匹配对 */
    sfm::Correspondences2D2D corr_all;
    std::ifstream in("./correspondences.txt");
    assert(in.is_open());

    std::string line, word;
    int n_line = 0;
    while(getline(in, line)){

        std::stringstream stream(line);
        if(n_line==0){
            int n_corrs = 0;
            stream>> n_corrs;
            corr_all.resize(n_corrs);

            n_line ++;
            continue;
        }
        if(n_line>0){

            stream>>corr_all[n_line-1].p1[0]>>corr_all[n_line-1].p1[1]
                  >>corr_all[n_line-1].p2[0]>>corr_all[n_line-1].p2[1];
        }
        n_line++;
    }

    /* calculate the min. num. of samples needed */
    const float inlier_ratio = 0.5;
    const int n_samples = 8;
    int n_iterations = calc_ransac_iterations(inlier_ratio, n_samples);

    // inlier matching threshold:
    const double inlier_thresh = 0.0015;

    // RANSAC best estimation:
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, corr_all.size() - 1);

    std::vector<int> best_inliers;

    std::cout << "RANSAC-F: Running for " << n_iterations
              << " iterations, threshold " << inlier_thresh
              << "..." << std::endl;
    for( int i=0; i < n_iterations; i++ ){
        // 1. generate correspondence proposal:
        std::set<int> indices;
        while( indices.size() < 8 ){
            int idx = distrib(gen);
            indices.insert(idx);
        }

        math::Matrix<double, 3, 8> pset1, pset2;
        std::set<int>::const_iterator iter = indices.cbegin();
        for(int j=0; j<8; j++, iter++){
            sfm::Correspondence2D2D const & match = corr_all[*iter];

            pset1(0, j) = match.p1[0];
            pset1(1, j) = match.p1[1];
            pset1(2, j) = 1.0;

            pset2(0, j) = match.p2[0];
            pset2(1, j) = match.p2[1];
            pset2(2, j) = 1.0;
        }

        // 2. solve fundamental matrix:
        FundamentalMatrix F;
        calc_fundamental_8_point(pset1, pset2, F);

        // 3. evaluate:
        std::vector<int> inlier_indices = find_inliers(corr_all, F, inlier_thresh);

        if( inlier_indices.size()> best_inliers.size() ){
            best_inliers.swap(inlier_indices);
        }
    }

    sfm::Correspondences2D2D corr_f;
    for(int i=0; i< best_inliers.size(); i++){
        corr_f.push_back(corr_all[best_inliers[i]]);
    }

    /*利用所有的内点进行最小二乘估计*/
    FundamentalMatrix F;
    calc_fundamental_least_squares(corr_f, F);

    std::cout<<"inlier number: "<< best_inliers.size()<<std::endl;
    std::cout<<"F\n: "<< F<<std::endl;

    std::cout<<"result should be: \n"
             <<"inliner number: 272\n"
             <<"F: \n"
             <<"-0.00961384 -0.0309071 0.703297\n"
             <<"0.0448265 -0.00158655 -0.0555796\n"
             <<"-0.703477 0.0648517 -0.0117791\n";

    return EXIT_SUCCESS;
}