//Created by sway on 2018/8/29.
   /* 测试8点法求取基础矩阵F
    *
    * [直接线性变换法]
    * 双目视觉中相机之间存在对极约束
    *
    *                       p2'Fp1=0,
    *
    * 其中p1, p2 为来自两个视角的匹配对的归一化坐标，并表示成齐次坐标形式，
    * 即p1=[x1, y1, z1]', p2=[x2, y2, z2],将p1, p2的表达形式带入到
    * 上式中，可以得到如下表达形式
    *
    *          [x2] [f11, f12, f13] [x1, y1, z1]
    *          [y2] [f21, f22, f23]                = 0
    *          [z2] [f31, f32, f33]
    *
    * 进一步可以得到
    * x1*x2*f11 + x2*y1*f12 + x2*f13 + x1*y2*f21 + y1*y2*f22 + y2*f23 + x1*f31 + y1*f32 + f33=0
    *
    * 写成向量形式
    *               [x1*x2, x2*y1,x2, x1*y2, y1*y2, y2, x1, y1, 1]*f = 0,
    * 其中f=[f11, f12, f13, f21, f22, f23, f31, f32, f33]'
    *
    * 由于F无法确定尺度(up to scale, 回想一下三维重建是无法确定场景真实尺度的)，因此F秩为8，
    * 这意味着至少需要8对匹配对才能求的f的解。当刚好有8对点时，称为8点法。当匹配对大于8时需要用最小二乘法进行求解
    *
    *   [x11*x12, x12*y11,x12, x11*y12, y11*y12, y12, x11, y11, 1]
    *   [x21*x22, x22*y21,x22, x21*y22, y21*y22, y22, x21, y21, 1]
    *   [x31*x32, x32*y31,x32, x31*y32, y31*y32, y32, x31, y31, 1]
    * A=[x41*x42, x42*y41,x42, x41*y42, y41*y42, y42, x41, y41, 1]
    *   [x51*x52, x52*y51,x52, x51*y52, y51*y52, y52, x51, y51, 1]
    *   [x61*x62, x62*y61,x62, x61*y62, y61*y62, y62, x61, y61, 1]
    *   [x71*x72, x72*y71,x72, x71*y72, y71*y72, y72, x71, y71, 1]
    *   [x81*x82, x82*y81,x82, x81*y22, y81*y82, y82, x81, y81, 1]
    *
    *现在任务变成了求解线性方程
    *               Af = 0
    *（该方程与min||Af||, subject to ||f||=1 等价)
    *通常的解法是对A进行SVD分解，取最小奇异值对应的奇异向量作为f分解
    *
    *本项目中对矩阵A的svd分解并获取其最小奇异值对应的奇异向量的代码为
    *   math::Matrix<double, 9, 9> V;
    *   math::matrix_svd<double, 8, 9>(A, nullptr, nullptr, &V);
    *   math::Vector<double, 9> f = V.col(8);
    *
    *
    *[奇异性约束]
    *  基础矩阵F的一个重要的性质是F是奇异的，秩为2，因此有一个奇异值为0。通过上述直接线性法求得
    *  矩阵不具有奇异性约束。常用的方法是将求得得矩阵投影到满足奇异约束得空间中。
    *  具体地，对F进行奇异值分解
    *               F = USV'
    *  其中S是对角矩阵，S=diag[sigma1, sigma2, sigma3]
    *  将sigma3设置为0，并重构F
    *                       [sigma1, 0,     ,0]
    *                 F = U [  0   , sigma2 ,0] V'
    *                       [  0   , 0      ,0]
    */

#include <math/matrix_svd.h>
#include "math/matrix.h"
#include "math/vector.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

typedef math::Matrix<double, 3, 3>  FundamentalMatrix;

constexpr int N = 8;

constexpr int U = 0;
constexpr int V = 1;

FundamentalMatrix fundamental_8_point (
    math::Matrix<double, 3, 8> const& points1, 
    math::Matrix<double, 3, 8> const& points2
){ 
    FundamentalMatrix F;

    //
    // a. build linear systems to get unconstrained F:
    //
    Eigen::MatrixXd A(N, 9);
    for (int n = 0; n < N; ++n) {
        A(n, 0) = points2(U, n) * points1(U, n);
        A(n, 1) = points2(U, n) * points1(V, n);
        A(n, 2) = points2(U, n);
        A(n, 3) = points2(V, n) * points1(U, n);
        A(n, 4) = points2(V, n) * points1(V, n);
        A(n, 5) = points2(V, n);     
        A(n, 6) = points1(U, n);
        A(n, 7) = points1(V, n);
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

    return F;
}

int main(int argc, char*argv[])
{

    // 第一幅图像中的对应点
    math::Matrix<double, 3, 8> pset1;
    pset1(0, 0) = 0.180123 ; pset1(1, 0)= -0.156584; pset1(2, 0)=1.0;
    pset1(0, 1) = 0.291429 ; pset1(1, 1)= 0.137662 ; pset1(2, 1)=1.0;
    pset1(0, 2) = -0.170373; pset1(1, 2)= 0.0779329; pset1(2, 2)=1.0;
    pset1(0, 3) = 0.235952 ; pset1(1, 3)= -0.164956; pset1(2, 3)=1.0;
    pset1(0, 4) = 0.142122 ; pset1(1, 4)= -0.216048; pset1(2, 4)=1.0;
    pset1(0, 5) = -0.463158; pset1(1, 5)= -0.132632; pset1(2, 5)=1.0;
    pset1(0, 6) = 0.0801864; pset1(1, 6)= 0.0236417; pset1(2, 6)=1.0;
    pset1(0, 7) = -0.179068; pset1(1, 7)= 0.0837119; pset1(2, 7)=1.0;
    //第二幅图像中的对应
    math::Matrix<double, 3, 8> pset2;
    pset2(0, 0) = 0.208264 ; pset2(1, 0)= -0.035405 ; pset2(2, 0) = 1.0;
    pset2(0, 1) = 0.314848 ; pset2(1, 1)=  0.267849 ; pset2(2, 1) = 1.0;
    pset2(0, 2) = -0.144499; pset2(1, 2)= 0.190208  ; pset2(2, 2) = 1.0;
    pset2(0, 3) = 0.264461 ; pset2(1, 3)= -0.0404422; pset2(2, 3) = 1.0;
    pset2(0, 4) = 0.171033 ; pset2(1, 4)= -0.0961747; pset2(2, 4) = 1.0;
    pset2(0, 5) = -0.427861; pset2(1, 5)= 0.00896567; pset2(2, 5) = 1.0;
    pset2(0, 6) = 0.105406 ; pset2(1, 6)= 0.140966  ; pset2(2, 6) = 1.0;
    pset2(0, 7) =  -0.15257; pset2(1, 7)= 0.19645   ; pset2(2, 7) = 1.0;

    FundamentalMatrix F = fundamental_8_point(pset1, pset2);


    std::cout<<"Fundamental matrix after singularity constraint is:\n "<<F<<std::endl;

    std::cout<<"Result should be: \n"<<"-0.0315082 -0.63238 0.16121\n"
                                     <<"0.653176 -0.0405703 0.21148\n"
                                     <<"-0.248026 -0.194965 -0.0234573\n" <<std::endl;

    return 0;
}