
// Created by sway on 2018/8/25.

/* 实现线性三角化方法(Linear triangulation methods), 给定匹配点
 * 以及相机投影矩阵(至少2对），计算对应的三维点坐标。给定相机内外参矩阵时，
 * 图像上每个点实际上对应三维中一条射线，理想情况下，利用两条射线相交便可以
 * 得到三维点的坐标。但是实际中，由于计算或者检测误差，无法保证两条射线的
 * 相交性，因此需要建立新的数学模型（如最小二乘）进行求解。
 *
 * 考虑两个视角的情况，假设空间中的三维点P的齐次坐标为X=[x,y,z,1]',对应地在
 * 两个视角的投影点分别为p1和p2，它们的图像坐标为
 *          x1=[x1, y1, 1]', x2=[x2, y2, 1]'.
 *
 * 两幅图像对应的相机投影矩阵为P1, P2 (P1,P2维度是3x4),理想情况下
 *             x1=P1X, x2=P2X
 *
 * 考虑第一个等式，在其两侧分别叉乘x1,可以得到
 *             x1 x (P1X) = 0
 *
 * 将P1X表示成[P11X, P21X, P31X]',其中P11，P21，P31分别是投影矩阵P1的第
 * 1～3行，我们可以得到
 *
 *          x1(P13X) - P11X     = 0
 *          y1(P13X) - P12X     = 0
 *          x1(P12X) - y1(P11X) = 0
 * 其中第三个方程可以由前两个通过线性变换得到，因此我们只考虑全两个方程。每一个
 * 视角可以提供两个约束，联合第二个视角的约束，我们可以得到
 *
 *                   AX = 0,
 * 其中
 *           [x1P13 - P11]
 *       A = [y1P13 - P12]
 *           [x2P23 - P21]
 *           [y2P23 - P22]
 *
 * 当视角个数多于2个的时候，可以采用最小二乘的方式进行求解，理论上，在不存在外点的
 * 情况下，视角越多估计的三维点坐标越准确。当存在外点(错误的匹配点）时，则通常采用
 * RANSAC的鲁棒估计方法进行求解。
 */


#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <iostream>


int main(int argc, char* argv[])
{
    // the corresponding projection points:
    Eigen::Vector2d p1(
         0.2899860,
        -0.0355493
    );
    Eigen::Vector2d p2(
        0.3161540,
        0.0898488
    );

    Eigen::MatrixXd P1(3, 4), P2(3, 4);

    P1 << 0.919653000, -0.000621866, -0.00124006,  0.00255933,
          0.000609954,  0.919607000, -0.00957316,  0.05407530,
          0.001354820,  0.010408700,  0.99994900, -0.12762400;

    P2 << 0.9200390, -0.01172140,  0.01442989,  0.0749395,
          0.0118301,  0.92012900, -0.00678373,  0.8627110,
         -0.0155846,  0.00757181,  0.99985400, -0.0887441;

    // build A matrix:
    Eigen::MatrixXd A(4, 4);

    A.block<1, 4>(0, 0) = P1.block<1, 4>(0, 0) - p1.x()*P1.block<1, 4>(2, 0);
    A.block<1, 4>(1, 0) = P1.block<1, 4>(1, 0) - p1.y()*P1.block<1, 4>(2, 0);
    A.block<1, 4>(2, 0) = P2.block<1, 4>(0, 0) - p2.x()*P2.block<1, 4>(2, 0);
    A.block<1, 4>(3, 0) = P2.block<1, 4>(1, 0) - p2.y()*P2.block<1, 4>(2, 0);

    // solve X:
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
    Eigen::VectorXd X_homogeneous = svd.matrixV().col(3);
    Eigen::Vector3d X(
        X_homogeneous(0) / X_homogeneous(3),
        X_homogeneous(1) / X_homogeneous(3),
        X_homogeneous(2) / X_homogeneous(3)
    );

    std::cout<<" trianglede point is :"<< X.x() <<" "<< X.y() <<" "<< X.z() <<std::endl;
    std::cout<<" the result should be "<<"2.14598 -0.250569 6.92321\n"<<std::endl;

    return EXIT_SUCCESS;
}
