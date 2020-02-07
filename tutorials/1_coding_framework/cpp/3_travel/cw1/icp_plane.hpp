#ifndef icp_plane
#define icp_plane

#include <stdio.h>
#include <ctime>
#include <math.h>
#include <random>
#include <iostream>
#include <igl/fit_plane.h>
#include <igl/per_vertex_normals.h>

#include "nanoflann.hpp"
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace nanoflann;


namespace icp {
	void findNormal(MatrixXd source, MatrixXd target, MatrixXd& normal);
	void icpPointToPlane(MatrixXd source, MatrixXd target, MatrixXd normal, pair<Matrix3d, Vector3d>& RT);
}




// Calculate normal for each vertices in source using KDTree
void icp::findNormal(MatrixXd source, MatrixXd target, MatrixXd& normal)
{
	// get centroid of target mesh to calculate normal direction
	Eigen::RowVector3d centroid(1, 3);
	centroid = target.colwise().sum() / double(target.rows());

	int pts_cnt = source.rows();
	int dimensionality = 3;

	normal.resize(pts_cnt, 3);


	KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree(dimensionality, target, 10);
	kd_tree.index->buildIndex();

	for (int i = 0; i < pts_cnt; i++) {

		Eigen::RowVector3d src_point = source.row(i);

		// closest 5 neighbors to form a plane
		const int neighbor_cnt = 5;
		vector<size_t> ret_index(neighbor_cnt);
		vector<double> dists(neighbor_cnt);

		//result set
		nanoflann::KNNResultSet<double> resultSet(neighbor_cnt);

		resultSet.init(&ret_index[0], &dists[0]);
		kd_tree.index->findNeighbors(resultSet, &src_point[0], nanoflann::SearchParams(10));

		// get closest neighbor points index in target
		Eigen::MatrixXd selectPts(neighbor_cnt, 3);
		for (size_t j = 0; j < neighbor_cnt; j++) {
			selectPts(j, 0) = target(ret_index[j], 0);
			selectPts(j, 1) = target(ret_index[j], 1);
			selectPts(j, 2) = target(ret_index[j], 2);
		}

		// compute plane normal and points on plane (Cvt)
		Eigen::RowVector3d curr_norm, Cvt;
		igl::fit_plane(selectPts, curr_norm, Cvt);


		// check the direction of normal vector
		// (M - Q).N > 0 invert the normal
		if ((centroid - target.row(i)).dot(curr_norm) > 0) {
			curr_norm *= -1;	//reverse direction
		}
		normal.row(i) = curr_norm;
	}
}


// Solve for rigid transformation (RT) for point-to-plane ICP
void icp::icpPointToPlane(MatrixXd source, MatrixXd target, MatrixXd normal, pair<Matrix3d, Vector3d>& RT)
{
	// build A and b
	int pts_cnt = source.rows();
	Eigen::MatrixXd A(pts_cnt, 6);
	Eigen::MatrixXd b(pts_cnt, 1);
	for (int i = 0; i < source.rows(); i++) {
		A(i, 0) = normal(i, 2)*target(i, 1) - normal(i, 1)*target(i, 2);
		A(i, 1) = normal(i, 0)*target(i, 2) - normal(i, 2)*target(i, 0);
		A(i, 2) = normal(i, 1)*target(i, 0) - normal(i, 0)*target(i, 1);
		A(i, 3) = normal(i, 0);
		A(i, 4) = normal(i, 1);
		A(i, 5) = normal(i, 2);

		Eigen::MatrixXd diff(pts_cnt, 3);
		diff = target - source;
		b(i) = 0;
		for (int n = 0; n < 3; n++) {
			b(i) = b(i) - (diff(i, n)*normal(i, n));
		}
	}

	// solve x
	Eigen::MatrixXd x = ((A.transpose()*A).inverse())*(A.transpose())*b;

	// rotation matrix
	Eigen::Matrix3d R;
	double sin_a = sin(x(0));
	double cos_a = cos(x(0));
	double sin_b = sin(x(1));
	double cos_b = cos(x(1));
	double sin_y = sin(x(2));
	double cos_y = cos(x(2));
	R(0, 0) = cos_y * cos_b;
	R(0, 1) = -sin_y * cos_a + cos_y * sin_b*sin_a;
	R(0, 2) = sin_y * sin_a + cos_y * sin_b*cos_a;
	R(1, 0) = sin_y * cos_b;
	R(1, 1) = cos_y * cos_a + sin_y * sin_b*sin_a;
	R(1, 2) = -cos_y * sin_a + sin_y * sin_b*cos_a;
	R(2, 0) = -sin_b;
	R(2, 1) = cos_b * sin_a;
	R(2, 2) = cos_b * cos_a;

	// translation vector
	Eigen::Vector3d T(x(3), x(4), x(5));

	RT = make_pair(R, T);
}


#endif








/*
 __  _ _____   ____  _ _____
|  \| | __\ `v' /  \| |_   _|
| | ' | _| `. .'| | ' | | |
|_|\__|_|   !_! |_|\__| |_|

*/
