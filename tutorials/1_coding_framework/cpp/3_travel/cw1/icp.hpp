
#ifndef icp_hpp
#define icp_hpp

#include <igl/fit_plane.h>
#include <igl/openGL/glfw/Viewer.h>
#include <igl/per_vertex_normals.h>

#include "nanoflann.hpp"

#include <math.h>
#include <random>
#include <stdio.h>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cstdlib>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace nanoflann;

namespace icp {
	MatrixXd getCorrespondingPoints(MatrixXd source, MatrixXd target);
	pair<Matrix3d, Vector3d> solveForRT(MatrixXd source, MatrixXd target);
	bool detectError(MatrixXd source, MatrixXd target, pair<Matrix3d, Vector3d> RT);
	MatrixXd getSampleSource(MatrixXd source, int sample_gap);
	MatrixXd addNoise(MatrixXd source, double noiseDiv);
	MatrixXd getNormal(MatrixXd source, MatrixXd target);
	pair<Matrix3d, Vector3d> point2planeICP(MatrixXd source, MatrixXd target, MatrixXd normal);
}

//Get corresponding points between targte mesh and source mesh - used KDTree for NN
MatrixXd icp::getCorrespondingPoints(MatrixXd source, MatrixXd target) {

	int pts_num = source.rows();
	int dimensionality = 3;
	KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree_index(dimensionality,target, 10);
	kd_tree_index.index->buildIndex();
	Eigen::MatrixXd matched = Eigen::MatrixXd::Zero(pts_num, 3);

	for (int idx = 0; idx < pts_num; idx++) {
		std::vector<double> query_pt(3);

		// vertex query (in source mesh)
		for (size_t d = 0; d < 3; d++) {
			query_pt[d] = source(idx, d);
		}

		// find closest vertex in target mesh
		const size_t results_num = 1;
		vector<size_t> ret_index(results_num);
		vector<double> out_dists_sqr(results_num);

		nanoflann::KNNResultSet<double> resultSet(results_num);     //result set

		resultSet.init(&ret_index[0], &out_dists_sqr[0]);
		kd_tree_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		// get closest vertex
		matched(idx, 0) = target(ret_index[0], 0);
		matched(idx, 1) = target(ret_index[0], 1);
		matched(idx, 2) = target(ret_index[0], 2);
	}
	return matched;
}


// Solve for rigid transformation (RT) for point-to-point ICP using JacobiSVD
pair<Matrix3d, Vector3d> icp::solveForRT(MatrixXd source, MatrixXd target) {
	long pointNum = source.rows();

	Eigen::Vector3d sumSrc, sumDst, pointSrc, pointDst;
	sumSrc = source.colwise().sum();
	sumDst = target.colwise().sum();

	// compute bar(p) and bar(q): zero mean point sets
	pointSrc = sumSrc / (double)pointNum;
	pointDst = sumDst / (double)pointNum;        // define bary centered point set

	// compute A
	Eigen::MatrixXd Src(pointNum, 3), Dst(pointNum, 3), Tem;
	Src = (source - pointSrc.replicate(1, source.rows()).transpose());
	Dst = (target - pointDst.replicate(1, target.rows()).transpose()).transpose();    // the least-square transformation

	Tem = Dst * Src;
	//ref: https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
	JacobiSVD<Eigen::MatrixXd> svd(Tem, ComputeThinU | ComputeThinV);
	Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();     // rotation matrix
	Eigen::Vector3d T = pointSrc - R * pointDst;                     // translation  vector

	return make_pair(R, T);
}

//Get sampled point set from source mesh 
MatrixXd icp::getSampleSource(MatrixXd source, int sample_gap) {
	int pts_num = round(source.rows() / sample_gap);

	Eigen::MatrixXd sample = Eigen::MatrixXd(pts_num, 3);
	for (int idx = 0; idx < pts_num; idx++) {
		sample.row(idx) = source.row(idx*sample_gap);
	}

	return sample;
}



//Add Gaussian noise to mesh
MatrixXd icp::addNoise(MatrixXd source, double noiseDiv) {
	default_random_engine randGen;
	normal_distribution<double> gaussian_dis(0.0, noiseDiv);    // gaussian_dis(dis_mean, dis_div)
	Eigen::Vector3d noise;
	for (int i = 0; i < source.rows(); i++) {
		noise[0] = gaussian_dis(randGen) / 100;
		noise[1] = gaussian_dis(randGen) / 100;
		noise[2] = gaussian_dis(randGen) / 100;
		source.row(i) = source.row(i) + noise.transpose();
	}
	return source;
}

//Error measurement (Residue error)
bool icp::detectError(MatrixXd source, MatrixXd target, pair<Matrix3d, Vector3d> RT) 
{
	source = (source - RT.second.replicate(1, source.rows()).transpose())* RT.first;
	Eigen::Vector3d diff = (source - target).colwise().sum();
	//Mean squared error
	double error = diff.norm();

	if (error < exp(-10)) {
		return false;
	}
	else {
		return true;
	}
}


// Calculate normal for each vertices in source using KDTree
MatrixXd icp::getNormal(MatrixXd source, MatrixXd target) 
{
	int pts_num = source.rows();
	int dimensionality = 3;
	Eigen::MatrixXd normal(pts_num, 3);

	// get meancenter of target mesh
	Eigen::MatrixXd meancenter(1, 3);
	meancenter = target.colwise().sum() / double(target.rows());

	KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree_index(dimensionality,target, 10);
	kd_tree_index.index->buildIndex();

	for (int idx = 0; idx < pts_num; idx++) {
		std::vector<double> query_pt(3);

		for (size_t d = 0; d < 3; d++) {
			query_pt[d] = source(idx, d);
		}

		// closest 8 neighbors to form a plane
		const size_t results_num = 8;         
		vector<size_t> ret_index(results_num);
		vector<double> out_dists_sqr(results_num);

		//result set
		nanoflann::KNNResultSet<double> resultSet(results_num);     

		resultSet.init(&ret_index[0], &out_dists_sqr[0]);
		kd_tree_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		// get cloest 8 neighbor points index in target
		Eigen::MatrixXd selectPts(results_num, 3);
		for (size_t i = 0; i < results_num; i++) {
			selectPts(i, 0) = target(ret_index[i], 0);
			selectPts(i, 1) = target(ret_index[i], 1);
			selectPts(i, 2) = target(ret_index[i], 2);
		}

		// compute plane normal
		Eigen::RowVector3d Nvt, Ct;
		igl::fit_plane(selectPts, Nvt, Ct);

		normal(idx, 0) = Nvt(0);
		normal(idx, 1) = Nvt(1);
		normal(idx, 2) = Nvt(2);

		// check the direction of normal vector
		if ((meancenter(0, 0) - target(idx, 0)) * normal(idx, 0) + (meancenter(0, 1) - target(idx, 1)) * normal(idx, 1) + (meancenter(0, 2) - target(idx, 2)) * normal(idx, 2) > 0) {
			normal(idx, 0) = -Nvt(0);
			normal(idx, 1) = -Nvt(1);
			normal(idx, 2) = -Nvt(2);
		}
	}
	return normal;
}


// Solve for rigid transformation (RT) for point-to-plane ICP
pair<Matrix3d, Vector3d> icp::point2planeICP(MatrixXd source, MatrixXd target, MatrixXd normal) 
{
	// build A and b
	int pts_num = source.rows();
	Eigen::MatrixXd A(pts_num, 6);
	Eigen::MatrixXd b(pts_num, 1);
	for (int idx = 0; idx < source.rows(); idx++) {
		A(idx, 0) = normal(idx, 2)*target(idx, 1) - normal(idx, 1)*target(idx, 2);
		A(idx, 1) = normal(idx, 0)*target(idx, 2) - normal(idx, 2)*target(idx, 0);
		A(idx, 2) = normal(idx, 1)*target(idx, 0) - normal(idx, 0)*target(idx, 1);
		A(idx, 3) = normal(idx, 0);
		A(idx, 4) = normal(idx, 1);
		A(idx, 5) = normal(idx, 2);

		Eigen::MatrixXd diff(pts_num, 3);
		diff = target - source;
		b(idx) = 0;
		for (int n = 0; n < 3; n++) {
			b(idx) = b(idx) - (diff(idx, n)*normal(idx, n));
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
	Eigen::Vector3d T;
	T(0) = x(3);
	T(1) = x(4);
	T(2) = x(5);

	return make_pair(R, T);
}

#endif /* icp_hpp */