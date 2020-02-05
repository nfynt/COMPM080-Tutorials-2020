
#ifndef icp_hpp
#define icp_hpp

#include <math.h>
#include <random>
#include <stdio.h>
#include <ctime>
#include <iostream>
#include <igl/fit_plane.h>
#include <igl/per_vertex_normals.h>

#include "nanoflann.hpp"
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace nanoflann;

namespace icp {
	void getCorrespondingPoints(MatrixXd source, MatrixXd target, MatrixXd& points);
	void solveForRT(MatrixXd source, MatrixXd target, pair<Matrix3d,Vector3d>& RT);
	bool detectError(MatrixXd source, MatrixXd target, pair<Matrix3d, Vector3d> RT);
	void gaussNoise(MatrixXd& source, double noiseDiv);
	void getSampledSource(MatrixXd source, int sample_gap, MatrixXd& sampledV);
	void getNormal(MatrixXd source, MatrixXd target, MatrixXd& normal);
	void point2planeICP(MatrixXd source, MatrixXd target, MatrixXd normal, pair<Matrix3d, Vector3d>& RT);
}

//Get corresponding points between targte mesh and source mesh - used KDTree for NN
void icp::getCorrespondingPoints(MatrixXd source, MatrixXd target, MatrixXd& points)
{
	int pts_num = source.rows();
	int dimensionality = 3;
	KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree_index(dimensionality,target, 10);
	kd_tree_index.index->buildIndex();
	points.resize(pts_num, 3);
	points = Eigen::MatrixXd::Zero(pts_num, 3);

	for (int idx = 0; idx < pts_num; idx++) {
		std::vector<double> query_pt(3);

		// vertex query (in source mesh)
		for (size_t d = 0; d < 3; d++) {
			query_pt[d] = source(idx, d);
		}

		// find closest vertex in target mesh
		const size_t results_num = 1;
		vector<size_t> ret_ind(results_num);
		vector<double> out_dists_sqr(results_num);

		nanoflann::KNNResultSet<double> result(results_num);     //result set

		result.init(&ret_ind[0], &out_dists_sqr[0]);
		kd_tree_index.index->findNeighbors(result, &query_pt[0], nanoflann::SearchParams(10));

		// get closest vertex
		points(idx, 0) = target(ret_ind[0], 0);
		points(idx, 1) = target(ret_ind[0], 1);
		points(idx, 2) = target(ret_ind[0], 2);
	}
}


// Solve for rigid transformation (RT) for point-to-point ICP using JacobiSVD
void icp::solveForRT(MatrixXd source, MatrixXd target, pair<Matrix3d, Vector3d>& RT) {
	long pointNum = source.rows();

	Eigen::Vector3d sumSrc, sumDst, pointSrc, pointDst;
	sumSrc = source.colwise().sum();
	sumDst = target.colwise().sum();

	// compute bar(p) and bar(q): zero mean point sets
	pointSrc = sumSrc / (double)pointNum;
	pointDst = sumDst / (double)pointNum; 

	// covariance matrix A
	Eigen::MatrixXd Src(pointNum, 3), Dst(pointNum, 3), Tem;
	Src = (source - pointSrc.replicate(1, source.rows()).transpose());
	Dst = (target - pointDst.replicate(1, target.rows()).transpose()).transpose();

	Tem = Dst * Src;
	//ref: https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
	JacobiSVD<Eigen::MatrixXd> svd(Tem, ComputeThinU | ComputeThinV);
	Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();     // R = V.U'
	Eigen::Vector3d T = pointSrc - R * pointDst;                     // t = p - Rq

	RT= make_pair(R, T);
}


//Add Gaussian noise to mesh
void icp::gaussNoise(MatrixXd& source, double noiseDiv) {
	default_random_engine randGen;
	normal_distribution<double> gaussian_dis(0.0, noiseDiv);    // gaussian_dis(dis_mean, dis_div)
	Eigen::Vector3d noise;
	for (int i = 0; i < source.rows(); i++) {
		noise[0] = gaussian_dis(randGen) / 100;
		noise[1] = gaussian_dis(randGen) / 100;
		noise[2] = gaussian_dis(randGen) / 100;
		source.row(i) = source.row(i) + noise.transpose();
	}
}

//Error measurement (Residue error)
bool icp::detectError(MatrixXd source, MatrixXd target, pair<Matrix3d, Vector3d> RT) 
{
	if (source.rows() < target.rows())
	{
		MatrixXd tar = target.block(0, 0, source.rows(), 3);
		target.resize(source.rows(), 3);
		target = tar;
	}
	else if (source.rows() > target.rows())
	{
		MatrixXd tar = source.block(0, 0, target.rows(), 3);
		source.resize(target.rows(), 3);
		source = tar;
	}
	source = (source - RT.second.replicate(1, source.rows()).transpose())* RT.first;
	//for(int i=0;i<target.rows)
	Eigen::Vector3d diff = (source - target).colwise().sum();
	//Eigen::Vector3d diff = target.colwise().sum() - source.colwise().sum();
	cout << source.rows() << "x" << target.rows() << endl;
	cout << diff << endl;
	//Mean squared error
	double err = diff.norm();
	cout <<"residue error:"<< err << endl;
	
	if (err < exp(-10))
		return false;
	else
		return true;
}

//Get sampled point set from source mesh 
void icp::getSampledSource(MatrixXd source, int sample_gap, MatrixXd& sampledV) {
	int pts_num = round(source.rows() / sample_gap);

	sampledV.resize(pts_num, 3);
	for (int idx = 0; idx < pts_num; idx++) {
		sampledV.row(idx) = source.row(idx*sample_gap);
	}
}



// Calculate normal for each vertices in source using KDTree
void icp::getNormal(MatrixXd source, MatrixXd target, MatrixXd& normal)
{
	int pts_num = source.rows();
	int dimensionality = 3;
	normal.resize(pts_num, 3);

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
		const size_t neighbor_cnt = 8;         
		vector<size_t> ret_index(neighbor_cnt);
		vector<double> out_dists_sqr(neighbor_cnt);

		//result set
		nanoflann::KNNResultSet<double> resultSet(neighbor_cnt);

		resultSet.init(&ret_index[0], &out_dists_sqr[0]);
		kd_tree_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		// get closest 8 neighbor points index in target
		Eigen::MatrixXd selectPts(neighbor_cnt, 3);
		for (size_t i = 0; i < neighbor_cnt; i++) {
			selectPts(i, 0) = target(ret_index[i], 0);
			selectPts(i, 1) = target(ret_index[i], 1);
			selectPts(i, 2) = target(ret_index[i], 2);
		}

		// compute plane normal (Nvt) and points on plane (Cvt)
		Eigen::RowVector3d Nvt, Cvt;
		igl::fit_plane(selectPts, Nvt, Cvt);

		normal(idx, 0) = Nvt(0);
		normal(idx, 1) = Nvt(1);
		normal(idx, 2) = Nvt(2);

		// check the direction of normal vector
		// (M - Q).N > 0 invert the normal
		if ((meancenter(0, 0) - target(idx, 0)) * normal(idx, 0) + (meancenter(0, 1) - target(idx, 1)) * normal(idx, 1) + (meancenter(0, 2) - target(idx, 2)) * normal(idx, 2) > 0) {
			normal(idx, 0) = -Nvt(0);
			normal(idx, 1) = -Nvt(1);
			normal(idx, 2) = -Nvt(2);
		}
	}
}


// Solve for rigid transformation (RT) for point-to-plane ICP
void icp::point2planeICP(MatrixXd source, MatrixXd target, MatrixXd normal, pair<Matrix3d, Vector3d>& RT)
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

	RT = make_pair(R, T);
}

#endif /* icp_hpp */







/*
 __  _ _____   ____  _ _____  
|  \| | __\ `v' /  \| |_   _| 
| | ' | _| `. .'| | ' | | |   
|_|\__|_|   !_! |_|\__| |_|
 
*/
