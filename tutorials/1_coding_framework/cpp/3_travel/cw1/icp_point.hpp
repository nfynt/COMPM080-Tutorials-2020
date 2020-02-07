#ifndef icp_point
#define icp_point

#include <stdio.h>
#include <ctime>
#include <iostream>
#include <math.h>
#include <random>
#include <igl/fit_plane.h>
#include <igl/per_vertex_normals.h>

#include "nanoflann.hpp"
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace nanoflann;


namespace icp {
	void getCorrespondingPoints(MatrixXd source, MatrixXd target, MatrixXd& points);
	void solveForRT(MatrixXd source, MatrixXd target, pair<Matrix3d, Vector3d>& RT);
	bool detectError(MatrixXd source, MatrixXd target, pair<Matrix3d, Vector3d> RT);
	void addGaussianNoise(MatrixXd& source, double noiseDiv);
	void getSampledSource(MatrixXd source, int sample_gap, MatrixXd& sampledV);
}




//Get corresponding points between targte mesh and source mesh - used KDTree for NN
void icp::getCorrespondingPoints(MatrixXd source, MatrixXd target, MatrixXd& points)
{
	int pts_cnt = source.rows();
	int dimensionality = 3;
	KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree_index(dimensionality, target, 10);
	kd_tree_index.index->buildIndex();
	
	points.resize(pts_cnt, 3);
	points = Eigen::MatrixXd::Zero(pts_cnt, 3);

	for (int i = 0; i < pts_cnt; i++) {
		
		RowVector3d curr_point = source.row(i);

		// find closest vertex in target mesh
		const size_t results_cnt = 1;
		vector<size_t> ret_ind(results_cnt);
		vector<double> dist(results_cnt);	//not used

		nanoflann::KNNResultSet<double> result(results_cnt);     //result set

		result.init(&ret_ind[0], &dist[0]);
		kd_tree_index.index->findNeighbors(result, &curr_point[0], nanoflann::SearchParams(10));

		// get closest vertex
		points.row(i) = target.row(ret_ind[0]);
	}
}


// Solve for rigid transformation (RT) for point-to-point ICP using JacobiSVD
void icp::solveForRT(MatrixXd source, MatrixXd target, pair<Matrix3d, Vector3d>& RT) {
	long pointNum = source.rows();

	// compute bar(p) and bar(q): zero mean point sets
	Eigen::Vector3d src_bar,target_bar;
	src_bar = source.colwise().sum() / (double)pointNum;
	target_bar = target.colwise().sum() / (double)pointNum;


	// covariance matrix A
	Eigen::MatrixXd Src(pointNum, 3), Dst(pointNum, 3), A;
	Src = (source - src_bar.replicate(1, source.rows()).transpose());
	Dst = (target - target_bar.replicate(1, target.rows()).transpose()).transpose();

	A = Dst * Src;
	//ref: https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
	JacobiSVD<Eigen::MatrixXd> svd(A, ComputeThinU | ComputeThinV);
	Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();     // R = V.U'
	Eigen::Vector3d T = src_bar - R * target_bar;                     // t = p - Rq

	RT = make_pair(R, T);
}


//Add Gaussian noise to mesh
void icp::addGaussianNoise(MatrixXd& source, double noiseDiv) {
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
	cout << "residue error:" << err << endl;

	if (err < 0.1)
		return false;
	else
		return true;
}

//Get sampled point set from source mesh 
void icp::getSampledSource(MatrixXd source, int sample_gap, MatrixXd& sampledV) {
	int pts_cnt = round(source.rows() / sample_gap);

	sampledV.resize(pts_cnt, 3);
	for (int idx = 0; idx < pts_cnt; idx++) {
		sampledV.row(idx) = source.row(idx*sample_gap);
	}
}


#endif







/*
 __  _ _____   ____  _ _____
|  \| | __\ `v' /  \| |_   _|
| | ' | _| `. .'| | ' | | |
|_|\__|_|   !_! |_|\__| |_|

*/

