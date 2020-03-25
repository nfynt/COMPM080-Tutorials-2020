#pragma once


#include "Discretization.hpp"
#include "MyContext.hpp"
#include "igl/jet.h"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

using namespace Eigen;
using namespace std;
using namespace cw2;

namespace cw2 {
	// Explicit smoothing of mesh
	Eigen::MatrixXd explicitSmoothing(MatrixXd V, MatrixXi F, const MyContext& mCtx, MatrixXd& col);
	Eigen::MatrixXd implicitSmoothing(MatrixXd V, MatrixXi F, const MyContext& mCtx, MatrixXd& col);
	MatrixXd addNoise(MatrixXd V, double noise_level);
}

MatrixXd cw2::explicitSmoothing(MatrixXd V, MatrixXi F, const MyContext& mCtx, MatrixXd& col) {
	
	MatrixXd N;
	igl::per_vertex_normals(V, F, N);

	Eigen::SparseMatrix<double> laplace(V.rows(), V.rows());
	Eigen::SparseMatrix<double> M, C;
	//laplace = cw2::cotan_discretization(V,F,mCtx);
	cw2::laplaceDecomposition(V, F, mCtx, laplace, M, C);

	MatrixXd diff_V(V.rows(),V.cols());
	//Diffusion eq: pi_n = (1 + lambda*laplace)*pi
	diff_V = V + mCtx.lambda * laplace * V;

	// ----- compute mean curvature
	Eigen::VectorXd H = 0.5*(laplace*V).rowwise().norm();

	int position, nei_N;
	for (int i = 0; i < V.rows(); i++) {
		int valence = one_ring_neighbor[i].size();
		Eigen:RowVector3d average;
		average.setZero();
		// centroid from neighbor
		for (int j = 0; j < valence; j++) {
			RowVector3d neighbor_pt = V.row(one_ring_neighbor[i][j].first);
			average += neighbor_pt/valence;
		}

		if (N.row(i).dot(average - V.row(i)) > 0) {
			H.row(i) = -H.row(i);
		}
	}

	//set mean curvature as color matrix
	col.resize(mCtx.F.rows(), 3);
	igl::jet(H, true, col);


	return diff_V;
}

MatrixXd cw2::implicitSmoothing(MatrixXd V, MatrixXi F, const MyContext& mCtx, MatrixXd& col) {
	
	SparseMatrix<double> laplace(V.rows(), V.rows()), 
		C(V.rows(), V.rows()), 
		M(V.rows(), V.rows());

	RowVector3d p1, p2, p3;
	int adj_tri_ind, adj_edge_ind;
	double alpha_ij, beta_ij, weight_ij;

	cw2::laplaceDecomposition(V, F, mCtx, laplace, M, C);

	// compute A = M - lambda*C
	SimplicialCholesky<SparseMatrix<double>> chol(M - mCtx.lambda * C);

	// compute b = M*P
	VectorXd b1 = chol.solve(M*V.col(0));
	VectorXd b2 = chol.solve(M*V.col(1));
	VectorXd b3 = chol.solve(M*V.col(2));

	MatrixXd x(V.rows(), 3);
	x << b1, b2, b3;

	Eigen::VectorXd H = 0.5*(laplace*V).rowwise().norm();

	// curvature direction
	for (int i = 0; i < V.rows(); i++) {

		int sz = one_ring_neighbor[i].size();

		RowVector3d avg;
		avg.setZero();

		for (int j = 0; j < sz; j++) {
			RowVector3d neighbor = V.row(one_ring_neighbor[i][j].first);
			avg += neighbor;
		}

		avg /= sz;

		//Dot product
		if (mCtx.VN.row(i).dot(avg - V.row(i)) > 0) {
			H.row(i) = -H.row(i);
		}
	}

	// mean curvature as color matrix
	col.resize(F.rows(), 3);
	igl::jet(H, true, col);

	return x;
}

// Zero mean gaussian noise
MatrixXd cw2::addNoise(MatrixXd V, double noise_level) {
	default_random_engine randGen;
	normal_distribution<double> gaussian_dis(0.0, noise_level);    // gaussian_dis(dis_mean, dis_div)
	Eigen::Vector3d noise;
	MatrixXd noise_V(V.rows(), V.cols());
	for (int i = 0; i < V.rows(); i++) {
		noise[0] = gaussian_dis(randGen) / 100;
		noise[1] = gaussian_dis(randGen) / 100;
		noise[2] = gaussian_dis(randGen) / 100;
		noise_V.row(i) = V.row(i) + noise.transpose();
	}
	return noise_V;
}







/*
 __  _ _____   ____  _ _____
|  \| | __\ `v' /  \| |_   _|
| | ' | _| `. .'| | ' | | |
|_|\__|_|   !_! |_|\__| |_|

*/

