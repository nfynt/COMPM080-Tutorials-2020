#pragma once

#include <igl/fit_plane.h>
#include <igl/avg_edge_length.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/parula.h>
#include <igl/per_corner_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/triangle_triangle_adjacency.h>

#include "nanoflann.hpp"
#include "DiscreteCurvature.hpp"
#include "MyContext.hpp"
#include "igl/jet.h"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cstdlib>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

using namespace Eigen;
using namespace std;
using namespace nanoflann;
using namespace cw2;

namespace cw2 {
	pair<MatrixXd, MatrixXd> explicit_smooth(MatrixXd V, MatrixXi F, const MyContext& mCtx);
	pair<MatrixXd, MatrixXd> implicit_smooth(MatrixXd V, MatrixXi F, const MyContext& mCtx);
	MatrixXd add_noise(MatrixXd V, double noise_level);
	float compute_error(MatrixXd original_V, MatrixXd smooth_V);
}

pair<MatrixXd, MatrixXd> cw2::explicit_smooth(MatrixXd V, MatrixXi F, const MyContext& mCtx) {
	
	// ----- compute required Laplacian Operator
	Eigen::SparseMatrix<double> Lapla(V.rows(), V.rows());
	
	Lapla = cw2::cotan_discretization(V, F,mCtx);
	


	Eigen::SparseMatrix<double> I(V.rows(), V.rows());
	I.setIdentity();

	MatrixXd smooth_V;
	smooth_V = (I + mCtx.lambda * Lapla)*V;

	// ----- compute mean curvature
	Eigen::VectorXd H = 0.5*(Lapla*V).rowwise().norm();

	// ----- orient mean curvature
	std::vector<int> uniq_neighbour;
	bool findV;
	int position, nei_N;
	for (int i = 0; i < V.rows(); i++) {
		for (int j = 0; j < F.rows(); j++) {
			findV = false;

			// go through all verteces compose the face and find the current one
			for (int k = 0; k < 3; k++) {
				if (F(j, k) == i) {
					position = k;
					findV = true;
					break;
				}
			}

			// save neighbours
			if (findV) {
				uniq_neighbour.push_back(F(j, (position + 1) % 3));
				uniq_neighbour.push_back(F(j, (position + 2) % 3));
			}
		}

		// get size of neighbours
		nei_N = uniq_neighbour.size();
	Eigen:RowVector3d average;
		average.setZero();

		// fill sparse matrix with neighbours
		for (int idx = 0; idx < nei_N; idx++) {
			RowVector3d currentRow = V.row(uniq_neighbour[idx]);
			average = average + currentRow / nei_N;
		}

		if (mCtx.VN.row(i).dot(average - V.row(i)) > 0) {
			H.row(i) = -H.row(i);
		}
		uniq_neighbour.clear();
	}

	// set mean curvature as color matrix
	Eigen::MatrixXd C(F.rows(), 3);
	igl::jet(H, true, C);

	return make_pair(smooth_V, C);
}

pair<MatrixXd, MatrixXd> cw2::implicit_smooth(MatrixXd V, MatrixXi F, const MyContext& mCtx) {
	
	SparseMatrix<double> Lapla(V.rows(), V.rows()), 
		C_mat(V.rows(), V.rows()), 
		M_inv(V.rows(), V.rows()), M(V.rows(), V.rows());
	RowVector3d P1, P2, P3;
	int adjTri_idx, adjEdge_idx;
	double alpha_ij, beta_ij, weight_ij;

	for (int i = 0; i < V.rows(); i++) {
		pair<double, double> areaAngle = cw2::calcAreaAngle(V, F, i);
		double area = areaAngle.first / 3;
		double wij_sum = 0;

		int nextP, preP, edge;
		for (int j = 0; j < F.rows(); j++) {
			nextP = -1;
			preP = -1;
			edge = -1;
			if (F(j, 0) == i) {
				nextP = F(j, 1);
				preP = F(j, 2);
				edge = 2;
			}
			else if (F(j, 1) == i) {
				nextP = F(j, 2);
				preP = F(j, 0);
				edge = 0;
			}
			else if (F(j, 2) == i) {
				nextP = F(j, 0);
				preP = F(j, 1);
				edge = 1;
			}

			// if any neighbor is found
			if (edge != -1) {
				P1 = V.row(i);
				P2 = V.row(nextP);
				P3 = V.row(preP);

				// compute angel between vertecies
				beta_ij = get_angle(P1, P2, P3);

				// find adjacne triangles
				adjTri_idx = mCtx.TT(j, edge);
				adjEdge_idx = mCtx.TTi(j, edge);

				RowVector3i adjF = F.row(adjTri_idx);
				if (adjEdge_idx == 0) {
					P1 = V.row(adjF(0));
					P2 = V.row(adjF(1));
					P3 = V.row(adjF(2));
				}
				else if (adjEdge_idx == 1) {
					P1 = V.row(adjF(1));
					P2 = V.row(adjF(2));
					P3 = V.row(adjF(0));
				}
				else if (adjEdge_idx == 2) {
					P1 = V.row(adjF(2));
					P2 = V.row(adjF(0));
					P3 = V.row(adjF(1));
				}

				// compute angle: alpha_ij
				alpha_ij = get_angle(P1, P3, P2);

				// compute cotan weight w_ij
				weight_ij = tan(M_PI / 2 - alpha_ij) + tan(M_PI / 2 - beta_ij);

				// fill laplacian matrix
				C_mat.insert(i, preP) = weight_ij;
				wij_sum = wij_sum + weight_ij;
			}

		}
		C_mat.insert(i, i) = -wij_sum;
		M.insert(i, i) = 2.0*area;
		M_inv.insert(i, i) = 1.0 / (2.0*area);

	}

	// compute laplacian operator
	Lapla = M_inv * C_mat;

	// compute A
	SimplicialCholesky<SparseMatrix<double>> chol(M - mCtx.lambda * M*Lapla);

	// compute b
	VectorXd x1 = chol.solve(M*V.col(0));
	VectorXd x2 = chol.solve(M*V.col(1));
	VectorXd x3 = chol.solve(M*V.col(2));
	MatrixXd x(V.rows(), 3);
	x << x1, x2, x3;

	// ----- compute mean curvature
	Eigen::VectorXd H = 0.5*(Lapla*V).rowwise().norm();

	// ----- orient mean curvature
	std::vector<int> uniq_neighbour;
	bool findV;
	int position, nei_N;
	for (int i = 0; i < V.rows(); i++) {
		for (int j = 0; j < F.rows(); j++) {
			findV = false;

			// go through all verteces compose the face and find the current one
			for (int k = 0; k < 3; k++) {
				if (F(j, k) == i) {
					position = k;
					findV = true;
					break;
				}
			}

			// save neighbours
			if (findV) {
				uniq_neighbour.push_back(F(j, (position + 1) % 3));
				uniq_neighbour.push_back(F(j, (position + 2) % 3));
			}
		}

		// get size of neighbours
		nei_N = uniq_neighbour.size();
	Eigen:RowVector3d average;
		average.setZero();

		// fill sparse matrix with neighbours
		for (int idx = 0; idx < nei_N; idx++) {
			RowVector3d currentRow = V.row(uniq_neighbour[idx]);
			average = average + currentRow / nei_N;
		}

		if (mCtx.VN.row(i).dot(average - V.row(i)) > 0) {
			H.row(i) = -H.row(i);
		}
		uniq_neighbour.clear();
	}

	// set mean curvature as color matrix
	Eigen::MatrixXd C(F.rows(), 3);
	igl::jet(H, true, C);

	return make_pair(x, C);
}

MatrixXd cw2::add_noise(MatrixXd V, double noise_level) {
	// ----- randomizer initialization
	std::srand(std::time(0));
	// compute noise variance
	double nX = V.col(0).maxCoeff() - V.col(0).minCoeff();
	double nY = V.col(1).maxCoeff() - V.col(1).minCoeff();
	double nZ = V.col(2).maxCoeff() - V.col(2).minCoeff();

	MatrixXd noise_V(V.rows(), 3);
	for (int i = 0; i < V.rows(); i++) {
		for (int j = 0; j < 3; j++) {
			noise_V(i, j) = V(i, j);
		}
	}

	for (int n = 0; n < V.rows(); n++) {
		noise_V.row(n) = noise_V.row(n) + RowVector3d(
			noise_level * nX * ((double)std::rand() / (RAND_MAX)),
			noise_level * nY * ((double)std::rand() / (RAND_MAX)),
			noise_level * nZ * ((double)std::rand() / (RAND_MAX)));
	}
	return noise_V;
}

float cw2::compute_error(MatrixXd original_V, MatrixXd smooth_V) {
	float error = (original_V - smooth_V).rowwise().norm().sum();
	return error;
}
