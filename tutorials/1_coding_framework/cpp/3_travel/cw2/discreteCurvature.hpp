#pragma once


#ifndef discreteCurvature_hpp
#define discreteCurvature_hpp

#include <math.h>
#include <stdio.h>
#include <random>
#include <vector>
#include <igl/fit_plane.h>
#include <igl/copyleft/marching_cubes.h>
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
#include "igl/jet.h"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cstdlib>

#include "Eigen/SparseCore"
#include "Eigen/SparseCholesky"
#include "Eigen/Sparse"
#include "Eigen/src/SparseCore/SparseMatrix.h"

using namespace Eigen;
using namespace std;
using namespace nanoflann;


namespace cw2 {
	//Member variables
	
	static vector<vector<int> > one_ring_neighbor;		//1 ring neighbor for each vertex of current mesh

	//Member Functions

	void updateOneRingNeighbor(MatrixXd V, MatrixXi F);
	//laplace beltrami operator for uniform discretization
	Eigen::SparseMatrix<double> laplaceOperator(MatrixXd V);
	//SparseMatrix<double> nonUniform_laplacian(MatrixXd V, MatrixXi F);
	// Calculate uniform mean curvature
	MatrixXd meanCurvature(MatrixXd V, MatrixXi F);
	MatrixXd gaussCurvature(MatrixXd V, MatrixXi F);
	MatrixXd eigenReconstruction(MatrixXd V, MatrixXi F, int k, int m);
	//calculate vertex normal of a mesh
	MatrixXd calculateNormals(MatrixXd V);
	//andgle in radian beteen ba and bc
	inline double angle_3pts(RowVector3d a, RowVector3d b, RowVector3d c);
	pair<double, double> get_area_angle(MatrixXd V, MatrixXi F, int idx);
	//MatrixXcd get_eigenvecs(SparseMatrix<double> lapla, int k, int m);
	MatrixXd complex2real(MatrixXcd complex_mat);
}

//Update/Recalculate one ring neighbor
void cw2::updateOneRingNeighbor(MatrixXd V, MatrixXi F)
{
	one_ring_neighbor.clear();
	
	//build neighbor of vertices
	for (int i = 0; i < V.rows(); i++) {
		
		std::vector<int> neighbor;
		for (int j = 0; j < F.rows(); j++) {

			//check face tris
			if (F(j, 0) == i + 1)
			{
				neighbor.push_back(F(j, 1));
				//neighbor.push_back(F(j, 2));
			}
			else if (F(j, 1) == i + 1) {
				neighbor.push_back(F(j, 2));
				//neighbor.push_back(F(j, 0));
			}
			else if (F(j, 2) == i) {
				neighbor.push_back(F(j, 0));
				//neighbor.push_back(F(j, 1));
			}

		}
		one_ring_neighbor.push_back(neighbor);
		
	}

	cout << "Finished calculating one ring neighbor\n";
}

// Calculate uniform mean curvature
Eigen::MatrixXd cw2::meanCurvature(MatrixXd V, MatrixXi F) {
	
	if (one_ring_neighbor.size() < 1) updateOneRingNeighbor(V, F);

	Eigen::MatrixXd N(V.rows(), 3);
	
	igl::per_vertex_normals(V, F, N);
	//N = calculateNormals(V);

	Eigen::SparseMatrix<double> lap_b = laplaceOperator(V);
	

	// mean curvature
	// -2Hn = lap_b*x
	Eigen::VectorXd H = 0.5*(lap_b*V).rowwise().norm();

	// direction of curvature;
	for (int i = 0; i < V.rows(); i++) {
		
		int sz = one_ring_neighbor[i].size();

		RowVector3d average;
		average.setZero();

		for (int j = 0; j < sz; j++) {
			RowVector3d currentRow = V.row(one_ring_neighbor[i][j]);
			average += currentRow;
		}

		average /= sz;

		//Dot product
		if (N.row(i).dot(average - V.row(i)) > 0) {
			H.row(i) = -H.row(i);
		}
	}

	// mean curvature as color matrix
	Eigen::MatrixXd C(F.rows(), 3);
	igl::jet(H, true, C);

	return C;
}

Eigen::MatrixXd cw2::gaussCurvature(MatrixXd V, MatrixXi F) {
	Eigen::VectorXd K(V.rows());

	for (int i = 0; i < V.rows(); i++) {
		pair<double, double> areaAngle = get_area_angle(V, F, i);
		double area = areaAngle.first;
		K(i) = areaAngle.second / (area / 3.0);
	}
	Eigen::MatrixXd C(F.rows(), 3);
	igl::jet(K, true, C);

	return C;
}


//laplace operator for current vertex mesh
Eigen::SparseMatrix<double> cw2::laplaceOperator(MatrixXd V) {
	// save memory for results
	Eigen::SparseMatrix<double> lap(V.rows(), V.rows());
	int valence;
	for (int i = 0; i < V.rows(); i++)
	{
		// valence of current vertex
		valence = one_ring_neighbor[i].size();

		// create sparse mat
		for (int j = 0; j < valence; j++) {
			lap.insert(i, one_ring_neighbor[i][j]) = 1.0 / double(valence);
		}

		lap.insert(i, i) = -1.0;
	}
	
	return lap;
}

pair<double, double> cw2::get_area_angle(MatrixXd V, MatrixXi F, int idx) {
	RowVector3d P1, P2, P3;
	double angle_deflicit = 2.0*M_PI;
	double area_sum = 0;

	//----- get faces with current vertex
	for (int j = 0; j < F.rows(); j++) {
		RowVector3i currentF = F.row(j);
		bool neighFound = false;
		if (currentF(0) == idx) {
			P1 = V.row(currentF(0));
			P2 = V.row(currentF(1));
			P3 = V.row(currentF(2));
			neighFound = true;
		}
		else if (currentF(1) == idx) {
			P1 = V.row(currentF(1));
			P2 = V.row(currentF(2));
			P3 = V.row(currentF(0));
			neighFound = true;
		}
		else if (currentF(2) == idx) {
			P1 = V.row(currentF(2));
			P2 = V.row(currentF(0));
			P3 = V.row(currentF(1));
			neighFound = true;
		}

		//----- compute area sum and angle deficit
		if (neighFound) {
			double theta = angle_3pts(P2, P1, P3);
			double area = 0.5*(P2 - P1).norm()*(P3 - P1).norm()*sin(theta);
			area_sum = area_sum + area;
			angle_deflicit = angle_deflicit - theta;
		}
	}
	return make_pair(area_sum, angle_deflicit);
}

MatrixXd cw2::calculateNormals(MatrixXd V) {

	int size = V.rows();
	Eigen::MatrixXd vert_norm(size, 3);

	Eigen::MatrixXd centroid(1, 3);
	centroid = V.colwise().sum() / double(size);

	int dim = 3;
	KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree_index(dim, V);
	kd_tree_index.index->buildIndex();

	for (int i = 0; i < size; i++) {
		std::vector<double> query_pt(3);

		query_pt[0] = V(i, 0);
		query_pt[1] = V(i, 1);
		query_pt[2] = V(i, 2);

		// 8 neighbors
		const size_t neigh_sz = 8;
		vector<size_t> ret_index(neigh_sz);
		vector<double> out_dists_sqr(neigh_sz);

		// result set
		nanoflann::KNNResultSet<double> resultSet(neigh_sz);

		resultSet.init(&ret_index[0], &out_dists_sqr[0]);
		kd_tree_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		// closest 8 neighbor points
		Eigen::MatrixXd selectPts(neigh_sz, 3);

		for (size_t i = 0; i < neigh_sz; i++) {
			selectPts(i, 0) = V(ret_index[i], 0);
			selectPts(i, 1) = V(ret_index[i], 1);
			selectPts(i, 2) = V(ret_index[i], 2);
		}

		Eigen::RowVector3d Nvt, Ct;
		igl::fit_plane(selectPts, Nvt, Ct);

		vert_norm(i, 0) = Nvt(0);
		vert_norm(i, 1) = Nvt(1);
		vert_norm(i, 2) = Nvt(2);

		//------ check the direction of normal vector
		if ((centroid(0, 0) - V(i, 0)) * vert_norm(i, 0) + (centroid(0, 1) - V(i, 1)) * vert_norm(i, 1) + (centroid(0, 2) - V(i, 2)) * vert_norm(i, 2) > 0) {
			vert_norm(i, 0) = -Nvt(0);
			vert_norm(i, 1) = -Nvt(1);
			vert_norm(i, 2) = -Nvt(2);
		}
	}
	return vert_norm;
}

// angle in radians between ba and bc
inline double cw2::angle_3pts(RowVector3d a, RowVector3d b, RowVector3d c) {
	return acos((a-b).dot(c-b) / ((a-b).norm()*(c-b).norm()));
}
#endif