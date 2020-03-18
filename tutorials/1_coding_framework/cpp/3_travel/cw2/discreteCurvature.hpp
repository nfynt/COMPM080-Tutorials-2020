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
	
	static vector<vector<pair<int,int> > > one_ring_neighbor;		//1 ring neighbor for each vertex of current mesh

	//Member Functions

	void updateOneRingNeighbor(MatrixXd V, MatrixXi F);
	//laplace beltrami operator for uniform discretization
	Eigen::SparseMatrix<double> laplaceOperator(MatrixXd V);
	Eigen::SparseMatrix<double> cotan_discretization(MatrixXd V, MatrixXi F);
	// Calculate uniform mean curvature
	MatrixXd meanCurvature(MatrixXd V, MatrixXi F, MatrixXd N);
	// Calculate non-uniform mean curvature
	MatrixXd nonUniformCurvature(MatrixXd V, MatrixXi F, MatrixXd N);
	MatrixXd gaussCurvature(MatrixXd V, MatrixXi F);
	MatrixXd eigenReconstruction(MatrixXd V, MatrixXi F, int k, int m);
	pair<double, double> cal_area_angle(MatrixXd V, MatrixXi F, int idx);
	//MatrixXcd get_eigenvecs(SparseMatrix<double> lapla, int k, int m);
	MatrixXd complex2real(MatrixXcd complex_mat);

	//Debug
	void debug1RingNeighbor()
	{
		for (int i = 0; i < one_ring_neighbor.size(); i++)
		{
			cout << "Neighbor " << i << ": ";
			for (int j = 0; j < one_ring_neighbor[i].size(); j++)
				cout << one_ring_neighbor[i][j].first<< "-"<<one_ring_neighbor[i][j].first << " ";
			cout << endl;
		}
	}
}

//Update/Recalculate one ring neighbor
void cw2::updateOneRingNeighbor(MatrixXd V, MatrixXi F)
{
	one_ring_neighbor.clear();
	/*cout << "endl";
	for (int j = 0; j < F.rows(); j++)
		cout << "F_"<<j<<": " << F.row(j)<<endl;*/
	//build neighbor of vertices
	for (int i = 0; i < V.rows(); i++) {
		std::vector<pair<int,int> > neighbor;
		for (int j = 0; j < F.rows(); j++) {
			//check face tris
			if (F(j, 0) == i)
			{
				neighbor.push_back(make_pair(F(j, 1), F(j, 2)));
				//neighbor.push_back(F(j, 2));
			}
			else if (F(j, 1) == i) {
				neighbor.push_back(make_pair(F(j, 2), F(j, 0)));
				//neighbor.push_back(F(j, 0));
			}
			else if (F(j, 2) == i) {
				neighbor.push_back(make_pair(F(j, 0), F(j, 1)));
				//neighbor.push_back(F(j, 1));
			}

		}
		one_ring_neighbor.push_back(neighbor);
		
	}

	cout << "Finished calculating one ring neighbor\n";
}

// Calculate uniform mean curvature
Eigen::MatrixXd cw2::meanCurvature(MatrixXd V, MatrixXi F, MatrixXd N) {
	
	if (one_ring_neighbor.size() < 1) updateOneRingNeighbor(V, F);

	//Eigen::MatrixXd N(V.rows(), 3);
	//igl::per_vertex_normals(V, F, N);

	Eigen::SparseMatrix<double> lap_b = laplaceOperator(V);
	

	// mean curvature
	// -2Hn = lap_b*x
	Eigen::VectorXd H = 0.5*(lap_b*V).rowwise().norm();

	// direction of curvature;
	for (int i = 0; i < V.rows(); i++) {
		
		int sz = one_ring_neighbor[i].size();

		RowVector3d avg;
		avg.setZero();

		for (int j = 0; j < sz; j++) {
			RowVector3d currentRow = V.row(one_ring_neighbor[i][j].first);
			avg += currentRow;
		}

		avg /= sz;

		//Dot product
		if (N.row(i).dot(avg - V.row(i)) > 0) {
			H.row(i) = -H.row(i);
		}
	}

	// jet mean curvature
	Eigen::MatrixXd C(F.rows(), 3);
	igl::jet(H, true, C);

	return C;
}

// Calculate gaussian curvature
Eigen::MatrixXd cw2::gaussCurvature(MatrixXd V, MatrixXi F) {
	Eigen::VectorXd K(V.rows());

	for (int i = 0; i < V.rows(); i++) {
		pair<double, double> areaAngle = cal_area_angle(V, F, i);
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
			lap.insert(i, one_ring_neighbor[i][j].first) = 1.0 / double(valence);
		}

		lap.insert(i, i) = -1.0;
	}
	
	return lap;
}

// calcualte angle (in radian) and area of curr vertex
pair<double, double> cw2::cal_area_angle(MatrixXd V, MatrixXi F, int ind) {
	
	double ang_deficit = 2.0*M_PI;
	double tot_area = 0;
	RowVector3d pt1, pt2, pt3;
	pt1 = V.row(ind);
	int valence = one_ring_neighbor[ind].size();
	for (int i = 0; i < valence; i++)
	{
		pt2 = V.row(one_ring_neighbor[ind][i].first);
		pt3 = V.row(one_ring_neighbor[ind][i].second);

		//dot product
		double theta = acos((pt2 - pt1).dot(pt3 - pt1) / ((pt2 - pt1).norm()*(pt3 - pt1).norm()));
		//cross product
		double area = (pt2 - pt1).norm()*(pt3 - pt1).norm()*sin(theta)/2;
		tot_area += area;
		ang_deficit -= theta;
	}

	return make_pair(tot_area, ang_deficit);
}


// Calculate non-uniform mean curvature
Eigen::MatrixXd cw2::nonUniformCurvature(MatrixXd V, MatrixXi F, MatrixXd N) {

	if (one_ring_neighbor.size() < 1) updateOneRingNeighbor(V, F);

	//Eigen::MatrixXd N(V.rows(), 3);
	//igl::per_vertex_normals(V, F, N);

	Eigen::SparseMatrix<double> lap_b = cotan_discretization(V,F);


	// mean curvature
	// -2Hn = lap_b*x
	Eigen::VectorXd H = 0.5*(lap_b*V).rowwise().norm();

	// direction of curvature;
	for (int i = 0; i < V.rows(); i++) {

		int sz = one_ring_neighbor[i].size();

		RowVector3d avg;
		avg.setZero();

		for (int j = 0; j < sz; j++) {
			RowVector3d currentRow = V.row(one_ring_neighbor[i][j].first);
			avg += currentRow;
		}

		avg /= sz;

		//Dot product
		if (N.row(i).dot(avg - V.row(i)) > 0) {
			H.row(i) = -H.row(i);
		}
	}

	// jet mean curvature
	Eigen::MatrixXd C(F.rows(), 3);
	igl::jet(H, true, C);

	return C;
}


#endif