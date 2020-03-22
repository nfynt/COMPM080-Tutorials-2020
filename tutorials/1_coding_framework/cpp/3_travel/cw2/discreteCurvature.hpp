#pragma once


#ifndef DiscreteCurvature_hpp
#define DiscreteCurvature_hpp

#include <math.h>
#include <vector>
#include <igl/fit_plane.h>
#include "igl/jet.h"

#include "MyContext.hpp"

#include "Eigen/SparseCore"
#include "Eigen/SparseCholesky"
#include "Eigen/Sparse"
#include "Eigen/src/SparseCore/SparseMatrix.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>


using namespace Eigen;
using namespace std;
using namespace Spectra;

namespace cw2 {
	//Member variables

	static vector<vector<pair<int, int> > > one_ring_neighbor;		//1 ring neighbor for each vertex of current mesh
	static vector<vector<int> > one_ring_faces;						//1 ring triangle faces
	static SparseMatrix<double> cotan_laplace;						//non-uniform laplace
	static SparseMatrix<double> uniform_laplace;					//uniform laplace

	//Member Functions

	//Angle (in rad) between p2p1 to p2p3 vector
	inline double get_angle(RowVector3d p1, RowVector3d p2, RowVector3d p3) {
		return acos((p1 - p2).dot(p3 - p2) / ((p1 - p2).norm()*(p3 - p2).norm()));
	}

	void updateOneRingNeighbor(MatrixXd V, MatrixXi F);
	//laplace beltrami operator for uniform discretization
	Eigen::SparseMatrix<double> laplaceOperator(MatrixXd V);
	// Calculate uniform mean curvature
	MatrixXd meanCurvature(MatrixXd V, MatrixXi F, MatrixXd N);
	// Gaussian curvature (uniform)
	MatrixXd gaussCurvature(MatrixXd V, MatrixXi F);
	//laplace beltrami operator for cotangent discretization
	Eigen::SparseMatrix<double> cotan_discretization(MatrixXd V, MatrixXi F, const MyContext& mCtx);
	// Calculate non-uniform mean curvature
	MatrixXd nonUniformCurvature(const MyContext& mCtx);
	// Mesh reconstruction using first k eigen vectors
	MatrixXd eigenReconstruction(MatrixXd V, MatrixXi F, const MyContext& mCtx, int k);
	pair<double, double> calcAreaAngle(MatrixXd V, MatrixXi F, int idx);

	//Debug
	void debug1RingNeighbor()
	{
		for (int i = 0; i < one_ring_neighbor.size(); i++)
		{
			cout << "Neighbor " << i << ": ";
			for (int j = 0; j < one_ring_neighbor[i].size(); j++)
				cout << one_ring_neighbor[i][j].first << "-" << one_ring_neighbor[i][j].second << " ";
			cout << "\nFaces " << i << ": ";
			for (int j = 0; j < one_ring_faces[i].size(); j++)
				cout << one_ring_faces[i][j] << " ";
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
		std::vector<int> faces;
		for (int j = 0; j < F.rows(); j++) {
			//check face tris
			if (F(j, 0) == i)
			{
				neighbor.push_back(make_pair(F(j, 1), F(j, 2)));
				//neighbor.push_back(F(j, 2));
				faces.push_back(j);

			}
			else if (F(j, 1) == i) {
				neighbor.push_back(make_pair(F(j, 2), F(j, 0)));
				//neighbor.push_back(F(j, 0));
				faces.push_back(j);
			}
			else if (F(j, 2) == i) {
				neighbor.push_back(make_pair(F(j, 0), F(j, 1)));
				//neighbor.push_back(F(j, 1));
				faces.push_back(j);
			}

		}
		one_ring_neighbor.push_back(neighbor);
		one_ring_faces.push_back(faces);
	}

	cout << "Finished calculating one ring neighbor\n";
}

// Calculate uniform mean curvature
Eigen::MatrixXd cw2::meanCurvature(MatrixXd V, MatrixXi F, MatrixXd N) {
	
	if (one_ring_neighbor.size() < 1) updateOneRingNeighbor(V, F);

	//Eigen::MatrixXd N(V.rows(), 3);
	//igl::per_vertex_normals(V, F, N);

	if (uniform_laplace.rows() < 1)
		uniform_laplace = laplaceOperator(V);
	

	// mean curvature
	// -2Hn = lap_b*x
	Eigen::VectorXd H = 0.5*(uniform_laplace*V).rowwise().norm();

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
	//curvature	
	Eigen::VectorXd K(V.rows());

	for (int i = 0; i < V.rows(); i++) {
		pair<double, double> areaAngle = calcAreaAngle(V, F, i);
		double area = areaAngle.first;
		K(i) = areaAngle.second / (area / 3.0);
	}
	Eigen::MatrixXd C(F.rows(), 3);
	igl::jet(K, true, C);

	return C;
}

//laplace operator for current vertex mesh
Eigen::SparseMatrix<double> cw2::laplaceOperator(MatrixXd V) {
	// sparse laplace matrix
	Eigen::SparseMatrix<double> lap(V.rows(), V.rows());
	int valence;
	for (int i = 0; i < V.rows(); i++)
	{
		// valence of current vertex
		valence = one_ring_neighbor[i].size();

		for (int j = 0; j < valence; j++) {
			lap.insert(i, one_ring_neighbor[i][j].first) = 1.0 / double(valence);
		}

		lap.insert(i, i) = -1.0;
	}
	
	return lap;
}

// calcualte angle (in radian) and 1 ring neighbor area of curr vertex
pair<double, double> cw2::calcAreaAngle(MatrixXd V, MatrixXi F, int ind) {
	
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

Eigen::SparseMatrix<double> cw2::cotan_discretization(MatrixXd V, MatrixXi F, const MyContext& mCtx) {
	SparseMatrix<double> lap_b(V.rows(), V.rows());
	SparseMatrix<double> C(V.rows(), V.rows());			//weight: sSymmetric matrix
	SparseMatrix<double> M_inv(V.rows(), V.rows());		//Diagonal matrix

	RowVector3d p1, p2, p3;
	int adj_tri_ind, adj_edge_ind;
	double alpha_ij, beta_ij, weight_ij;

	for (int i = 0; i < V.rows(); i++) {
		//M-1 mat
		pair<double, double> areaAngle = calcAreaAngle(V, F, i);
		double area = areaAngle.first / 3;
		M_inv.insert(i, i) = 1.0 / (2.0*area);

		//C mat
		double wij_sum = 0;

		int nxtP, preP, edgeId;
		for (int j = 0; j < one_ring_faces[i].size(); j++)
		{
			int face = one_ring_faces[i][j];
			if (F(face, 0) == i)
			{
				nxtP = F(face, 1);	preP = F(face, 2);
				edgeId = 2;
			}
			else if (F(face, 1) == i)
			{
				nxtP = F(face, 2);	preP = F(face, 0);
				edgeId = 0;
			}
			else {
				nxtP = F(face, 0); preP = F(face, 1);
				edgeId = 1;
			}

			p1 = V.row(i);
			p2 = V.row(nxtP);
			p3 = V.row(preP);

			// compute angel between vertecies
			beta_ij = get_angle(p1,p2,p3);
			// adjacent triangles
			adj_tri_ind = mCtx.TT(face, edgeId);	// id of triangle adjacent to ei edge of traingle j
			adj_edge_ind = mCtx.TTi(face, edgeId);	//id of edge TT(j,ei) that is adjacent with triangle i

			RowVector3i adjF = F.row(adj_tri_ind);
			if (adj_edge_ind == 0) {
				p1 = V.row(adjF(0));
				p2 = V.row(adjF(1));
				p3 = V.row(adjF(2));
			}
			else if (adj_edge_ind == 1) {
				p1 = V.row(adjF(1));
				p2 = V.row(adjF(2));
				p3 = V.row(adjF(0));
			}
			else if (adj_edge_ind == 2) {
				p1 = V.row(adjF(2));
				p2 = V.row(adjF(0));
				p3 = V.row(adjF(1));
			}

			alpha_ij = get_angle(p1,p3,p2);

			// cotan weight w_ij
			weight_ij = (1.0 / tan(alpha_ij) + 1 / tan(beta_ij))*(V.row(i) - V.row(preP)).norm();

			// fill laplacian matrix
			C.insert(i, preP) = weight_ij;
			wij_sum += weight_ij;
		}

		C.insert(i, i) = -wij_sum;

	}

	// compute laplacian operator
	lap_b = M_inv * C;

	return lap_b;
}



// Calculate non-uniform mean curvature
Eigen::MatrixXd cw2::nonUniformCurvature(const MyContext& mCtx) {

	if (one_ring_neighbor.size() < 1) updateOneRingNeighbor(mCtx.V, mCtx.F);

	//Eigen::MatrixXd N(V.rows(), 3);
	//igl::per_vertex_normals(V, F, N);

	if (cotan_laplace.rows() < 1)
		cotan_laplace = cotan_discretization(mCtx.V, mCtx.F, mCtx);


	// mean curvature
	// -2Hn = lap_b*x
	Eigen::VectorXd H = 0.5*(cotan_laplace*mCtx.V).rowwise().norm();

	// direction of curvature;
	for (int i = 0; i < mCtx.V.rows(); i++) {

		int sz = one_ring_neighbor[i].size();

		RowVector3d avg;
		avg.setZero();

		for (int j = 0; j < sz; j++) {
			RowVector3d currentRow = mCtx.V.row(one_ring_neighbor[i][j].first);
			avg += currentRow;
		}

		avg /= sz;

		//Dot product
		if (mCtx.VN.row(i).dot(avg - mCtx.V.row(i)) > 0) {
			H.row(i) = -H.row(i);
		}
	}

	// jet mean curvature
	Eigen::MatrixXd C(mCtx.F.rows(), 3);
	igl::jet(H, true, C);

	return C;
}


Eigen::MatrixXd cw2::eigenReconstruction(MatrixXd V, MatrixXi F, const MyContext& mCtx, int k) {
	// Laplace beltrami operator
	SparseMatrix<double> laplace(V.rows(), V.rows());

	if(cotan_laplace.rows()<1)
		cotan_laplace = cotan_discretization(V, F,mCtx);

	// compute k smallest eigenvectors
	// ref: https://spectralib.org/quick-start.html
	SparseGenMatProd<double> op(cotan_laplace);
	GenEigsSolver< double, SMALLEST_MAGN, SparseGenMatProd<double> > eigs(&op, k, 150);

	eigs.init();
	int nconv = eigs.compute();

	VectorXcd complex_evals;
	MatrixXcd complex_evecs;

	if (eigs.info() == SUCCESSFUL) {
		complex_evals = eigs.eigenvalues();
		complex_evecs = eigs.eigenvectors();
		std::cout << "Eigenvalues found:\n" << complex_evals << endl;
	}
	else {
		std::cout << "Failed in calculating eigen calues\n";
	}

	MatrixXd real_evecs(complex_evecs.rows(), complex_evecs.cols());
	for (int i = 0; i < complex_evecs.rows(); i++) {
		for (int j = 0; j < complex_evecs.cols(); j++) {
			real_evecs(i, j) = complex_evecs(i, j).real();
		}
	}
	
	MatrixXd recons_V(V.rows(), 3);
	recons_V.setZero();

	MatrixXd scaler(1, 3);
	for (int i = 0; i < real_evecs.cols(); i++) {
		scaler(0, 0) = (V.col(0).transpose()) * real_evecs.col(i);
		scaler(0, 1) = (V.col(1).transpose()) * real_evecs.col(i);
		scaler(0, 2) = (V.col(2).transpose()) * real_evecs.col(i);

		recons_V.col(0) = recons_V.col(0) + scaler(0, 0) * real_evecs.col(i);
		recons_V.col(1) = recons_V.col(1) + scaler(0, 1) * real_evecs.col(i);
		recons_V.col(2) = recons_V.col(2) + scaler(0, 2) * real_evecs.col(i);
	}

	return recons_V;
}



#endif