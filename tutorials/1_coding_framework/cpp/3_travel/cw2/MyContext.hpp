#pragma once


#include <Eigen/src/Core/Matrix.h>
#include <iostream>
#include <vector>

using namespace std;


class MyContext
{
public:
	//magic Eigen3 macro : https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		//vertices
		Eigen::MatrixXd V;
	//vertex normals
	Eigen::MatrixXd VN;
	//faces
	Eigen::MatrixXi F;
	//face normals
	Eigen::MatrixXd FN;
	//face colors
	Eigen::MatrixXd C;
	//VF  #V list of lists of incident faces (adjacency list)
	//std::vector<std::vector<int> > VF;
//   TT   #F by #3 adjacent matrix, the element i,j is the id of the triangle
//   adjacent to the j edge of triangle i
	Eigen::MatrixXi TT;
	//VI  #V list of lists of index of incidence within incident faces listed
	//std::vector<std::vector<int> > VFi;
//   TTi  #F by #3 adjacent matrix, the element i,j is the id of edge of the
//   triangle TT(i,j) that is adjacent with triangle i
	Eigen::MatrixXi TTi;

	//Default properties
	float  nv_len = 0.5;	//normal vector length
	//float  point_size;	//vertex point size
	float  line_width;	//line width
	bool show_mesh = true;
	bool show_normals = false;
	bool show_wireframe = false;

	//Coursework params
	int eigen_ks;	//eigen first k vectors
	double lambda=0.1;		//smooth lambda
	int smooth_itr=5;	//smooth iterations
	double noise = 0.05;	//mesh noise level
};

