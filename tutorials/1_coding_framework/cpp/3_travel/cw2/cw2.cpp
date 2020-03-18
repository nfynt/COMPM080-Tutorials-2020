//
// Coursework 2
// Shubham Singh
//

#define EIGEN_DONT_ALIGN_STATICALLY 

#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/jet.h>
#include <imgui/imgui.h>
#include <iostream>
#include <vector>
#include <ctime>

#include "discreteCurvature.hpp"

using namespace std;

enum Discretization { UNIFORM = 0, COTANGENT };

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
	std::vector<std::vector<int> > VF;
	//VI  #V list of lists of index of incidence within incident faces listed
	std::vector<std::vector<int> > VFi;

	//Default properties
	float  nv_len=0.5;	//normal vector length
	//float  point_size;	//vertex point size
	float  line_width;	//line width
	bool show_mesh=true;
	bool show_normals=false;
	bool show_wireframe=false;

	//Coursework params
	int eigen_ks;
	Discretization dType = UNIFORM;
};


MyContext g_myctx;

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{

	std::cout << "Key: " << key << " " << (unsigned int)key << std::endl;
	if (key == 'q' || key == 'Q')
	{
		exit(0);
	}
	return false;
}


void reset_display(igl::opengl::glfw::Viewer& viewer, MyContext & ctx)
{

	viewer.data().clear();

	//======================================================================
	// show mesh 
	if (ctx.show_mesh) {

		viewer.data().set_mesh(ctx.V, ctx.F);
		viewer.data().set_colors(ctx.C);
		viewer.core().align_camera_center(ctx.V, ctx.F);

	}
	//======================================================================


	// hide default wireframe
	//viewer.data().show_lines = 1;
	//viewer.data().show_overlay_depth = 1;
	//viewer.data().show_faces = 1;
	//======================================================================
	
	
	// show normal lines
	if (ctx.show_normals)
	{
		Eigen::MatrixXd EV1(ctx.V);
		Eigen::MatrixXd EV2;

		// show real VN
		EV2 = EV1 + ctx.VN * ctx.nv_len;

		//add_edges(viewer, EV1, EV2, Eigen::RowVector3d(1, 1, 1));
		viewer.data().add_edges(EV1, EV2, Eigen::RowVector3d(1, 1, 1));
		viewer.core().align_camera_center(ctx.V, ctx.F);
	}
	//======================================================================

	viewer.data().line_width = ctx.line_width;
	//viewer.data().point_size = ctx.point_size;

}

void initialize_viewer(igl::opengl::glfw::Viewer& viewer, igl::opengl::glfw::imgui::ImGuiMenu & menu)
{
	// Add additional windows via defining a Lambda expression with captures by reference([&])
	menu.callback_draw_custom_window = [&]()
	{
		bool require_reset = false;

		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(300, 560), ImGuiSetCond_FirstUseEver);
		ImGui::Begin("Laplacian Filtering", nullptr, ImGuiWindowFlags_NoSavedSettings);

		if (ImGui::CollapsingHeader("Basic Properties", false)) {
			// point size
			// [event handle] if value changed
			//if (ImGui::InputFloat("point_size", &g_myctx.point_size))
			//{
			//	std::cout << "point_size changed\n";
			//	viewer.data().point_size = g_myctx.point_size;
			//}

			// line width
			// [event handle] if value changed
			if (ImGui::InputFloat("line_width", &g_myctx.line_width))
			{
				std::cout << "line_width changed\n";
				viewer.data().line_width = g_myctx.line_width;
			}

			// length of normal line
			// [event handle] if value changed
			if (ImGui::SliderFloat("nv_length", &g_myctx.nv_len, 0, 1, "%.3f"))
			{
				require_reset = 1;
			}

			if (ImGui::Checkbox("show_mesh", &g_myctx.show_mesh))
			{
				require_reset = 1;
			}

			if (ImGui::Checkbox("show_normal", &g_myctx.show_normals))
			{
				require_reset = 1;
			}

			if (ImGui::Checkbox("Show wireframe", &g_myctx.show_wireframe))
			{
				//require_reset = 1;
				if (viewer.data().show_lines)
					viewer.data().show_lines = 0;
				else
					viewer.data().show_lines = 1;
			}
		}
		//End of basic properties

		//ImGui::Text("Type of Mesh");

		//const char* items[] = { "Cube", "Bunny", "Camel" };
		//static const char* m_item = NULL;
		//if (ImGui::BeginCombo("##combo", m_item)) {
		//	for (int n = 0; n < IM_ARRAYSIZE(items); n++)
		//	{
		//		bool is_selected = (m_item == items[n]);
		//		if (ImGui::Selectable(items[n], is_selected))
		//		{
		//			m_item = items[n];
		//			cout << m_item << " is selected" << endl;
		//		}
		//		if (is_selected)
		//			ImGui::SetItemDefaultFocus();
		//	}
		//	ImGui::EndCombo();
		//}

		if (ImGui::CollapsingHeader("Discrete Curvature and Spectral meshes", false))
		{
			ImGui::Text("Uniform Laplace:");
			if (ImGui::Button("Mean Curvature (H)", ImVec2(150, 50))) {
				cout << "Estimating Mean Curvature...\n";
				MatrixXd res = cw2::meanCurvature(g_myctx.V, g_myctx.F);
				g_myctx.C = res;
				require_reset = 1;
				cout << "Done\n";
			}

			if (ImGui::Button("Gaussian Curvature (K)", ImVec2(150, 50))) {
				cout << "Estimating Mean Curvature...\n";
				MatrixXd res = cw2::gaussCurvature(g_myctx.V, g_myctx.F);
				g_myctx.C = res;
				require_reset = 1;
				cout << "Done\n";
			}
		}

		if (ImGui::CollapsingHeader("Cotan Discretization", false))
		{
			if (ImGui::Button("Mean Curvature", ImVec2(100, 30))) {
				//discreteCurvature::mean_curvature
				cout << "Cotan Mean Curvature\n";
			}
		}

		if (ImGui::CollapsingHeader("Spectral reconstruction", false))
		{
			if (ImGui::InputInt("eigen length", &g_myctx.eigen_ks))
			{
				std::cout << "Eigen length changed to - "<<g_myctx.eigen_ks<<"\n";
			}

			if (ImGui::Button("Reconstruction", ImVec2(100, 30))) {
				//discreteCurvature::mean_curvature
				cout << "Eigen based reconstruction\n";
			}
		}

		if (ImGui::CollapsingHeader("Laplacian Mesh Smoothing"))
		{

			if (ImGui::Button("Implicit", ImVec2(100, 30))) {
				//discreteCurvature::mean_curvature
				cout << "Implicit lap smooth\n";
			}


			if (ImGui::Button("Explicit", ImVec2(100, 30))) {
				//discreteCurvature::mean_curvature
				cout << "Explicit lap smooth\n";
			}


			if (ImGui::Button("Add noise", ImVec2(100, 30))) {
				//discreteCurvature::mean_curvature
				cout << "Niose\n";
			}
		}



		if (require_reset)
		{
			reset_display(viewer, g_myctx);
		}

		ImGui::End();
	};


	// registered a event handler
	viewer.callback_key_down = &key_down;
}


int main(int argc, char* argv[])
{
	double lambda = 0.1;
	double noise = 0.1;
	int iteration = 1;
	int k = 1;


	//DEFINE HERE THE MESH TO BE READ!!!
	igl::readOBJ("../cw2_data/bumpy-cube.obj", g_myctx.V, g_myctx.F);

	//set color
	g_myctx.C = Eigen::MatrixXd(g_myctx.F.rows(), 3);
	g_myctx.C << Eigen::RowVector3d(0.9, 0.775, 0.25).replicate(g_myctx.F.rows(), 1);

	// calculate normal
	igl::per_vertex_normals(g_myctx.V, g_myctx.F, g_myctx.VN);

	// build adjacent matrix  
	//igl::vertex_triangle_adjacency(g_myctx.V.rows(), g_myctx.F, g_myctx.VF, g_myctx.VFi);

	//Calculate one ring neigbor
	cw2::updateOneRingNeighbor(g_myctx.V, g_myctx.F);

	igl::opengl::glfw::Viewer viewer;


	// Attach a default menu plugin
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	menu.callback_draw_viewer_menu = [&]()
	{
		menu.draw_viewer_menu();
	};

	// Add our GUI items
	initialize_viewer(viewer, menu);

	// set up initial display 
	reset_display(viewer, g_myctx);
	viewer.data().show_lines = 0;	//disable wireframe lines

	// Call GUI
	viewer.launch();


	return 0;
}





/*
 __  _ _____   ____  _ _____
|  \| | __\ `v' /  \| |_   _|
| | ' | _| `. .'| | ' | | |
|_|\__|_|   !_! |_|\__| |_|

*/
