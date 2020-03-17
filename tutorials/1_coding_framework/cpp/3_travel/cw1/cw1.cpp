//
// Coursework 1
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

#include "icp_point.hpp"
#include "icp_plane.hpp"
#include <ctime>

using namespace std;

//Mesh container for loaded objects
struct Mesh {
	Eigen::MatrixXd vertices;
	Eigen::MatrixXi faces;
	Eigen::MatrixXd colors;
	std::vector<Eigen::Vector3d> position;
	Mesh(Eigen::MatrixXd vert, Eigen::MatrixXi face) :vertices(vert), faces(face) {}
	Mesh(Eigen::MatrixXd vert, Eigen::MatrixXi face, Eigen::MatrixXd cols) :vertices(vert), faces(face), colors(cols) {}
	Mesh() {}
};

// Create a context object 
class MyContext
{
public:
	//magic Eigen3 macro : https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		// vertices n,3
		Eigen::MatrixXd V;
	// vertices normals 
	Eigen::MatrixXd VN;

	// faces k,3 
	Eigen::MatrixXi F;
	// face normals k,3 
	Eigen::MatrixXd FN;
	//vertex-face adjacency
	std::vector<std::vector<int> > VF;
	std::vector<std::vector<int> > VFi;

	//Mesh Color
	Eigen::MatrixXd C;

	//loaded meshes
	std::vector<Mesh> meshes;
	Mesh copy_mesh;		//for storing a copy of mesh - task2
	bool show_copy_mesh;

	int num_vex;		//total number of vertices for current viewer
	int sel_vidx = 0;	//selected vertex index
	int no_of_mesh;		//number of mesh to load (1-5)
	float nv_len;		//normal vector length
	float  point_size;	//point size
	float  line_width;	//line width

	bool show_mesh;		//show selected mesh
	bool show_normals;	//show mesh normals
	//bool show_colors;

	//ICP params
	int task_1_iterations;
	int task_2_iterations;
	int task_2_zrotation;
	float task_3_noise_level;
	int task_3_iterations;
	int sample_step;
	int task_4_iterations;
	int task_5_iterations;
	int task_6_iterations;
};

class Colors {
public:
	static Eigen::RowVector3d Red() {
		return Eigen::RowVector3d(0.9, 0.2, 0.2);
	}
	static Eigen::RowVector3d Green() {
		return Eigen::RowVector3d(0.2, 0.9, 0.2);
	}
	static Eigen::RowVector3d Blue() {
		return Eigen::RowVector3d(0.2, 0.2, 0.9);
	}
	static Eigen::RowVector3d Yellow() {
		return Eigen::RowVector3d(0.8, 0.8, 0.2);
	}
	static Eigen::RowVector3d Magenta() {
		return Eigen::RowVector3d(0.8, 0.2, 0.8);
	}
	static Eigen::RowVector3d Cyan() {
		return Eigen::RowVector3d(0.2, 0.8, 0.8);
	}
	static Eigen::RowVector3d Orange() {
		return Eigen::RowVector3d(0.9, 0.6, 0.1);
	}
};

MyContext g_myctx;


void update_context_mesh(MyContext& ctx, igl::opengl::glfw::Viewer& viewer)
{
	int vcnt = 0;
	int fcnt = 0;
	for (int i = 0; i < ctx.no_of_mesh; i++)
	{
		vcnt += ctx.meshes[i].vertices.rows();
		fcnt += ctx.meshes[i].faces.rows();
		std::cout << "vcnt:" << vcnt << "\tfcnt:" << fcnt << std::endl;
	}
	std::cout << "Mesh size:" << ctx.meshes.size() << std::endl;
	ctx.V.resize(vcnt, 3);
	ctx.F.resize(fcnt, 3);
	ctx.C.resize(fcnt, 3);

	if (ctx.no_of_mesh == 1)
	{
		ctx.V << ctx.meshes[0].vertices;
		ctx.F << ctx.meshes[0].faces;

		ctx.C << Colors::Yellow().replicate(ctx.meshes[0].faces.rows(), 1);
	}
	else if (ctx.no_of_mesh == 2)
	{
		ctx.V << ctx.meshes[0].vertices, ctx.meshes[1].vertices;
		ctx.F << ctx.meshes[0].faces, (ctx.meshes[1].faces.array() + ctx.meshes[0].vertices.rows());

		ctx.C << Colors::Yellow().replicate(ctx.meshes[0].faces.rows(), 1),
			Colors::Red().replicate(ctx.meshes[1].faces.rows(), 1);
	}
	else if (ctx.no_of_mesh == 3)
	{
		ctx.V << ctx.meshes[0].vertices, ctx.meshes[1].vertices, ctx.meshes[2].vertices;
		ctx.F << ctx.meshes[0].faces,
			(ctx.meshes[1].faces.array() + ctx.meshes[0].vertices.rows()),
			(ctx.meshes[2].faces.array() + ctx.meshes[0].vertices.rows() + ctx.meshes[1].vertices.rows());

		ctx.C << Colors::Yellow().replicate(ctx.meshes[0].faces.rows(), 1),
			Colors::Red().replicate(ctx.meshes[1].faces.rows(), 1),
			Colors::Green().replicate(ctx.meshes[2].faces.rows(), 1);
	}
	else if (ctx.no_of_mesh == 4)
	{
		ctx.V << ctx.meshes[0].vertices, ctx.meshes[1].vertices, ctx.meshes[2].vertices, ctx.meshes[3].vertices;
		ctx.F << ctx.meshes[0].faces,
			(ctx.meshes[1].faces.array() + ctx.meshes[0].vertices.rows()),
			(ctx.meshes[2].faces.array() + ctx.meshes[0].vertices.rows() + ctx.meshes[1].vertices.rows()),
			(ctx.meshes[3].faces.array() + ctx.meshes[0].vertices.rows() + ctx.meshes[1].vertices.rows() + ctx.meshes[2].vertices.rows());

		ctx.C << Colors::Yellow().replicate(ctx.meshes[0].faces.rows(), 1),
			Colors::Red().replicate(ctx.meshes[1].faces.rows(), 1),
			Colors::Green().replicate(ctx.meshes[2].faces.rows(), 1),
			Colors::Blue().replicate(ctx.meshes[3].faces.rows(), 1);
	}
	else if (ctx.no_of_mesh == 5)
	{
		ctx.V << ctx.meshes[0].vertices, ctx.meshes[1].vertices, ctx.meshes[2].vertices, ctx.meshes[3].vertices, ctx.meshes[4].vertices;
		ctx.F << ctx.meshes[0].faces,
			(ctx.meshes[1].faces.array() + ctx.meshes[0].vertices.rows()),
			(ctx.meshes[2].faces.array() + ctx.meshes[0].vertices.rows() + ctx.meshes[1].vertices.rows()),
			(ctx.meshes[3].faces.array() + ctx.meshes[0].vertices.rows() + ctx.meshes[1].vertices.rows() + ctx.meshes[2].vertices.rows()),
			(ctx.meshes[4].faces.array() + ctx.meshes[0].vertices.rows() + ctx.meshes[1].vertices.rows() + ctx.meshes[2].vertices.rows() + ctx.meshes[3].vertices.rows());

		ctx.C << Colors::Yellow().replicate(ctx.meshes[0].faces.rows(), 1),
			Colors::Red().replicate(ctx.meshes[1].faces.rows(), 1),
			Colors::Green().replicate(ctx.meshes[2].faces.rows(), 1),
			Colors::Blue().replicate(ctx.meshes[3].faces.rows(), 1),
			Colors::Magenta().replicate(ctx.meshes[4].faces.rows(), 1);
	}

	ctx.num_vex = vcnt;

	//Calculate normal
	igl::per_vertex_normals(ctx.V, ctx.F, ctx.VN);

	//build adjacent matrix
	igl::vertex_triangle_adjacency(ctx.V.rows(), ctx.F, ctx.VF, ctx.VFi);
}


void initial_context(MyContext & ctx, igl::opengl::glfw::Viewer& viewer)
{
	ctx.no_of_mesh = 2;
	ctx.show_normals = 0;
	ctx.show_mesh = 1;
	//ctx.show_colors = 0;
	ctx.point_size = 5;
	ctx.nv_len = 0.01;
	ctx.line_width = 1;
	ctx.sel_vidx = 0;
	std::cout << "My context properties:\n";

	update_context_mesh(ctx, viewer);
}

void add_points(igl::opengl::glfw::Viewer& viewer, Eigen::MatrixXd const  & pts_n3, Eigen::RowVector3d const  & color)
{
	//mark points 
	viewer.data().add_points(pts_n3, color);
}

void add_edges(igl::opengl::glfw::Viewer& viewer, Eigen::MatrixXd const &  p0, Eigen::MatrixXd const  & p1, Eigen::MatrixXd const &  color)
{
	viewer.data().add_edges(p0, p1, color);
}

void update_mesh(igl::opengl::glfw::Viewer& viewer, Eigen::MatrixXd const &  V, Eigen::MatrixXi const & F)
{
	//viewer.data().clear();
	viewer.data().set_mesh(V, F);
}

void update_mesh(igl::opengl::glfw::Viewer& viewer, Eigen::MatrixXd const &  V, Eigen::MatrixXi const & F, Eigen::MatrixXd const & color)
{
	// color  #V|#F|1x3 list of colors
	//viewer.data().clear();
	viewer.data().set_mesh(V, F);
	viewer.data().set_colors(color);
}

void reset_display(igl::opengl::glfw::Viewer& viewer, MyContext & ctx, bool updateMesh)
{
	if (updateMesh)
		update_context_mesh(ctx, viewer);

	viewer.data().clear();

	//======================================================================
	// show mesh 
	if (ctx.show_mesh) {

		//update_mesh(viewer, ctx.V, ctx.F);
		//Eigen::MatrixXd C(ctx.F.rows(),3);

		//viewer.data().set_colors(C);
		update_mesh(viewer, ctx.V, ctx.F, ctx.C);

		viewer.core().align_camera_center(ctx.V, ctx.F);

	}

	//======================================================================
	// hide default wireframe
	viewer.data().show_lines = 0;
	//viewer.data().show_overlay_depth = 1;
	//viewer.data().show_faces = 1;

	//======================================================================
	/*/ visualize adjacent vertices
	{
		Eigen::MatrixXd adj_vex;
		adj_vex.resize(ctx.VF[ctx.sel_vidx].size() * 2, 3);

		int count = 0;
		// get adjacent faces & vertices
		for (size_t i = 0; i < ctx.VF[ctx.sel_vidx].size(); i++)
		{
			int face_idx = ctx.VF[ctx.sel_vidx][i];
			int v_local_idx = ctx.VFi[ctx.sel_vidx][i];

			for (int iv = 0; iv < 3; iv++)
			{
				if (iv != v_local_idx)
				{
					Eigen::RowVector3d const vex = ctx.V.row(ctx.F(face_idx, iv));
					adj_vex.row(count) = vex;
					count++;
				}
			}
		}

		//add  points
		add_points(viewer, ctx.V.row(ctx.sel_vidx), Eigen::RowVector3d(1, 0, 0));

		add_points(viewer, adj_vex, Eigen::RowVector3d(1, 0, 0));

		// add links
		Eigen::MatrixXd EV1 = adj_vex;
		Eigen::MatrixXd EV2(adj_vex.rows(), 3);
		EV2.setZero();
		EV2 = EV2.rowwise() + ctx.V.row(ctx.sel_vidx);

		add_edges(viewer, EV1, EV2, Eigen::RowVector3d(1, 0, 0));

		if (!ctx.show_mesh && !ctx.show_normals) {
			viewer.core().align_camera_center(EV1);
		}
	}*/

	//======================================================================
	// add normal lines
	if (ctx.show_normals)
	{
		Eigen::MatrixXd EV1(ctx.V);
		Eigen::MatrixXd EV2;

		// show real VN
		EV2 = EV1 + ctx.VN * ctx.nv_len;

		add_edges(viewer, EV1, EV2, Eigen::RowVector3d(1, 1, 1));
		viewer.core().align_camera_center(ctx.V, ctx.F);
	}


	//======================================================================

	viewer.data().line_width = ctx.line_width;
	viewer.data().point_size = ctx.point_size;

}


bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{

	std::cout << "Key: " << key << " " << (unsigned int)key << std::endl;
	if (key == 'q' || key == 'Q')
	{
		exit(0);
	}
	return false;
}

void initial_viewer(igl::opengl::glfw::Viewer& viewer, igl::opengl::glfw::imgui::ImGuiMenu & menu)
{
	// Add additional windows via defining a Lambda expression with captures by reference([&])
	menu.callback_draw_custom_window = [&]()
	{
		bool require_reset = false;
		bool refresh_mesh = false;
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(300, 500), ImGuiSetCond_FirstUseEver);
		ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_NoSavedSettings);

		ImGui::Text("General Properties");

		if (ImGui::CollapsingHeader("General", false)) {

			// point size
			// [event handle] if value changed
			if (ImGui::InputFloat("point_size", &g_myctx.point_size))
			{
				std::cout << "point_size changed\n";
				viewer.data().point_size = g_myctx.point_size;
			}

			// line width
			// [event handle] if value changed
			if (ImGui::InputFloat("line_width", &g_myctx.line_width))
			{
				std::cout << "line_width changed\n";
				viewer.data().line_width = g_myctx.line_width;
			}

			// length of normal line
			// [event handle] if value changed
			//if (ImGui::InputFloat("nv_length", &g_myctx.nv_len))
			if (ImGui::SliderFloat("nv_length", &g_myctx.nv_len, 0, 1, "%.3f"))
			{
				require_reset = 1;
			}

			// Number of mesh to load
			// [event handle] if value changed
			//if (ImGui::InputFloat("nv_length", &g_myctx.nv_len))
			if (ImGui::SliderInt("no_of_mesh", &g_myctx.no_of_mesh, 1, 5, "%.3f"))
			{
				require_reset = 1;
				refresh_mesh = true;
			}

			// vertex index
			if (ImGui::SliderInt("sel_vex_index", &g_myctx.sel_vidx, 0, g_myctx.num_vex - 1))
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

		}


		ImGui::Spacing();

		ImGui::Text("Assignment Tasks");
		if (ImGui::CollapsingHeader("Task1: ICP (Req 2 mesh)") && g_myctx.no_of_mesh == 2)
		{
			if (ImGui::Button("Move M2 by 0.1", ImVec2(150, 20))) {
				//update m2 position
				if (g_myctx.no_of_mesh == 2)
				{
					Eigen::Vector3d pos;
					pos(0) = -0.1; pos(1) = 0.0; pos(2) = 0.0;
					g_myctx.meshes[1].vertices -= pos.replicate(1, g_myctx.meshes[1].vertices.rows()).transpose();
					
					g_myctx.V << g_myctx.meshes[0].vertices, g_myctx.meshes[1].vertices;
					require_reset = true;
					//refresh_mesh = true;
				}
			}
			ImGui::SameLine();
			if (ImGui::Button("Rotate M2 by 30deg", ImVec2(150, 20))) {
				//update m2 position
				Eigen::Matrix3d Rz;
				Rz << Eigen::AngleAxisd(30 * M_PI / 180, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

				g_myctx.meshes[1].vertices = g_myctx.meshes[1].vertices*Rz;
				
				g_myctx.V << g_myctx.meshes[0].vertices, g_myctx.meshes[1].vertices;
				require_reset = true;
				//refresh_mesh = true;
			}

			if (ImGui::InputInt("Iterations", &g_myctx.task_1_iterations))
			{
				std::cout << "Task 1: iterations = " << g_myctx.task_1_iterations << std::endl;
			}
			if (ImGui::Button("Point-to-Point ICP-alignment",ImVec2(200,40)))
			{
				std::cout << "Starting Point-to-point ICP alignment" << std::endl;
				
				double start_time = clock();
				Eigen::MatrixXd match(g_myctx.meshes[1].vertices.rows(), 3);
				pair<Eigen::Matrix3d, Eigen::Vector3d> RT;
				for (int i = 0; i < g_myctx.task_1_iterations; i++)
				{
					//ImGui::Text(to_string(i).c_str());

					icp::getCorrespondingPoints(g_myctx.meshes[1].vertices, g_myctx.meshes[0].vertices,match);
					icp::solveForRT(g_myctx.meshes[1].vertices, match, RT);

					if (icp::detectError(g_myctx.meshes[1].vertices, g_myctx.meshes[0].vertices, RT))
					{
						g_myctx.meshes[1].vertices = (g_myctx.meshes[1].vertices - RT.second.replicate(1, g_myctx.meshes[1].vertices.rows()).transpose())*RT.first;
					}

					//update mesh for each iteration
					//g_myctx.V << g_myctx.meshes[0].vertices, g_myctx.meshes[1].vertices;
					//update_mesh(viewer, g_myctx.V, g_myctx.F, g_myctx.C);
				}

				double time_cost = (clock() - start_time) / CLOCKS_PER_SEC;
				std::cout << "Iteration: " << g_myctx.task_1_iterations << endl;
				std::cout << "Time Cost: " << time_cost << endl;

				refresh_mesh = true;
				require_reset = true;
			}
		}
		ImGui::Spacing();
		if (ImGui::CollapsingHeader("Task2: localization test (Req 1 mesh)") && g_myctx.no_of_mesh==1)
		{
			if (!g_myctx.show_copy_mesh)
			{
				//initialise copy mesh
				g_myctx.copy_mesh.vertices.resize(g_myctx.meshes[0].vertices.rows(), 3);
				g_myctx.copy_mesh.vertices = g_myctx.meshes[0].vertices;
				g_myctx.copy_mesh.faces.resize(g_myctx.meshes[0].faces.rows(), 3);
				g_myctx.copy_mesh.faces = g_myctx.meshes[0].faces;
				g_myctx.copy_mesh.colors.resize(g_myctx.meshes[0].colors.rows(), 3);
				g_myctx.copy_mesh.colors = Colors::Orange().replicate(g_myctx.meshes[0].colors.rows(), 1);
				
				g_myctx.V.resize(g_myctx.meshes[0].vertices.rows() * 2, 3);
				g_myctx.F.resize(g_myctx.meshes[0].faces.rows() * 2, 3);
				g_myctx.C.resize(g_myctx.meshes[0].faces.rows() * 2, 3);
				g_myctx.V << g_myctx.meshes[0].vertices, g_myctx.copy_mesh.vertices;
				g_myctx.F << g_myctx.meshes[0].faces, (g_myctx.copy_mesh.faces.array() + g_myctx.meshes[0].vertices.rows());

				g_myctx.C << Colors::Yellow().replicate(g_myctx.meshes[0].faces.rows(), 1),
					Colors::Orange().replicate(g_myctx.copy_mesh.faces.rows(), 1);

				require_reset = 1;
				g_myctx.show_copy_mesh = true;
			}

			if (ImGui::Button("Rotate", ImVec2(80, 40)))
			{
				Eigen::Matrix3d Rz;
				Rz << Eigen::AngleAxisd(g_myctx.task_2_zrotation * M_PI / 180, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

				g_myctx.copy_mesh.vertices = g_myctx.copy_mesh.vertices*Rz;

				g_myctx.V.resize(g_myctx.meshes[0].vertices.rows() * 2, 3);
				g_myctx.V << g_myctx.meshes[0].vertices, g_myctx.copy_mesh.vertices;
				require_reset = true;
				//refresh_mesh = true;
			}
			ImGui::SameLine();
			if (ImGui::InputInt("Rot-Z", &g_myctx.task_2_zrotation,10))
			{
				std::cout << "Task 2: roation = " << g_myctx.task_2_zrotation << std::endl;
				
			}
			if (ImGui::InputInt("Iterations", &g_myctx.task_2_iterations,10))
			{
				std::cout << "Task 2: iterations = " << g_myctx.task_2_iterations << std::endl;
			}
			if (ImGui::Button("Point-to-Point ICP-alignment", ImVec2(200, 40)))
			{
				std::cout << "Starting Point-to-point ICP alignment" << std::endl;

				double start_time = clock();
				Eigen::MatrixXd match(g_myctx.copy_mesh.vertices.rows(), 3);
				pair<Eigen::Matrix3d, Eigen::Vector3d> RT;

				for (int i = 0; i < g_myctx.task_2_iterations; i++)
				{
					//std::cout << "Iteration: " << i << std::endl;

					icp::getCorrespondingPoints(g_myctx.copy_mesh.vertices, g_myctx.meshes[0].vertices,match);
					icp::solveForRT(g_myctx.copy_mesh.vertices, match, RT);

					if (icp::detectError(g_myctx.copy_mesh.vertices, g_myctx.meshes[0].vertices, RT))
					{
						g_myctx.copy_mesh.vertices = (g_myctx.copy_mesh.vertices - RT.second.replicate(1, g_myctx.copy_mesh.vertices.rows()).transpose())*RT.first;
					}

				}

				double time_cost = (clock() - start_time) / CLOCKS_PER_SEC;
				std::cout << "Iteration: " << g_myctx.task_2_iterations << endl;
				std::cout << "Time Cost: " << time_cost << endl;


				g_myctx.V.resize(g_myctx.meshes[0].vertices.rows() * 2, 3);
				g_myctx.F.resize(g_myctx.meshes[0].faces.rows() * 2, 3);
				g_myctx.V << g_myctx.meshes[0].vertices, g_myctx.copy_mesh.vertices;
				g_myctx.F << g_myctx.meshes[0].faces, (g_myctx.copy_mesh.faces.array() + g_myctx.meshes[0].vertices.rows());

				require_reset = true;
			}
		}
		else if (g_myctx.show_copy_mesh) 
		{

			g_myctx.V.resize(g_myctx.meshes[0].vertices.rows(), 3);
			g_myctx.F.resize(g_myctx.meshes[0].faces.rows(), 3);
			g_myctx.C.resize(g_myctx.meshes[0].faces.rows(), 3);
			g_myctx.V << g_myctx.meshes[0].vertices;
			g_myctx.F << g_myctx.meshes[0].faces;

			g_myctx.C << Colors::Yellow().replicate(g_myctx.meshes[0].faces.rows(), 1);

			g_myctx.show_copy_mesh = false;
		}//end of task 2
		ImGui::Spacing();
		if (ImGui::CollapsingHeader("Task3: Gaussian Noise (Req 2 mesh)") && g_myctx.no_of_mesh == 2)
		{
			if (ImGui::Button("Set", ImVec2(50, 20))) {
				icp::addGaussianNoise(g_myctx.meshes[1].vertices, g_myctx.task_3_noise_level);
				g_myctx.V << g_myctx.meshes[0].vertices, g_myctx.meshes[1].vertices;
				require_reset = true;
			}
			ImGui::SameLine();
			if (ImGui::InputFloat("Noise", &g_myctx.task_3_noise_level, 10))
			{
				std::cout << "Task 3: noise = " << g_myctx.task_3_noise_level << std::endl;
			}
			if (ImGui::InputInt("Iterations", &g_myctx.task_3_iterations, 10))
			{
				std::cout << "Task 3: iterations = " << g_myctx.task_3_iterations << std::endl;
			}
			if (ImGui::Button("Point-to-Point ICP-alignment", ImVec2(200, 40)))
			{
				std::cout << "Starting Point-to-point ICP alignment" << std::endl;

				double start_time = clock();
				Eigen::MatrixXd match(g_myctx.meshes[1].vertices.rows(), 3);
				pair<Eigen::Matrix3d, Eigen::Vector3d> RT;

				for (int i = 0; i < g_myctx.task_3_iterations; i++)
				{
					//std::cout << "Iteration: " << i << std::endl;

					icp::getCorrespondingPoints(g_myctx.meshes[1].vertices, g_myctx.meshes[0].vertices,match);
					icp::solveForRT(g_myctx.meshes[1].vertices, match,RT);

					if (icp::detectError(g_myctx.meshes[1].vertices, g_myctx.meshes[0].vertices, RT))
					{
						g_myctx.meshes[1].vertices = (g_myctx.meshes[1].vertices - RT.second.replicate(1, g_myctx.meshes[1].vertices.rows()).transpose())*RT.first;
					}

				}

				double time_cost = (clock() - start_time) / CLOCKS_PER_SEC;
				std::cout << "Iteration: " << g_myctx.task_3_iterations << endl;
				std::cout << "Time Cost: " << time_cost << endl;

				g_myctx.V << g_myctx.meshes[0].vertices, g_myctx.meshes[1].vertices;
				require_reset = true;
			}
		} //end of task 3
		ImGui::Spacing();

		if (ImGui::CollapsingHeader("Task4: Subsampling (Req 2 mesh)") && g_myctx.no_of_mesh == 2)
		{
			if (ImGui::InputInt("Sample step", &g_myctx.sample_step, 10))
			{
				std::cout << "Task 4: sample step = " << g_myctx.sample_step << std::endl;
			}
			if (ImGui::InputInt("Iterations", &g_myctx.task_4_iterations, 10))
			{
				std::cout << "Task 4: iterations = " << g_myctx.task_4_iterations << std::endl;
			}
			if (ImGui::Button("Point-to-Point ICP-alignment", ImVec2(200, 40)))
			{
				std::cout << "Starting Point-to-point ICP alignment" << std::endl;

				double start_time = clock();
				Eigen::MatrixXd sampledV;
				Eigen::MatrixXd match;
				pair<Eigen::Matrix3d, Eigen::Vector3d> RT;

				for (int i = 0; i < g_myctx.task_4_iterations; i++)
				{
					icp::getSampledSource(g_myctx.meshes[1].vertices, g_myctx.sample_step, sampledV);
					icp::getCorrespondingPoints(sampledV, g_myctx.meshes[0].vertices,match);
					icp::solveForRT(sampledV, match, RT);

					if (icp::detectError(g_myctx.meshes[1].vertices, g_myctx.meshes[0].vertices, RT))
					{
						g_myctx.meshes[1].vertices = (g_myctx.meshes[1].vertices - RT.second.replicate(1, g_myctx.meshes[1].vertices.rows()).transpose())*RT.first;
					}

				}

				double time_cost = (clock() - start_time) / CLOCKS_PER_SEC;
				std::cout << "Iteration: " << g_myctx.task_4_iterations << endl;
				std::cout << "Time Cost: " << time_cost << endl;

				g_myctx.V << g_myctx.meshes[0].vertices, g_myctx.meshes[1].vertices;
				require_reset = true;
			}
		} //end of task 4

		ImGui::Spacing();

		if (ImGui::CollapsingHeader("Task5: Multi-meshes (Req 5 mesh") && g_myctx.no_of_mesh == 5)
		{
			if (ImGui::InputInt("Iterations", &g_myctx.task_5_iterations))
			{
				cout << "Task 5 iterations:" << g_myctx.task_5_iterations << endl;
			}
			if (ImGui::Button("Point-to-point ICP",ImVec2(200,40))) {
				cout << "Starting ICP for aligning 5 meshes\n";
				
				double start_time = clock();
				//init 1st mesh
				/*g_myctx.V.resize(g_myctx.meshes[0].vertices.rows(), 3);
				g_myctx.V << g_myctx.meshes[0].vertices;
				g_myctx.F.resize(g_myctx.meshes[0].faces.rows(), 3);
				g_myctx.F << g_myctx.meshes[0].faces;
				g_myctx.C.resize(g_myctx.meshes[0].colors.rows(), 3);
				g_myctx.C << g_myctx.meshes[0].colors;*/

				int sample_step = 20;
				Eigen::MatrixXd sampledV;
				Eigen::MatrixXd match;
				pair<Eigen::Matrix3d, Eigen::Vector3d> RT;
				//align mesh1 -> mesh2
				for (int i = 0; i < g_myctx.task_5_iterations; i++)
				{
					icp::getSampledSource(g_myctx.meshes[0].vertices, sample_step,sampledV);
					icp::getCorrespondingPoints(sampledV, g_myctx.meshes[1].vertices,match);
					icp::solveForRT(sampledV, match, RT);

					if (icp::detectError(g_myctx.meshes[0].vertices, g_myctx.meshes[1].vertices, RT))
					{
						g_myctx.meshes[0].vertices = (g_myctx.meshes[0].vertices - RT.second.replicate(1, g_myctx.meshes[0].vertices.rows()).transpose())*RT.first;
					}
				}
				cout << "aligned mesh1 -> mesh2" << endl;
				//align mesh1,mesh2 -> mesh3
				for (int i = 0; i < g_myctx.task_5_iterations; i++)
				{
					icp::getSampledSource(g_myctx.meshes[1].vertices, sample_step,sampledV);
					//cout << g_myctx.meshes[1].vertices.rows()<<" "<<sampledV.rows()<<endl;
					icp::getCorrespondingPoints(sampledV, g_myctx.meshes[2].vertices,match);
					icp::solveForRT(sampledV, match, RT);

					if (icp::detectError(g_myctx.meshes[1].vertices, g_myctx.meshes[2].vertices, RT))
					{
						g_myctx.meshes[1].vertices = (g_myctx.meshes[1].vertices - RT.second.replicate(1, g_myctx.meshes[1].vertices.rows()).transpose())*RT.first;
						g_myctx.meshes[0].vertices = (g_myctx.meshes[0].vertices - RT.second.replicate(1, g_myctx.meshes[0].vertices.rows()).transpose())*RT.first;
					}
				}
				cout << "aligned mesh1,mesh2 -> mesh3" << endl;
				//align mesh1,mesh2,mesh3 -> mesh4
				for (int i = 0; i < g_myctx.task_5_iterations; i++)
				{
					icp::getSampledSource(g_myctx.meshes[2].vertices, sample_step, sampledV);
					//cout << g_myctx.meshes[2].vertices.rows() << " " << sampledV.rows() << endl;
					icp::getCorrespondingPoints(sampledV, g_myctx.meshes[3].vertices, match);
					icp::solveForRT(sampledV, match, RT);

					if (icp::detectError(g_myctx.meshes[2].vertices, g_myctx.meshes[3].vertices, RT))
					{
						g_myctx.meshes[2].vertices = (g_myctx.meshes[2].vertices - RT.second.replicate(1, g_myctx.meshes[2].vertices.rows()).transpose())*RT.first;
						g_myctx.meshes[1].vertices = (g_myctx.meshes[1].vertices - RT.second.replicate(1, g_myctx.meshes[1].vertices.rows()).transpose())*RT.first;
						g_myctx.meshes[0].vertices = (g_myctx.meshes[0].vertices - RT.second.replicate(1, g_myctx.meshes[0].vertices.rows()).transpose())*RT.first;
					}
				}
				cout << "aligned mesh1,mesh2,mesh3 -> mesh4" << endl;
				//align mesh1,mesh2,mesh3,mesh4 -> mesh5
				for (int i = 0; i < g_myctx.task_5_iterations; i++)
				{
					icp::getSampledSource(g_myctx.meshes[3].vertices, sample_step, sampledV);
					icp::getCorrespondingPoints(sampledV, g_myctx.meshes[4].vertices, match);
					icp::solveForRT(sampledV, match, RT);

					if (icp::detectError(g_myctx.meshes[3].vertices, g_myctx.meshes[4].vertices, RT))
					{
						g_myctx.meshes[3].vertices = (g_myctx.meshes[3].vertices - RT.second.replicate(1, g_myctx.meshes[3].vertices.rows()).transpose())*RT.first;
						g_myctx.meshes[2].vertices = (g_myctx.meshes[2].vertices - RT.second.replicate(1, g_myctx.meshes[2].vertices.rows()).transpose())*RT.first;
						g_myctx.meshes[1].vertices = (g_myctx.meshes[1].vertices - RT.second.replicate(1, g_myctx.meshes[1].vertices.rows()).transpose())*RT.first;
						g_myctx.meshes[0].vertices = (g_myctx.meshes[0].vertices - RT.second.replicate(1, g_myctx.meshes[0].vertices.rows()).transpose())*RT.first;
					}
				}
				cout << "aligned mesh1,mesh2,mesh3,mesh4 -> mesh5" << endl;

				double time_cost = (clock() - start_time) / CLOCKS_PER_SEC;
				std::cout << "Iterations: " << g_myctx.task_5_iterations << endl;
				std::cout << "Time Cost: " << time_cost << endl;

				require_reset = true;
				refresh_mesh = true;
			}//end icp button
		}//end task 5

		if (ImGui::CollapsingHeader("Task6: Point-to-plane ICP (Req 2 mesh") && g_myctx.no_of_mesh == 2) {
			if (ImGui::InputInt("Iterations", &g_myctx.task_6_iterations))
			{
				cout << "Task 6 iterations:" << g_myctx.task_6_iterations << endl;
			}
			if (ImGui::Button("Point-to-plane ICP",ImVec2(200,40))) {

				double start_time = clock();
				Eigen::MatrixXd normal;
				Eigen::MatrixXd matched;
				pair<Matrix3d, Vector3d> RT;
				for (int i = 0; i < g_myctx.task_6_iterations; i++) {
					// get normal
					icp::findNormal(g_myctx.meshes[1].vertices, g_myctx.meshes[0].vertices,normal);
					// get matched point set
					icp::getCorrespondingPoints(g_myctx.meshes[1].vertices, g_myctx.meshes[0].vertices,matched);
					icp::icpPointToPlane(g_myctx.meshes[1].vertices, matched, normal, RT);
					if (icp::detectError(g_myctx.meshes[1].vertices, g_myctx.meshes[0].vertices, RT)) {
						g_myctx.meshes[1].vertices = (g_myctx.meshes[1].vertices - RT.second.replicate(1, g_myctx.meshes[1].vertices.rows()).transpose())* RT.first;
					}
				}

				double time_cost = (clock() - start_time) / CLOCKS_PER_SEC;
				std::cout << "Iterations: " << g_myctx.task_6_iterations << endl;
				std::cout << "Time Cost: " << time_cost << endl;

				g_myctx.V << g_myctx.meshes[0].vertices, g_myctx.meshes[1].vertices;
				require_reset = true;
			}
		}//end of task 6

		if (require_reset)
		{
			reset_display(viewer, g_myctx, refresh_mesh);
		}

		ImGui::End();
	};

	// registered a event handler
	viewer.callback_key_down = &key_down;
}


int main(int argc, char *argv[])
{
	std::vector<std::string> buuny_path{
		"../bunny_v2/bun315_v2.ply",
		"../bunny_v2/bun270_v2.ply",
		"../bunny_v2/bun090_v2.ply",
		"../bunny_v2/bun180_v2.ply",
		"../bunny_v2/bun000_v2.ply"
		//"../bunny_v2/bun315_v2.ply"
	};

	igl::opengl::glfw::Viewer viewer;

	for (const auto & name : buuny_path)
	{
		//viewer.load_mesh_from_file(std::string(name));
		Mesh m;
		igl::readPLY(name, m.vertices, m.faces);
		m.colors.resize(m.faces.rows(), 3);
		m.colors << Colors::Yellow().replicate(m.faces.rows(), 1);
		g_myctx.meshes.push_back(m);
		std::cout << "Read file:" << name << std::endl;
	}

	/*/UV colors
	for (int i = 0; i < viewer.data_list.size(); i++)
	{
		Mesh m(viewer.data_list[i].V, viewer.data_list[i].F);
		// Use the z coordinate as a scalar field over the surface
		Eigen::VectorXd Z = viewer.data_list[i].V.col(2);
		// Compute per-vertex colors
		igl::jet(Z, true, m.colors);
	}*/

	//initialize inspector properties
	initial_context(g_myctx, viewer);

	// Attach a default menu plugin
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	menu.callback_draw_viewer_menu = [&]()
	{
		menu.draw_viewer_menu();
	};

	// Add our GUI items
	initial_viewer(viewer, menu);

	// set up initial display 
	reset_display(viewer, g_myctx, true);

	// Call GUI
	viewer.launch();
}







/*
 __  _ _____   ____  _ _____  
|  \| | __\ `v' /  \| |_   _| 
| | ' | _| `. .'| | ' | | |   
|_|\__|_|   !_! |_|\__| |_|
 
*/
