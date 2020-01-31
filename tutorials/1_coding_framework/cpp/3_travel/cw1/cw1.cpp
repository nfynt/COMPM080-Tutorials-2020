//
// Coursework 1
// Shubham Singh
//

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

	int num_vex;		//total number of vertices for current viewer
	int sel_vidx = 0;	//selected vertex index
	int noOfMesh;		//number of mesh to load (1-5)
	float nv_len;		//normal vector length
	float  point_size;	//point size
	float  line_width;	//line width

	bool show_mesh;
	bool show_normals;
	//bool show_colors;
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
};

MyContext g_myctx;


void update_context_mesh(MyContext& ctx, igl::opengl::glfw::Viewer& viewer)
{
	int vcnt = 0;
	int fcnt = 0;
	for (int i = 0; i < ctx.noOfMesh; i++)
	{
		vcnt += ctx.meshes[i].vertices.rows();
		fcnt += ctx.meshes[i].faces.rows();
		std::cout << "vcnt:" << vcnt << "\tfcnt:" << fcnt << std::endl;
	}
	std::cout << "Mesh size:" << ctx.meshes.size() << std::endl;
	ctx.V.resize(vcnt, 3);
	ctx.F.resize(fcnt, 3);
	ctx.C.resize(fcnt, 3);

	if (ctx.noOfMesh == 1)
	{
		ctx.V << ctx.meshes[0].vertices;
		ctx.F << ctx.meshes[0].faces;

		ctx.C << Colors::Yellow().replicate(ctx.meshes[0].faces.rows(), 1);
	}
	else if (ctx.noOfMesh == 2)
	{
		ctx.V << ctx.meshes[0].vertices, ctx.meshes[1].vertices;
		ctx.F << ctx.meshes[0].faces, (ctx.meshes[1].faces.array() + ctx.meshes[0].vertices.rows());

		ctx.C << Colors::Yellow().replicate(ctx.meshes[0].faces.rows(), 1),
			Colors::Red().replicate(ctx.meshes[1].faces.rows(), 1);
	}
	else if (ctx.noOfMesh == 3)
	{
		ctx.V << ctx.meshes[0].vertices, ctx.meshes[1].vertices, ctx.meshes[2].vertices;
		ctx.F << ctx.meshes[0].faces, 
			(ctx.meshes[1].faces.array() + ctx.meshes[0].vertices.rows()), 
			(ctx.meshes[2].faces.array() + ctx.meshes[0].vertices.rows() + ctx.meshes[1].vertices.rows());

		ctx.C << Colors::Yellow().replicate(ctx.meshes[0].faces.rows(), 1),
			Colors::Red().replicate(ctx.meshes[1].faces.rows(), 1),
			Colors::Green().replicate(ctx.meshes[2].faces.rows(), 1);
	}
	else if (ctx.noOfMesh == 4)
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
	else if (ctx.noOfMesh == 5)
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
	ctx.noOfMesh = 2;
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
	// color  #V|#F|1 by 3 list of colors
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
	//viewer.data().show_lines = 1;
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
		bool update_mesh = false;
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(300, 160), ImGuiSetCond_FirstUseEver);
		ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_NoSavedSettings);

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
		if (ImGui::SliderInt("no_of_mesh", &g_myctx.noOfMesh, 1, 5, "%.3f"))
		{
			require_reset = 1;
			update_mesh = true;
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

		/*if (ImGui::Checkbox("show_color", &g_myctx.show_colors))
		{
			require_reset = 1;
		}*/

		if (require_reset)
		{
			reset_display(viewer, g_myctx, update_mesh);
		}

		ImGui::End();
	};


	// registered a event handler
	viewer.callback_key_down = &key_down;
}


int main(int argc, char *argv[])
{
	std::vector<std::string> buuny_path{
		"../bunny_v2/bun000_v2.ply",
		"../bunny_v2/bun045_v2.ply",
		"../bunny_v2/bun090_v2.ply",
		"../bunny_v2/bun180_v2.ply",
		"../bunny_v2/bun270_v2.ply",
	};

	igl::opengl::glfw::Viewer viewer;

	for (const auto & name : buuny_path)
	{
		//viewer.load_mesh_from_file(std::string(name));
		Mesh m;
		igl::readPLY(name, m.vertices, m.faces);
		m.colors.resize(m.faces.rows(),3);
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
	reset_display(viewer, g_myctx,true);

	// Call GUI
	viewer.launch();
}