#include "Environment.h"

#include <fstream>
#include <iostream>
#include "Boxel.h"

Environment::Environment(){

}

void Environment::init(){

	//PHYSICS
	physics_handler = PhysicsEngine();
	physics_handler.init(
		&points_pos, &points_pos_last, &points_vel, &points_vel_true, &points_mass, &points_fixed,
		&a_index, &b_index, &length_eq, &length_eq_goal, &init_length_eq, &spring_const,
		&edges, &objects);
	physics_handler.point_is_colliding = &point_is_colliding;

	num_points = 0;
}

void Environment::set_surface_edge_color(int color) {
	
	for (int i = 0; i < objects.size(); i++) {
		for (int j = 0; j < objects[i]->surface_edges_index.cols(); j++) {
			edges[objects[i]->surface_edges_index[j]].color = color;
			edges[objects[i]->surface_edges_index[j]].isOnSurface = true;
		}
	}
}

//int Environment::createPoint(Vector2d_old pos, double mass) {
//
//	points_pos_old.push_back(pos);
//	points_vel_old.push_back(Vector2d_old(0, 0));
//	points_mass_old.push_back(mass);
//	points_fixed_old.push_back(false);
//
//	return points_pos_old.size() - 1;
//}

void Environment::create_points(vector <Vector2d>* pos, vector <Vector2d>* vel, vector <double>* mass, vector <bool>* fixed) {
	
	int old_size = num_points;
	int new_size = pos->size() + old_size;

	points_pos.conservativeResize(2, new_size);
	points_pos_last.conservativeResize(2, new_size);
	points_vel.conservativeResize(2, new_size);
	points_vel_true.conservativeResize(2, new_size);
	points_mass.conservativeResize(2, new_size);
	points_fixed.conservativeResize(2, new_size);
	
	for (int i = 0; i < pos->size(); i++) {
		points_pos.col(i + old_size) = pos->at(i);
		points_pos_last .col(i + old_size) = pos->at(i);
		points_vel.col(i + old_size) = vel->at(i);
		points_vel_true.col(i + old_size) = vel->at(i);
		points_mass.col(i + old_size) = Vector2d(mass->at(i), mass->at(i));
		points_fixed(0, i + old_size) = fixed->at(i);
		points_fixed(1, i + old_size) = fixed->at(i);
	}

	// for (int i = 0; i < points_pos.rows(); i++){
	// 	for(int j = 0; j < points_pos.cols(); j++){
	// 		if (points_fixed(i, j)){
	// 			continue;
	// 		}
		
	// 		double randn = ((double)rand() / (RAND_MAX))/ 1e7;
	// 		for (int f = 0; f < 12; f++){
	// 			randn = 2.0 * ((double)rand() / (RAND_MAX)) / 1e7;
	// 			randn = randn - 1e-7;
	// 		}
		
	// 		double old = points_pos(i, j); 
	// 		points_pos(i, j) = points_pos(i, j) + randn;
	// 		// cout << i << " " << j << " " << old << " " << randn << " " << points_pos(i, j) << "\n"; 
	// 	}
	// }
	

	num_points = new_size; 
}

void Environment::create_edges(vector <Edge>* new_edges) {

	int old_size = a_index.cols();
	int new_size = new_edges->size() + old_size;

	a_index.conservativeResize(1, new_size);
	b_index.conservativeResize(1, new_size);
	length_eq.conservativeResize(1, new_size);
	length_eq_goal.conservativeResize(1, new_size);
	init_length_eq.conservativeResize(1, new_size);
	spring_const.conservativeResize(1, new_size);

	for (int i = 0; i < new_edges->size(); i++) {
		edges.push_back(new_edges->at(i));
		a_index[i + old_size] = new_edges->at(i).a_index;
		b_index[i + old_size] = new_edges->at(i).b_index;
		length_eq[i + old_size] = new_edges->at(i).init_length_eq;
		length_eq_goal[i + old_size] = new_edges->at(i).init_length_eq;
		init_length_eq[i + old_size] = new_edges->at(i).init_length_eq;
		spring_const[i + old_size] = new_edges->at(i).spring_const;
	}
}

void Environment::swap_edge(int edge_index) {
	edges[edge_index].a_index = b_index[edge_index];
	edges[edge_index].b_index = a_index[edge_index];
	a_index[edge_index] = edges[edge_index].a_index;
	b_index[edge_index] = edges[edge_index].b_index;
}

//int Environment::create_edge(Edge edge) {
//
//	edges.push_back(edge);
//	return edges.size() - 1;
//}

void Environment::init_robot(string robot_name) {
	
	if (object_name_to_index.count(robot_name) <= 0)
	{
		cout << "Error: No robot named " << robot_name << ".\n";
		return;
	}
	if (!objects[object_name_to_index[robot_name]]->is_robot)
	{
		cout << "Error: " << robot_name << " is not robot.\n";
		return;
	}

	((Robot*)objects[object_name_to_index[robot_name]])->init();
}

bool Environment::add_object_name(string object_name, int index) {

	if (object_name_to_index.count(object_name) > 0)
		return false;

	object_name_to_index[object_name] = index;
	return true;
}

bool Environment::special_step() {
	physics_handler.resolve_edge_constraints();
	return false;
}
void Environment::print_poses(){
	cout << "pos\nnp.array([";
	for (int i = 0; i < points_pos.cols(); i++){
		cout << "[" << points_pos(0, i) << ", " << points_pos(1, i) << "],"; 
	}
	cout << "])\n\n";
	cout << "vel\nnp.array([";
	for (int i = 0; i < points_vel.cols(); i++){
		cout << "[" << points_vel(0, i) << ", " << points_vel(1, i) << "],"; 
	}
	cout << "])";
	cout << "\n\n\n\n";
}
bool Environment::step() {


	// for (int i = 0; i < points_pos.rows(); i++){
	// 	for(int j = 0; j < points_pos.cols(); j++){
	// 		if (points_fixed(i, j)){
	// 			continue;
	// 		}
			
	// 		double randn = ((double)rand() / (RAND_MAX))/ 1e7;
	// 		for (int f = 0; f < 28; f++)
	// 			randn = ((double)rand() / (RAND_MAX))/ 1e7; 
			
	// 		double old = points_pos(i, j); 
	// 		points_pos(i, j) = points_pos(i, j) + randn;
	// 		// cout << i << " " << j << " " << old << " " << randn << " " << points_pos(i, j) << "\n"; 
	// 	}
	// }
	// cout << "\n\n\n\n";

	// print_poses();
	//physics_handler.apply_external_force(); //在一次步进中施加外部力
	physics_handler.resolve_collisions();   //调用物理处理器的碰撞解决方法，处理当前状态下的所有碰撞。
	physics_handler.update_actuators();    //更新与物理系统交互的执行器状态。
	// print_poses();
	physics_handler.step_rk4(0.0001);   //使用四阶龙格-库塔法（RK4）进行物理状态的时间积分，时间步长为 0.0001。
	// print_poses();
	
	//physics_handler.step_rk4(0.00000005);
	//physics_handler.step_rk4(0.00001);
	physics_handler.resolve_edge_constraints(); //解决边缘约束，确保物理对象之间的约束关系得到满足。
	physics_handler.update_true_vel(0.0001); //更新物理对象的真实速度，以反映当前状态。




	return physics_handler.is_any_robot_self_colliding();  //返回是否有机器人自我碰撞的状态。

	//physics_handler.step_euler(0.0000005);

}

void Environment::set_robot_action(string robot_name, Ref <Matrix <double, Dynamic, 2>> action) {

	if (object_name_to_index.count(robot_name) <= 0)
	{
		cout << "Error: No robot named " << robot_name << ".\n";
		return;
	}
	if (!objects[object_name_to_index[robot_name]]->is_robot)
	{
		cout << "Error: " << robot_name << " is not robot.\n";
		return;
	}

	int robot_index = object_name_to_index[robot_name];

	vector <Boxel*> actuators;
	vector <double> actuations;

	for (int i = 0; i < action.rows(); i++) {

		if (((Robot*)objects[robot_index])->get_actuator(action(i, 0)) == NULL) {
			cout << "Error: No actuator found at index " << action(i, 0) << " of robot " << robot_name << ".\n";
			return;
		}

		actuators.push_back(((Robot*)objects[robot_index])->get_actuator(action(i, 0)));
		actuations.push_back(action(i, 1));
	}

	physics_handler.update_actuator_goals(&actuators, &actuations);
}

void Environment::save_snapshot(long int sim_time) {

	/*if (history.count(sim_time) > 0)
		return;*/

	history[sim_time] = Snapshot(points_pos, points_vel, sim_time);
}

bool Environment::revert_to_snapshot(long int sim_time) {

	if (history.count(sim_time) <= 0) {
		cout << "Error: Could not revert simulation - no save found for time argument.\n";
		return false;
	}
		
	points_pos = history[sim_time].points_pos;
	points_vel = history[sim_time].points_vel;

	points_pos_last = points_pos;
	points_vel_true = points_vel;

	history.erase(next(history.find(sim_time), 1), history.end());

	return true;
}

Ref <MatrixXd> Environment::get_pos_at_time(long int sim_time) {
	
	if (history.count(sim_time) <= 0) {
		Matrix <double, 2, Dynamic> empty;
		empty.resize(2, 4);
		empty << 0,0,0,0,0,0,0,0;
		return empty;
	}

	return history[sim_time].points_pos;
}
Ref <MatrixXd> Environment::get_vel_at_time(long int sim_time) {

	if (history.count(sim_time) <= 0) {
		Matrix <double, 2, Dynamic> empty;
		empty.resize(2, 4);
		empty << 0, 0, 0, 0, 0, 0, 0, 0;
		return empty;
	}

	return history[sim_time].points_vel;
}

Ref <MatrixXd> Environment::object_pos_at_time(long int sim_time, string object_name) {

	if (history.count(sim_time) <= 0 || object_name_to_index.count(object_name) <= 0) {
		Matrix <double, 2, Dynamic> empty;
		empty.resize(2, 4);
		empty << 0, 0, 0, 0, 0, 0, 0, 0;
		return empty;
	}

	int object_index = object_name_to_index[object_name];
	int min_index = objects[object_index]->min_point_index;
	int max_index = objects[object_index]->max_point_index;
	return history[sim_time].points_pos(Eigen::all, Eigen::seq(min_index, max_index));

}
Ref <MatrixXd> Environment::object_pos_at_time_matrix_for_robot(long int sim_time, string object_name) {

	if (history.count(sim_time) <= 0 || object_name_to_index.count(object_name) <= 0) {
		Matrix <double, 2, Dynamic> empty;
		empty.resize(2, 4);
		empty << 0, 0, 0, 0, 0, 0, 0, 0;
		return empty;
	}

	int object_index = object_name_to_index[object_name];
	int min_index = objects[object_index]->min_point_index;
	int max_index = objects[object_index]->max_point_index;
	auto matrix_2_mul_points = history[sim_time].points_pos(Eigen::all, Eigen::seq(min_index, max_index));

	// cout << matrix_2_mul_points << endl;
	// cout << matrix_2_mul_points.size() << endl;
	// cout << "id_2_xy size:" << idx_2_xy.size() << endl;
	// for(auto xy : idx_2_xy){
	// 	cout << xy.first << ' ' << xy.second << endl;
	// }
	// cout << endl;


	// Matrix <double, 2, Dynamic> ret;
	// ret.resize(2, world_grid_height * world_grid_width);

	// cout << "robot_grid_height = " << robot_grid_height << endl;
	// cout << "robot_grid_width = " << robot_grid_width << endl;

	// MatrixXd ret(2, robot_grid_height * robot_grid_width);
	MatrixXd ret = MatrixXd::Zero(2, robot_grid_height * robot_grid_width);

	// cout << "ret size:" << ret.size() << endl;
	// cout << ret << endl;

	// Matrix <double, world_grid_width, world_grid_height, 2> ret;

	// Matrix3d ret(world_grid_width, world_grid_height, 2);
	// cout << matrix_2_mul_points(Eigen::all, 0) << endl;
	// cout << ret(0, 0, Eigen::all) << endl;

	for(int idx = 0; idx < idx_2_xy.size(); idx++){
		auto [x, y] = idx_2_xy[idx];
		ret.col(y * robot_grid_width + x) = matrix_2_mul_points(Eigen::all, idx);
	}
	// cout << "ret size:" << ret.size() << endl;
	// cout << ret << endl;

	return ret;

}
Ref <MatrixXd> Environment::object_pos_at_time_matrix_for_robot_2(long int sim_time, string object_name) {

	if (history.count(sim_time) <= 0 || object_name_to_index.count(object_name) <= 0) {
		Matrix <double, 2, Dynamic> empty;
		empty.resize(2, 4);
		empty << 0, 0, 0, 0, 0, 0, 0, 0;
		return empty;
	}

	int object_index = object_name_to_index[object_name];
	int min_index = objects[object_index]->min_point_index;
	int max_index = objects[object_index]->max_point_index;
	auto matrix_2_mul_points = history[sim_time].points_pos(Eigen::all, Eigen::seq(min_index, max_index));

	// cout << "min_index = " << min_index << endl;
	// cout << "max_index = " << max_index << endl;
	// cout << "max_index - min_index = " << max_index - min_index << endl;
	// cout << "matrix_2_mul_points.size() = " << matrix_2_mul_points.size() << endl;
	// cout << "matrix_2_mul_points = " << matrix_2_mul_points << endl;
	// return matrix_2_mul_points;

	// MatrixXd ret(2, 4 * robot_grid_height_block * robot_grid_width_block);
	// MatrixXd ret = MatrixXd::Zero(2, 4 * robot_grid_height_block * robot_grid_width_block);

	// Matrix <double, 2, Dynamic> ret;

	// ret放到局部会导致矩阵第一个元素未定义，我也不知道为什么，这个bug我也是醉了，调了好久
	// 第一个元素未定义会导致有时变成inf！！
	// 有没有人对这个bug负下责啊

	// ret.resize(2, 4 * robot_grid_height_block * robot_grid_width_block);
	// for(int i = 0; i < 4 * robot_grid_height_block * robot_grid_width_block; i ++){
	// 	ret.col(i) = MatrixXd::Zero(2, 1);
	// }
	ret = MatrixXd::Zero(2, 4 * robot_grid_height_block * robot_grid_width_block);

	// cout << 4 * robot_grid_height_block * robot_grid_width_block << endl;
	for(auto iter : idx_2_xy_block){
		auto [x, y] = iter.first;
		auto idxs = iter.second;
		auto area = robot_grid_height_block * robot_grid_width_block;
		for(int i = 0; i < 4; i++){
			// cout << idxs[i] << ' ' << i * area + y * robot_grid_width_block + x << endl;

			ret.col(i * area + y * robot_grid_width_block + x) = matrix_2_mul_points(Eigen::all, idxs[i]);
			// 做了一堆无用的改动，发现问题不在这
			// ret(0, i * area + y * robot_grid_width_block + x) = matrix_2_mul_points(0, idxs[i]);
			// ret(1, i * area + y * robot_grid_width_block + x) = matrix_2_mul_points(1, idxs[i]);
		}
	}

	// cout << "ret = " << ret <<endl;

	// cout << "over" << endl;
	// return ret(Eigen::all, Eigen::seq(0, 4 * robot_grid_height_block * robot_grid_width_block - 1));
	return ret;

}
Ref <MatrixXd> Environment::object_vel_at_time(long int sim_time, string object_name) {

	if (history.count(sim_time) <= 0 || object_name_to_index.count(object_name) <= 0) {
		Matrix <double, 2, Dynamic> empty;
		empty.resize(2, 4);
		empty << 0, 0, 0, 0, 0, 0, 0, 0;
		return empty;
	}

	int object_index = object_name_to_index[object_name];
	int min_index = objects[object_index]->min_point_index;
	int max_index = objects[object_index]->max_point_index;
	return history[sim_time].points_vel(Eigen::all, Eigen::seq(min_index, max_index));

}double Environment::object_orientation_at_time(long int sim_time, string object_name) {

	if (history.count(sim_time) <= 0 || object_name_to_index.count(object_name) <= 0) {
		return 0;
	}

	int object_index = object_name_to_index[object_name];
	int min_index = objects[object_index]->min_point_index;
	int max_index = objects[object_index]->max_point_index;

	Matrix <double, 2, Dynamic> pos_new = history[sim_time].points_pos(Eigen::all, Eigen::seq(min_index, max_index));
	Matrix <double, 2, Dynamic> pos_old = history[0].points_pos(Eigen::all, Eigen::seq(min_index, max_index));

	//cout << "pos_new\n" << pos_new << "\n\n";
	//cout << "pos_old\n" << pos_old << "\n\n";

	Vector2d pos_new_com = pos_new.rowwise().sum() / pos_new.cols();
	Vector2d pos_old_com = pos_old.rowwise().sum() / pos_old.cols();

	//cout << "pos_new_com\n" << pos_new_com << "\n\n";
	//cout << "pos_old_com\n" << pos_old_com << "\n\n";

	pos_new = pos_new.colwise() - pos_new_com;
	pos_old = pos_old.colwise() - pos_old_com;

	//cout << "pos_new centered\n" << pos_new << "\n\n";
	//cout << "pos_old centered\n" << pos_old << "\n\n";

	Array <double, 1, Dynamic> mags_new = pos_new.colwise().norm();
	Array <double, 1, Dynamic> mags_old = pos_old.colwise().norm();

	pos_new = pos_new.colwise().normalized();
	pos_old = pos_old.colwise().normalized();

	//cout << "pos_new centered normalized\n" << pos_new << "\n\n";
	//cout << "pos_old centered normalized\n" << pos_old << "\n\n";

	Array <double, 1, Dynamic> angles = (pos_new.array() * pos_old.array()).colwise().sum();
	//cout << "dot\n" << angles << "\n\n";

	//cout << "full dot\n" << angles << "\n\n";
	angles = ((mags_new * mags_old) < 0.0000001).select(0.99999, angles);
	angles = (angles > 1.0).select(0.99999, angles);
	angles = (angles < -1.0).select(-0.99999, angles);

	//cout << "full dot after correction\n" << angles << "\n\n";

	//cout << "dot\n" << angles(Eigen::all, Eigen::seq(0, 16)) << "\n\n";
	angles = angles.acos();
	//cout << "acos\n" << angles << "\n\n";
	angles = angles * mags_old;

	double pi = 3.1415926;
	double average_angle = (angles.sum() / mags_old.sum());

	Array <double, 1, Dynamic> crosses = (pos_new(0, Eigen::all).array() * pos_old(1, Eigen::all).array()) - (pos_new(1, Eigen::all).array() * pos_old(0, Eigen::all).array());

	double cross = crosses.sum();

	if (cross > 0)
		average_angle = 2 * pi - average_angle;

	return average_angle;
}
void Environment::translate_object(double x, double y, string object_name) {
	
	if (object_name_to_index.count(object_name) <= 0)
		return;

	int object_index = object_name_to_index[object_name];
	int min_index = objects[object_index]->min_point_index;
	int max_index = objects[object_index]->max_point_index;

	Vector2d dx = Vector2d(x, y);
	points_pos(Eigen::all, Eigen::seq(min_index, max_index)).colwise() += dx;
}


Matrix<double, 2, Dynamic>* Environment::get_pos() {
	return &points_pos;
}
Matrix <double, 2, Dynamic>* Environment::get_vel() {
	return &points_vel;
}
Matrix <double, 2, Dynamic>* Environment::get_mass() {
	return &points_mass;
}
Matrix <bool, 2, Dynamic>* Environment::get_fixed() {
	return &points_fixed;
}

int Environment::get_num_points() {
	return num_points;
}
int Environment::get_num_edges() {
	return edges.size();
}

vector<Edge>* Environment::get_edges() {
	return &edges;
}
vector<SimObject*>* Environment::get_objects() {
	return &objects;
}
Robot* Environment::get_robot(string robot_name) {

	if (object_name_to_index.count(robot_name) <= 0)
	{
		cout << "Error: No robot named " << robot_name << ".\n";
		return NULL;
	}
	if (!objects[object_name_to_index[robot_name]]->is_robot)
	{
		cout << "Error: " << robot_name << " is not robot.\n";
		return NULL;
	}
	//return robots[robot_name_to_index[robot_name]];
	return (Robot*)(objects[object_name_to_index[robot_name]]);
}

//vector<Vector2d_old>* Environment::getPointsPos() {
//	return &points_pos_old;
//}
//vector<bool>* Environment::getPointsFixed() {
//	return &points_fixed_old;
//}
//vector<double>* Environment::getMasses() {
//	return &points_mass_old;
//}



Environment::~Environment(){

}
