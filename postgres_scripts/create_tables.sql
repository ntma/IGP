CREATE EXTENSION postgis;

/*
DROP TABLE inverted_list;
DROP TABLE points3d;
DROP TABLE inverted_costs;
DROP TABLE visibility_graph;
*/

-- Contains all the 3D points from the point cloud
CREATE  TABLE points3d(
	id int,          -- point id
	x float,         -- x coordinate
	y float,         -- y coordinate
	z float,         -- z coordinate
	pos_3d geometry  -- geometry position
);

-- Contains the relationship between 3D points and their fine/coarse visual words
CREATE TABLE inverted_list(
	pt3d_id int,         -- 3D point id
	wid_fine int,        -- fine word id
	wid_coarse_lvl2 int, -- coarse word id at level 2
	wid_coarse_lvl3 int, -- coarse word id at level 3
	m_descriptor bytea   -- bytea containing the 128 binary descriptors
);

-- Contains the costs of search on each fine word
CREATE TABLE inverted_costs(
	wid int,          -- fine word id
	search_cost float -- related search cost
);

-- Contains the visibility graph build by clustering K nearest cameras
CREATE TABLE visibility_graph(
	pt3d_id int,   -- 3D point id
	cluster_id int -- related cluster id
);
