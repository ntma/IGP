------------------
-- C Extensions --
------------------
CREATE OR REPLACE FUNCTION float_euclidean_distance(float4 ARRAY[128], bytea) RETURNS float4
     AS '$libdir/PGSEDistance','sed_float_to_bytea'
     LANGUAGE C IMMUTABLE STRICT;

CREATE OR REPLACE FUNCTION int_euclidean_distance(int ARRAY[128], bytea) RETURNS int
     AS '$libdir/PGSEDistance','sed_int_to_bytea'
     LANGUAGE C IMMUTABLE STRICT;

CREATE OR REPLACE FUNCTION toint128(bytea) RETURNS int[128]
     AS '$libdir/PGSEDistance','bytea_to_int128'
     LANGUAGE C IMMUTABLE STRICT;

----------------------
-- ATOMIC FUNCTIONS --
----------------------

-- points3d
CREATE OR REPLACE FUNCTION get_3d_position(int)
RETURNS geometry
AS 'SELECT pos_3d
    FROM points3d
    WHERE id = $1;'
LANGUAGE SQL STABLE STRICT;

CREATE OR REPLACE FUNCTION get_3d_xyz(int)
RETURNS TABLE(x float,y float,z float)
AS 'SELECT x,y,z
    FROM points3d
    WHERE id = $1;'
LANGUAGE SQL STABLE STRICT;

CREATE OR REPLACE FUNCTION get_3d_nn(geometry, int)
RETURNS TABLE(id integer)
AS 'SELECT points3d.id
    FROM points3d
    ORDER BY pos_3d <-> $1 LIMIT $2'
LANGUAGE SQL STABLE STRICT;

-- inverted_list
CREATE OR REPLACE FUNCTION get_fine_inverted_list(int)
RETURNS TABLE(id int, descriptor bytea)
AS 'SELECT pt3d_id, m_descriptor
    FROM inverted_list
    WHERE wid_fine = $1;'
LANGUAGE SQL STABLE STRICT;

CREATE OR REPLACE FUNCTION get_3d_descriptor_lvl3(int, int[])
RETURNS TABLE(descriptor bytea)
AS 'SELECT m_descriptor
    FROM inverted_list
    WHERE pt3d_id = $1 AND wid_coarse_lvl3 = ANY ($2);'
LANGUAGE SQL STABLE STRICT;

CREATE OR REPLACE FUNCTION get_3d_descriptor_lvl2(int, int[])
RETURNS TABLE(descriptor bytea)
AS 'SELECT m_descriptor
    FROM inverted_list
    WHERE pt3d_id = $1 AND wid_coarse_lvl2 = ANY ($2);'
LANGUAGE SQL STABLE STRICT;

CREATE OR REPLACE FUNCTION get_nn3d_coarseids(int)
RETURNS TABLE(id int, wid_coarse int)
AS 'SELECT pt3d_id, wid_coarse_lvl3
    FROM inverted_list
    WHERE pt3d_id = $1'
LANGUAGE SQL STABLE STRICT;

-- inverted_costs
CREATE OR REPLACE FUNCTION get_searchcost_by_wid(IN inwid int)
RETURNS float
AS 'SELECT search_cost
    FROM inverted_costs
    WHERE wid = inwid AND search_cost > 0;'
LANGUAGE sql STABLE STRICT;

-----------------------
-- COMPLEX FUNCTIONS --
-----------------------

CREATE OR REPLACE FUNCTION get_2nn_linear(float4[128], int)
RETURNS TABLE(pt_id int, distance float4)
AS 'SELECT pt3d_id, float_euclidean_distance($1, m_descriptor) as ed
    FROM inverted_list
    WHERE $2 = wid_fine
    ORDER BY ed ASC
    LIMIT 2;'
LANGUAGE SQL STABLE STRICT;

CREATE OR REPLACE FUNCTION get_2nn_linear_int(int[128], int)
RETURNS TABLE(pt_id int, distance int)
AS 'SELECT pt3d_id, int_euclidean_distance($1, x.m_descriptor) as ed
    FROM(SELECT pt3d_id,  m_descriptor
         FROM inverted_list
         WHERE $2 = wid_fine) x
    ORDER BY ed ASC
    LIMIT 2;'
LANGUAGE SQL STABLE STRICT;

CREATE OR REPLACE FUNCTION get_3d_nn_clustersets(geometry, int)
RETURNS TABLE(id int, cluster_id int)
AS 'SELECT nn3d.id, visibility_graph.cluster_id
    FROM visibility_graph, get_3d_nn($1, $2) as nn3d
    WHERE nn3d.id = visibility_graph.pt3d_id'
LANGUAGE SQL STABLE STRICT;

CREATE OR REPLACE FUNCTION get_vizclusters_from_3dpt(int)
RETURNS TABLE(cluster_id int)
AS 'SELECT visibility_graph.cluster_id
    FROM visibility_graph
    WHERE pt3d_id = $1'
LANGUAGE SQL STABLE STRICT;

CREATE OR REPLACE FUNCTION get_3d_nn_filtered_lvl3(IN inid int, IN inlimit int)
RETURNS TABLE(id int, wids int[])
AS 'SELECT id, array_agg(DISTINCT wid_coarse_lvl3) as wids
    FROM inverted_list,
        (SELECT DISTINCT id
         FROM
            (SELECT pos_3d
             FROM points3d
             WHERE id = $1) x,
             get_3d_nn_clustersets(x.pos_3d, $2)
         WHERE id != $1 AND cluster_id IN (SELECT get_vizclusters_from_3dpt($1))) y
    WHERE y.id = inverted_list.pt3d_id
    GROUP BY id'
LANGUAGE sql IMMUTABLE STRICT;

CREATE OR REPLACE FUNCTION get_3d_nn_filtered_lvl2(IN inid int, IN inlimit int)
RETURNS TABLE(id int, wids int[])
AS 'SELECT id, array_agg(DISTINCT wid_coarse_lvl2) as wids
    FROM inverted_list,
        (SELECT DISTINCT id
         FROM
            (SELECT pos_3d
             FROM points3d
             WHERE id = $1) x,
              get_3d_nn_clustersets(x.pos_3d, $2)
          WHERE id != $1 AND cluster_id IN (SELECT get_vizclusters_from_3dpt($1))) y
    WHERE y.id = inverted_list.pt3d_id
    GROUP BY id'
LANGUAGE sql IMMUTABLE STRICT;
