import argparse
import logging as lg
import struct
from numpy import uint8
from numpy import array as nparray

from src.core.pgres_wrapper import PGWrapper
from src.utils.converters import npint2pgbyte


pg_index_3did  = "CREATE INDEX points3d_id_idx on points3d(id);"
pg_index_3dpos = "CREATE INDEX points3d_geom_idx ON points3d USING gist(pos_3d);"

points3d_idxes = [pg_index_3did,
                  pg_index_3dpos]

pg_index_ilist_fine = "CREATE INDEX ilist_widfine_idx on inverted_list(wid_fine);"
pg_index_ilist_coarselvl2 = "CREATE INDEX ilist_widcoarse2_idx on inverted_list(wid_coarse_lvl2);"
pg_index_ilist_coarselvl3 = "CREATE INDEX ilist_widcoarse3_idx on inverted_list(wid_coarse_lvl3);"
pg_index_ilist_3did = "CREATE INDEX ilist_pt3d_idx on inverted_list(pt3d_id);"
pg_index_ilist_3d_coarse = "CREATE INDEX ilist_3did_widcoarse on inverted_list(pt3d_id, wid_coarse_lvl3);"
pg_index_ilist_2d_coarse = "CREATE INDEX ilist_2did_widcoarse on inverted_list(pt3d_id, wid_coarse_lvl2);"

ilist_idxes = [pg_index_ilist_fine,
               pg_index_ilist_coarselvl2,
               pg_index_ilist_coarselvl3,
               pg_index_ilist_3did,
               pg_index_ilist_3d_coarse,
               pg_index_ilist_2d_coarse]

pg_index_viz_3did = "CREATE INDEX viz_graph_pt3d_idx on visibility_graph(pt3d_id);"
pg_index_viz_clusterid = "CREATE INDEX viz_graph_cluster_id_idx on visibility_graph(cluster_id);"

vizgraph_idxes = [pg_index_viz_3did,
                  pg_index_viz_clusterid]

pg_index_ilist_wid = "CREATE INDEX icosts_id_idx on inverted_costs(wid);"

icosts_idxes = [pg_index_ilist_wid]


if __name__ == "__main__":
    # Set the logging module
    lg.basicConfig(format='%(message)s', level=lg.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Set the argument options
    parser.add_argument('-p', action='store', dest='CSV_PATH', help='CSVs path', default='')
    parser.add_argument('-k', action='store', dest='PGKEY_PATH', help='Postgres key path', default='pg_key.csv')

    arg_v = parser.parse_args()

    # Path to the dataset folder
    dst_path = arg_v.CSV_PATH
    pgkey_path = arg_v.PGKEY_PATH

    # Add / if not present
    if dst_path[-1] != '/':
        dst_path += '/'

    # Set the required paths to pre-process the dataset
    cameras_path = dst_path + "cameras.csv"
    points3d_path = dst_path + "points3d.csv"
    viewlist_path = dst_path + "viewlist.csv"
    vizgraph_path = dst_path + "viz_graph.csv"
    assignments_path = dst_path + "assignments.bin"

    pgman = PGWrapper()

    if not pgman.connect_pg(pgkey_path):
        lg.info("Could not connect to database...")

        exit(-1)

    lg.info("   Inserting 3d")

    pgman.curr.copy_from(open(points3d_path, "r"), 'points3d', columns=('id', 'x', 'y', 'z'))

    pgman.commit()

    pgman.execute_query("UPDATE points3d SET pos_3d = st_makepoint(x,y,z);")
    pgman.execute_query("CREATE INDEX pos_3d_idx on points3d USING GIST ( pos_3d );")

    pgman.commit()

    # Create the indexes
    for q in points3d_idxes:
        pgman.execute_query(q)

    pgman.commit()

    lg.info("   Insert viz_graph")

    pgman.curr.copy_from(open(vizgraph_path, "r"), 'visibility_graph', columns=('pt3d_id', 'cluster_id'))

    pgman.commit()

    for q in vizgraph_idxes:
        pgman.execute_query(q)

    pgman.commit()

    lg.info("   Insert mean per visual word descriptors")
    with open(assignments_path, "rb") as f:

        stop = False

        to_insert = []
        ti_size = 0
        curr_idx = 0

        while not stop:
            eof = f.read(4)

            if not eof:
                break

            n_descriptors = struct.unpack("i", eof)[0]

            for i in xrange(n_descriptors):

                pt3d_id = struct.unpack("i", f.read(4))[0]
                fine_wid = struct.unpack("i", f.read(4))[0]
                lvl2_wid = struct.unpack("i", f.read(4))[0]
                lvl3_wid = struct.unpack("i", f.read(4))[0]
                m_descriptor = npint2pgbyte(nparray(struct.unpack("B"*128, f.read(128)), dtype=uint8))

                to_insert.append([pt3d_id, fine_wid, lvl2_wid, lvl3_wid, m_descriptor])

                ti_size += 1
            curr_idx += 1

            if ti_size >= 5000:
                pgman.execute_multiple_query(
                    "INSERT INTO inverted_list(pt3d_id,wid_fine,wid_coarse_lvl2, wid_coarse_lvl3,m_descriptor) VALUES(%s,%s,%s,%s,%s);",
                    to_insert)
                to_insert = []
                ti_size = 0
                lg.debug("  ongoing point: " + str(curr_idx))

        if ti_size > 0:
            pgman.execute_multiple_query(
                "INSERT INTO inverted_list(pt3d_id,wid_fine,wid_coarse_lvl2, wid_coarse_lvl3,m_descriptor) VALUES(%s,%s,%s,%s,%s);",
                to_insert)

            lg.debug("  ongoing point: " + str(curr_idx))

    f.close()

    for q in ilist_idxes:
        pgman.execute_query(q)

    pgman.commit()

    lg.info("   Computing fine word costs")

    pgman.execute_query("INSERT INTO inverted_costs "
                         "SELECT wid_fine as w, count(pt3d_id) c "
                         "FROM inverted_list GROUP BY wid_fine ORDER BY wid_fine ASC;")

    pgman.commit()

    lg.info("   Filling empty word costs")

    pgman.execute_query("INSERT INTO inverted_costs "
                         "SELECT sid, 0.0::float as c "
                         "FROM (SELECT sid "
                         "FROM generate_series(0, 99999) AS s(sid)) sub "
                         "WHERE not exists (SELECT 1 "
                         "FROM inverted_costs "
                         "WHERE wid = sub.sid);")

    pgman.commit()

    for q in icosts_idxes:
        pgman.execute_query(q)

    pgman.commit()

    lg.info("...done.")
