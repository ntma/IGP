import psycopg2 as ps2
import numpy as np
import array
import time

#######################
# PostgresSQL wrapper #
#######################
class PGWrapper:
    def __init__(self):
        ''' Constructor for this class. '''
        self.conn = None
        self.curr = None
        self.conn_string = None
        self.overall_time = 0.0

    def connect_pg(self, path):
        """
        Connect to postgres
        :param path: path to the csv with the connection string
        :return: 0: success, -1 failed
        """

        self.read_conn_string(path)

        try:
            self.conn = ps2.connect(self.conn_string)
        except ps2.Error as e:
            print "I am unable to connect to the database"
            print e
            print e.pgcode
            print e.pgerror
            return False

        self.curr = self.conn.cursor()

        return True

    def execute_multiple_query(self, query, values):
        """
        Execute multiple queries for values
        :param query: query to execute multiple times
        :param values: list of values to process
        :return: rowcount
        """
        t_start = time.time()

        try:
            self.curr.executemany(query, values)
        except ps2.Error as e:
            print "Cant execute the query"
            print e
            print e.pgcode
            print e.pgerror
            return -1
        self.overall_time += time.time() - t_start

        return self.curr.rowcount

    def execute_query(self, query):
        """
        Execute a query
        :param query: query to execute
        :return: rowcount
        """

        t_start = time.time()

        try:
            self.curr.execute(query)
        except ps2.Error as e:
            print "Cant execute the query"
            print e
            print e.pgcode
            print e.pgerror

            return -1
        self.overall_time += time.time() - t_start

        return self.curr.rowcount

    def commit(self):
        """
        Commits the last transaction
        :return:
        """
        self.conn.commit()

    def roolback(self):
        """
        Rollback the last transaction
        :return:
        """
        self.conn.rollback()

    def fetch_all(self):
        """
        Fetch the results of the last query
        :return: dictionary with the result
        """

        t_start = time.time()
        a = self.curr.fetchall()
        self.overall_time += time.time() - t_start
        return a

    def fetch_one(self):
        """
        Fetch one result from the last query
        :return:
        """

        t_start = time.time()
        a = self.curr.fetchone()
        self.overall_time += time.time() - t_start
        return a

    def read_conn_string(self, path):
        """
        Reads the connection string from a csv file
        Format: host,post,database,user
        :param path: File path to csv
        :return:
        """

        f = open(path, "r")

        v = f.readline().split(',')

        f.close()

        self.conn_string = "host=" + v[0] + " port=" + v[1] + " dbname=" + v[2] + " user=" + v[3]

    def get_overall_time(self):
        """
        Returns the overall time spent by this module
        :return:
        """
        return self.overall_time

    #################################################
    # Auxiliars for byte128 <-> float128 convertion #
    #################################################
    def float2binarystring(self, nparray):
        """
        Converts a 128float array to uchar (escaped bytes)
        :param nparray: 128float
        :return: binary string
        """

        nparray = np.floor(nparray * 512.0 + 0.5)

        l = nparray.astype(dtype=int).tolist()

        b = array.array('B', l).tostring()

        binstring = str(ps2.Binary(b))[1:-8]
        binstring = binstring.replace("''", "\'")

        return binstring

    def binarystring2float(self, b):
        """
        Converts binary string to 128float
        :param b: binary string
        :return: 128float numpy array
        """
        return np.array(bytearray(b)[:], dtype=float) / 512.0

    def float128tostring(self, d):
        """
        Converts a 128float array to postgresql float ARRAY[128] string
        :param d: 128float
        :return: float ARRAY[128] string
        """
        return "ARRAY%s" % str(d.tolist())
