import heapq as hq
from itertools import count


#########################################
# Class to hold priority queue elements #
#########################################
class PQElem(object):
    __slots__ = ('cost', 'tiebreaker', 'wid', 'idx2d', 'idx3d', 'mode')


###############################################
# Class to hold a priority queue (heap style) #
###############################################
class PriorityQueue:

    def __init__(self):

        # Priority queue list
        self.pqueue = None

        # To support ordering in the priority queue
        self.tiebreaker = count()

        self.overall_time = 0.0

    def set_queue(self, list_head):
        """
        Set a priority queue from a list
        :param list_head:
        :return:
        """

        self.pqueue = list_head

        hq.heapify(self.pqueue)

    # Pop and return priority queue head
    def get_head(self):
        """
        Pop and return queue head
        :return: queue head
        """
        a = hq.heappop(self.pqueue)

        return a

    # Insert element into the priority queue
    def add_element(self, c, wid, p2d, p3d, m):
        """
        Add element to queue
        :param e: element to add
        :return:
        """

        # Create the tuple and push to heap
        hq.heappush(self.pqueue, (c, next(self.tiebreaker), wid, p2d, p3d, m))
