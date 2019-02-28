from copy import deepcopy
from graph_nets import Aggregation, EdgeUpdate, Graph, GlobalStateUpdate, NodeUpdate


class GNBlock:

    def __init__(self,
                 phi_e: EdgeUpdate, phi_v: NodeUpdate, phi_u: GlobalStateUpdate,
                 rho_ev: Aggregation, rho_vu: Aggregation, rho_eu: Aggregation) -> None:
        """
        Graph network block initialization function.
        :param phi_e: Edge update function
        :param phi_v: Node update function
        :param phi_u: Global state update function
        :param rho_ev: Edge aggregation function for nodes
        :param rho_vu: Node aggregation function for the global state
        :param rho_eu: Edge aggregation function for the global state
        """

        self.phi_e, self.phi_v, self.phi_u = phi_e, phi_v, phi_u
        self.rho_ev, self.rho_vu, self.rho_eu = rho_ev, rho_vu, rho_eu

    def __call__(self, g: Graph) -> Graph:
        """
        Converts a graph g=(u, V, E) into an updated graph g'=(u', V', E') by applying an edge block, a node block, and
        a global block.
        :param g: Input graph g=(u, V, E), will not be modified
        :return: Output graph g'=(u', V', E')
        """

        g = deepcopy(g)

        self._edge_block(g)
        self._node_block(g)
        self._global_block(g)

        return g

    def _edge_block(self, g: Graph) -> None:
        """
        Applies the edge update function to every edge in the graph.
        :param g: Graph to work on (edges will be modified)
        """

        for e in g.edges:
            e.attribute = self.phi_e(e=e.attribute,
                                     v_r=e.receiver.attribute,
                                     v_s=e.sender.attribute,
                                     u=g.attribute)

    def _node_block(self, g: Graph) -> None:
        """
        Applies the node update function to every node in the graph.
        :param g: Graph to work on (node attributes will be modified)
        """

        for v in g.nodes:
            # aggregate incoming edges (where v is e.r)
            aggr_e = self.rho_ev(list(v.receiving_edges))

            # update node
            v.attribute = self.phi_v(aggr_e=aggr_e,
                                     v=v.attribute,
                                     u=g.attribute)

    def _global_block(self, g: Graph) -> None:
        """
        Applies the global state update function to the graph's global state.
        :param g: Graph to work on (global state will be modified)
        """

        u = g.attribute

        # aggregate nodes and edges
        aggr_e = self.rho_eu(list(g.edges))
        aggr_v = self.rho_vu(list(g.nodes))

        # update global state (graph attribute)
        u_prime = self.phi_u(aggr_e=aggr_e, aggr_v=aggr_v, u=u)

        g.attribute = u_prime
