from copy import deepcopy
from typing import Optional, Tuple
from torch import nn
from graph_nets import Aggregation, EdgeUpdate, Graph, GlobalStateUpdate, NodeUpdate, IndependentEdgeUpdate, \
    IdentityNodeUpdate, IdentityGlobalStateUpdate, IdentityEdgeUpdate, IndependentNodeUpdate, \
    IndependentGlobalStateUpdate, ConstantAggregation


class GNBlock(nn.Module):

    def __init__(self, phi_e: EdgeUpdate, phi_v: NodeUpdate, phi_u: GlobalStateUpdate, rho_ev: Aggregation,
                 rho_vu: Aggregation, rho_eu: Aggregation) -> None:
        """
        Graph network block initialization function.
        :param phi_e: Edge update function
        :param phi_v: Node update function
        :param phi_u: Global state update function
        :param rho_ev: Edge aggregation function for nodes
        :param rho_vu: Node aggregation function for the global state
        :param rho_eu: Edge aggregation function for the global state
        """

        super().__init__()
        self.phi_e, self.phi_v, self.phi_u = phi_e, phi_v, phi_u
        self.rho_ev, self.rho_vu, self.rho_eu = rho_ev, rho_vu, rho_eu

    def forward(self, g: Graph) -> Graph:
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
            e.attr = self.phi_e(e=e.attr,
                                     v_r=e.receiver.attr,
                                     v_s=e.sender.attr,
                                     u=g.attr)

    def _node_block(self, g: Graph) -> None:
        """
        Applies the node update function to every node in the graph.
        :param g: Graph to work on (node attributes will be modified)
        """

        for v in g.nodes:
            # aggregate incoming edges (where v is e.r)
            aggr_e = self.rho_ev([e.attr for e in v.receiving_edges])

            # update node
            v.attr = self.phi_v(aggr_e=aggr_e,
                                     v=v.attr,
                                     u=g.attr)

    def _global_block(self, g: Graph) -> None:
        """
        Applies the global state update function to the graph's global state.
        :param g: Graph to work on (global state will be modified)
        """

        u = g.attr

        # aggregate nodes and edges
        aggr_e = self.rho_eu([e.attr for e in g.edges])
        aggr_v = self.rho_vu([n.attr for n in g.nodes])

        # update global state (graph attribute)
        u_prime = self.phi_u(aggr_e=aggr_e, aggr_v=aggr_v, u=u)

        g.attr = u_prime


LinearConfig = Tuple[int, int, bool]
OptLinearConfig = Optional[LinearConfig]


class LinearIndependentGNBlock(nn.Module):

    def __init__(self, e_config: OptLinearConfig = None, v_config: OptLinearConfig = None,
                 u_config: OptLinearConfig = None) -> None:
        super().__init__()

        phi_e = IdentityEdgeUpdate() if e_config is None else IndependentEdgeUpdate(nn.Linear(*e_config))
        phi_v = IdentityNodeUpdate() if v_config is None else IndependentNodeUpdate(nn.Linear(*v_config))
        phi_u = IdentityGlobalStateUpdate() if u_config is None else IndependentGlobalStateUpdate(nn.Linear(*u_config))

        self.block = GNBlock(
            phi_e, phi_v, phi_u,
            rho_ev=ConstantAggregation(),
            rho_vu=ConstantAggregation(),
            rho_eu=ConstantAggregation()
        )

    def forward(self, g: Graph) -> Graph:
        return self.block(g)


