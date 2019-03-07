from typing import Optional, Dict
from graph_nets import Attribute


class PageAttribute(Attribute):

    def __init__(self, base_url: str, client_status: Optional, https: bool, server_status: Optional[int],
                 start_node: bool, title: str) -> None:
        assert base_url is not None, "'base_url' must not be None"
        if client_status is None:
            client_status = ''
        if https is None:
            https = False
        if server_status is None:
            server_status = 0
        if start_node is None:
            start_node = False
        if title is None:
            title = ''
        val = {
            'base_url': base_url,
            'client_status': client_status,
            'https': https,
            'server_status': server_status,
            'start_node': start_node,
            'title': title,
        }
        super().__init__(val)

    @staticmethod
    def from_json(json: Dict) -> 'PageAttribute':
        return PageAttribute(
            json['base_url'],
            json['client_status'],
            json['https'],
            json['server_status'],
            json['startNode'],
            json['title']
        )


class LinkAttribute(Attribute):

    def __init__(self, val: str) -> None:
        super().__init__(val)
