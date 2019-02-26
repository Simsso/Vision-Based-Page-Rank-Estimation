from typing import Optional, Dict
from rank_predictor.data_structures.attribute import Attribute


class PageAttributeVal:

    def __init__(self, base_url: str, client_status: Optional, https: bool, server_status: Optional[int],
                 start_node: bool, title: str) -> None:
        self.base_url = base_url
        self.client_status = client_status
        self.https = https
        self.server_status = server_status
        self.start_node = start_node
        self.title = title

    @staticmethod
    def from_json(json: Dict):
        return PageAttributeVal(
            json['base_url'],
            json['client_status'],
            json['https'],
            json['server_status'],
            json['startNode'],
            json['title']
        )


class PageAttribute(Attribute):

    def __init__(self, val: PageAttributeVal) -> None:
        super().__init__(val)


class LinkAttribute(Attribute):

    def __init__(self, val: str) -> None:
        super().__init__(val)
