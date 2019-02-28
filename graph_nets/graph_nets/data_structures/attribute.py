class Attribute:

    def __init__(self, val: any = None) -> None:
        self.val = val

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Attribute):
            return True
        return self.val == o.val
