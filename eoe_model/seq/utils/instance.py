from typing import List


class Instance:
    """
    This class is the basic Instance for a datasample
    """

    def __init__(self, doc_id: int = None, input: List[str] = None, event: str = None, event_id: int = None, title: str = None, output: List[str] = None, target: List[str] = None) -> None:
        """
        Constructor for the instance.
        :param input: sentence containing the words
        :param output: a list of labels
        """
        self.doc_id = doc_id
        self.input = input
        self.event = event
        self.event_id = event_id
        self.title = title
        self.output = output
        self.sent_ids = None
        self.char_ids = None
        self.output_ids = None
        self.vec = None
        self.target = target

    def __len__(self):
        return len(self.input)