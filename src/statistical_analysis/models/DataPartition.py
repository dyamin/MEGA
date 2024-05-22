import inspect


class DataPartition:

    @staticmethod
    def make(partition_by: list):
        assert (len(partition_by) == 3), f'Must specify partition_by Session, Memory and Confidence'
        return DataPartition(session=partition_by[0], memory=partition_by[1], confidence=partition_by[2])

    def __init__(self, session: str, memory: str, confidence: str):
        self.ses = self.__assert_attribute_value('session', session, ['A', 'B'])
        self.mem = self.__assert_attribute_value('memory', memory, ['Y', 'N'])
        self.conf = self.__assert_attribute_value('confidence', confidence, ['H', 'L'])
        return

    def get_splitter_arguments(self) -> list:
        non_routine_attributes = inspect.getmembers(self, lambda att: not (inspect.isroutine(att)))
        members = dict(att for att in non_routine_attributes if not (att[0].startswith('__') and att[0].endswith('__')))
        return [k for (k, v) in members.items() if v is not None]

    @staticmethod
    def __assert_attribute_value(attribute: str, given_value: str, acceptable_values: list) -> str or None:
        if (given_value is None) or (given_value == ''):
            return None
        matching = list(filter(lambda acceptable: acceptable.lower() == given_value.lower(), acceptable_values))
        if len(matching):
            return matching[0]
        raise AssertionError(
            f'Invalid value for level {attribute.capitalize()}:\n\tmust be one of {acceptable_values}, you provided {given_value}.')
