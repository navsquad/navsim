import re


class InvalidConfigurationFormatting(Exception):
    def __init__(self, config_name: str):
        message = f"invalid configuration format in {config_name}."
        self.message = message
        super().__init__(self.message)


class EmptyRequiredConfigurationField(Exception):
    def __init__(self, class_type, field_name: str):
        class_name = re.findall("[A-Z][^A-Z]*", class_type.__name__)
        message = f"required {class_name[0]} field '{field_name}' is empty."
        self.message = message
        super().__init__(self.message)


class NonexistentTruthStates(Exception):
    def __init__(self):
        message = f"no truth states were generated. initialize truth states with simulate_truth() method."
        self.message = message
        super().__init__(self.message)
