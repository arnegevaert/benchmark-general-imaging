import yaml
from . import attribution
from inspect import signature, Parameter


class MethodLoader:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load_config(self, loc):
        with open(loc) as fp:
            methods = {}
            data = yaml.full_load(fp)
            wrappers = []
            if "wrappers" in data.keys():
                # Handle post-processing wrappers
                for wrapper in data["wrappers"].keys():
                    wrappers.append({
                        "constructor": getattr(attribution, wrapper),
                        "args": data["wrappers"][wrapper]
                    })
            if "methods" in data.keys():
                # Handle methods
                for entry in data["methods"]:
                    # Method entries must be either string, or dictionary with a single entry
                    # This corresponds to valid yaml list with optional nested parameters
                    if (type(entry) not in (str, dict)) or (type(entry) == dict and len(entry) > 1):
                        raise ValueError(f"Invalid configuration file")
                    method_name = entry if type(entry) == str else next(iter(entry))
                    method_type = entry[method_name]["type"] if type(entry) == dict else method_name
                    constructor = getattr(attribution, method_type)
                    method_args = entry[method_name] if type(entry) == dict else {}
                    method_args = {k: method_args[k] for k in method_args if k != "type"}
                    sig = signature(constructor)
                    args = {}
                    for param in sig.parameters.keys():
                        if param in self.kwargs:
                            args[param] = self.kwargs[param]
                        elif param in method_args:
                            args[param] = method_args[param]
                        elif sig.parameters[param].default == Parameter.empty:
                            # Default is empty so parameter is required
                            raise ValueError(f"Required parameter {param} for method {method_name} not found")
                    method_obj = constructor(**args)
                    for wrapper in wrappers:
                        method_obj = wrapper["constructor"](base_method=method_obj, **wrapper["args"])
                    methods[method_name] = method_obj
            else:
                raise ValueError(f"Invalid configuration file: file must contain key 'methods'")
            return methods
