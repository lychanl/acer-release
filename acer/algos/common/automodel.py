from typing import Any, Callable, Collection, Dict, Iterable, Tuple


class AutoModelCyclicReferenceError(Exception):
    def __init__(self, cyclic_list) -> None:
        super().__init__(f"Possible AutoModel cyclic reference in ({', '.join(cyclic_list)})")


class AutoModelComponent:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.methods = {}
        self.targets = []

    def register_method(self, name: str, method: Callable, args: Dict[str, str]):
        self.methods[name] = (method, args)


class AutoModel:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._components = {}

    def register_component(self, name: str, component: AutoModelComponent):
        self._components[name] = component

    def prepare_default_call_list(self, data: Collection[str]) -> Tuple[Iterable[Tuple[str, Callable, Dict[str, str]]], Collection[str]]:
        targets = []
        for name, comp in self._components.items():
            targets.extend(f"{name}.{target}" for target in comp.targets)

        return self.prepare_call_list(targets, data)

    def prepare_call_list(self, to_call: Iterable[str], data: Collection[str]) -> Tuple[Iterable[Tuple[str, Callable, Dict[str, str]]], Collection[str]]:
        call_list = []
        ready_calls = set()
        required_data = set()

        while to_call:
            next_to_call = []
            changed = False

            for method in to_call:
                fun, args = self._get_method(method)
                args_ready = True

                for arg in args.values():
                    if arg in data:
                        required_data.add(arg)

                    elif arg not in ready_calls:
                        args_ready = False

                        if arg not in next_to_call and arg not in to_call:
                            next_to_call.append(arg)
                            changed = True

                if args_ready:
                    changed = True
                    ready_calls.add(method)
                    call_list.append((method, fun, args))
                elif method not in next_to_call:
                    next_to_call.append(method)

            if not changed:
                raise AutoModelCyclicReferenceError(next_to_call)

            to_call = next_to_call

        return call_list, list(required_data)

    def _get_method(self, method: str) -> Tuple[Callable, Dict[str, str]]:
        component_name, method_name = method.split('.')

        if component_name not in self._components:
            raise ValueError(f"Missing component {component_name} in {method}")
        componenet = self._components[component_name]

        if method_name not in componenet.methods:
            raise ValueError(f"Missing method {method_name} in {component_name}")

        return componenet.methods[method_name]

    def call_list(self, list: Iterable[Tuple[str, Callable, Dict[str, str]]], data: Dict[str, Any], preprocessing: Dict[str, Callable]):
        data_dict = {
            field: preprocessing[field](value) if field in preprocessing else value
            for field, value in data.items()
        }

        for name, func, args_dict in list:
            args = {arg_name: data_dict[arg_spec] for arg_name, arg_spec in args_dict.items()}
            result = func(**args)
            data_dict[name] = result

        return data_dict
