from typing import Any, Callable, Collection, Dict, Iterable, Tuple


class AutoModelCyclicReferenceError(Exception):
    def __init__(self, cyclic_list) -> None:
        super().__init__(f"Possible AutoModel cyclic reference in ({', '.join(cyclic_list)})")


class AutoModelComponent:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.methods = {}
        self.targets = []
        self.parameterized_methods = {}
        self.parameterized_method_calls = {}

        self.automodel = None

    def register_method(self, name: str, method: Callable, args: Dict[str, str]):
        self.methods[name] = (method, args)

    def register_parameterized_method(self, name: str, method: Callable, args: Dict[str, str], params: Iterable[str]):
        self.parameterized_methods[name] = (method, args, params)

    def register_parameterized_method_call(self, name: str, method: str, params: Dict[str, str]):
        self.parameterized_method_calls[name] = (method, params)

    def call(self, name: str, args: Dict[str, Any]):
        return self.automodel.call_now(name, args)


class AutoModel:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._components = {}

    def register_component(self, name: str, component: AutoModelComponent):
        self._components[name] = component
        component.automodel = self

    def prepare_default_call_list(
        self, data: Collection[str], additional: Iterable[str] = ()
    ) -> Tuple[Iterable[Tuple[str, Callable, Dict[str, str]]], Collection[str]]:
        targets = list(additional)
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
                component, fun, args = self._get_method(method)
                args_ready = True

                args = {
                    arg: source.replace('self.',  f'{component}.') if source.startswith('self.') else source
                    for arg, source in args.items()
                }

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
        component = self._components[component_name]

        if method_name in component.parameterized_method_calls:
            return (component_name,) + self._get_parameterized_method(component, method, method_name)
        if method_name not in component.methods:
            raise ValueError(f"Missing method {method_name} in {component_name}")

        return (component_name,) + component.methods[method_name]

    def _get_parameterized_method(self, component: AutoModelComponent, name: str, method_name: str):
        name, params = component.parameterized_method_calls[method_name]
        base_component_name, base_method_name = name.split('.')
        
        if base_component_name not in self._components:
            raise ValueError(f"Missing component {base_component_name} in {base_method_name} (from {name})")
        base_component = self._components[base_component_name]

        if base_method_name not in base_component.parameterized_methods:
            raise ValueError(f"Missing method {base_method_name} in {base_component_name} (from {name})")

        fun, args, param_names = base_component.parameterized_methods[base_method_name]
        if set(params.keys()) != set(param_names):
            raise ValueError(f"Invalid args in call {method_name} (from {name}). Expected {param_names}, got {list(params.keys())}")
        all_args = dict(args)
        for param_name, param_value in params.items():
            all_args[param_name] = param_value

        return (fun, all_args)

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

    def call_now(self, name: str, args: Dict[str, Any]):
        _, method, method_args = self._get_method(name)
        args = {method_args[arg]: value for arg, value in args.items()}
        return method(**args)
