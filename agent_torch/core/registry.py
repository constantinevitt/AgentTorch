from typing import Type

import torch.nn as nn
import json

from agent_torch.core.substep import SubstepTransition


class Registry(nn.Module):
    helpers = {
        "transition": {},
        "observation": {},
        "policy": {},
        "initialization": {},
        "network": {},
    }

    def __init__(self):
        super().__init__()
        self.initialization_helpers = self.helpers["initialization"]
        self.observation_helpers = self.helpers["observation"]
        self.policy_helpers = self.helpers["policy"]
        self.transition_helpers = self.helpers["transition"]
        self.network_helpers = self.helpers["network"]

    def register(self, obj_source, name, key):
        """Inserts a new function into the registry"""
        self.helpers[key][name] = obj_source

    def view(self):
        """Pretty prints the entire registry as a JSON object"""
        return json.dumps(self.helpers, indent=2)

    def forward(self):
        print("Invoke registry.register(class_obj, key)")

    @classmethod
    def register_helper(cls, name, key):
        def decorator(fn):
            cls.helpers[key][name] = fn
            return fn

        return decorator

    register_substep = register_helper

    def register_transition(
        self,
        obj_source: Type[SubstepTransition],
        name: str = None,
    ) -> None:
        obj_name = obj_source.__dict__.get("name", None)
        if name is not None:
            obj_name = name

        if obj_name is None:
            raise ValueError(
                "SubstepTransition must implement a 'name' attribute "
                "or one must be provided explicitly."
            )

        self.register(obj_source, obj_name, key="transition")
