"""
Base functions for (Ordinary/Stochastic/Delay) Differential Equations
Build new models on top of the BaseDE class and you will have *happy times*

Featuring:
  - Handling of parameters and model choice from the input dictionary
  - Plotting of timeseries and phase portraits
  - maybe future shit?
"""

class BaseModel:
    def __init__(self, **params):
        self.name = None
        self.params = params
        self.model = {} # get from _set_params
        self._set_params()

    def _set_params(self):
        for key, value in self.params.items():
            setattr(self, key, value)
            self.model[key] = value

    def update_params(self, **new_params):
        self.params.update(new_params)
        self._set_params()

    def solve(self):
        pass # to be defined by the model class