from typing import Callable

class Engine(object):
    """ Logic class with support callbacks """

    def __init__(self, *args, **kwargs):
        self.callbacks_list = {}
        self.cache = {} # Used for everyone
        self.is_running = False

    def add_callback(self, event_name, func):
        if event_name not in self.callbacks_list:
            self.callbacks_list[event_name] = []
        
        self.callbacks_list[event_name].append(func)

    def fire_event(self, event_name, *args, **kwargs):
        for v in self.callbacks_list.get(event_name, []):
            v(self, *args, **kwargs)

    def terminate(self):
        self.is_running = False

    @staticmethod
    def add_event_listener(event_name: str, *event_args, **event_kwargs) -> Callable:
        """ Decorator to warp function and name event """ 

        def decorator(function: Callable) -> Callable:
                def new_function(self, *args, **kwargs):
                    self.fire_event(f"on_{event_name}_start", 
                                     *event_args, 
                                     **event_kwargs
                                   )
                                   
                    result = function(self, *args, **kwargs)
                    if isinstance(result, dict):
                        event_kwargs.update(result)
                        
                    self.fire_event(f"on_{event_name}_end", 
                                    *event_args, 
                                    **event_kwargs
                                   )
                return new_function
        return decorator