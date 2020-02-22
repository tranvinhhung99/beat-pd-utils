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
            v(*args, **kwargs)

    def terminate(self):
        self.is_running = False