"""dummy cloud driver that stores all the actions applied to it
and provides data about the current state."""

from philharmonic import Pause, Unpause, Migration

#TODO: we probably don't want all these actions here - the Schedule is already
# built inside the simulator

class simdriver:
    def __init__(self):
        self.events = []

    def connect(self):
        """Connect to the cloud simulation."""
        self.events = []

    def store(self, action):
        t = self.environment.t
        event = (t, action)
        self.events.append(event)

    def suspend(self, instance):
        raise NotImplementedError

    def resume(self, instance):
        raise NotImplementedError

    def pause(self, instance):
        action = Pause(instance)
        self.store(action)

    def unpause(self, instance):
        action = Unpause(instance)
        self.store(action)

    def migrate(self, instance, machine):
        action = Migration(instance, machine)
        self.store(action)

    def apply_action(self, *args):
        """Apply an action to the simulation."""
        pass

