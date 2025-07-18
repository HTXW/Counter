"""cloud driver that doesn't do anything"""

class nodriver:
    def __init__(self):
        pass

    def connect(self):
        pass

    def suspend(self, instance):
        pass

    def resume(self, instance):
        pass

    def pause(self, instance):
        pass

    def unpause(self, instance):
        pass

    def migrate(self, instance, machine):
        pass

    def apply_action(self, *args):
        pass

