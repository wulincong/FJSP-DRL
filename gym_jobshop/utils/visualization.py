class Observer:
    def update(self, state):
        pass

class ConsoleLogger(Observer):
    def update(self, state):
        print(f"Current state: {state}")
