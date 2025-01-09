class LoggingDecorator:
    def __init__(self, env):
        self.env = env

    def step(self, action):
        print(f"Action taken: {action}")
        return self.env.step(action)
