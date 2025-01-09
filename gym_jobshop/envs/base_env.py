class EnvironmentFactory:
    @staticmethod
    def create_env(env_type, **kwargs):
        if env_type == "SingleMachine":
            return SingleMachineEnv(**kwargs)
        elif env_type == "FlowShop":
            return FlowShopEnv(**kwargs)
        elif env_type == "JobShop":
            return JobShopEnv(**kwargs)
        # 扩展更多类型
class BaseScheduler:
    def initialize(self):
        pass

    def schedule(self):
        raise NotImplementedError("Subclasses should implement this method")

    def update_state(self):
        pass
