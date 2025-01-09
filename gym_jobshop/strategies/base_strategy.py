class SchedulingStrategy:
    def apply_rule(self, jobs, machines):
        pass

class ShortestProcessingTimeStrategy(SchedulingStrategy):
    def apply_rule(self, jobs, machines):
        # 实现 SPT 策略
        return sorted(jobs, key=lambda x: x.processing_time)
