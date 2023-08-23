import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.final_start_time = None
        self.final_end_time = None

    def start(self):
        """开始计时"""
        self.start_time = time.time()

    def stop(self):
        """结束计时"""
        self.end_time = time.time()

    def final_start(self):
        """开始计时"""
        self.final_start_time = time.time()

    def final_stop(self):
        """结束计时"""
        self.final_end_time = time.time()

    def elapsed_time(self):
        """返回经过的时间（秒）"""
        if self.start_time is None:
            raise ValueError("计时未开始，请先调用start()方法开始计时。")

        if self.end_time is None:
            raise ValueError("计时未结束，请先调用stop()方法结束计时。")

        return self.end_time - self.start_time

    def final_elapsed_time(self):
        if self.final_start_time is None:
            raise ValueError("计时未开始，请先调用start()方法开始计时。")

        if self.final_end_time is None:
            raise ValueError("计时未结束，请先调用stop()方法结束计时。")

        return self.final_end_time - self.final_start_time
