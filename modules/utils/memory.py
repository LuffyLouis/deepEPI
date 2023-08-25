import psutil
import os

def get_memory_usage(process):
    # 获取进程的内存使用量
    memory_info = process.memory_info()
    memory_usage = memory_info.rss
    return memory_usage

def convert_bytes_to_human_readable(size_bytes):
    # 定义存储大小单位
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

    # 当前使用的单位索引
    unit_index = 0

    # 循环将字节大小转换为更大单位，直到大小小于 1024 或没有更大的单位
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024
        unit_index += 1

    # 格式化输出转换后的大小
    return "{:.2f} {}".format(size_bytes, units[unit_index])

class MemoryUseReporter:
    def __init__(self, pid):
        self.pid = pid
        self.current_process = psutil.Process(self.pid)

    def get_memory(self):
        self.total_memory_usage = 0
        self.total_memory_usage += get_memory_usage(self.current_process)
        # 获取当前进程的所有子进程
        children = self.current_process.children(recursive=True)
        for child in children:
            self.total_memory_usage += get_memory_usage(child)
        return convert_bytes_to_human_readable(self.total_memory_usage)

    def get_memory_byte(self):
        total_memory_usage = 0
        total_memory_usage += get_memory_usage(self.current_process)
        # 获取当前进程的所有子进程
        children = self.current_process.children(recursive=True)
        for child in children:
            total_memory_usage += get_memory_usage(child)
        return total_memory_usage