import time
import warnings
from contextlib import contextmanager

import torch


class TimeCounter:
    names = dict()

    # Avoid instantiating every time
    @classmethod
    def count_time(cls, log_interval=1, warmup_interval=1, with_sync=True):
        assert warmup_interval >= 1

        def _register(func):
            if func.__name__ in cls.names:
                raise RuntimeError(
                    'The registered function name cannot be repeated!')
            # When adding on multiple functions, we need to ensure that the
            # data does not interfere with each other
            cls.names[func.__name__] = dict(
                count=0,
                pure_inf_time=0,
                log_interval=log_interval,
                warmup_interval=warmup_interval,
                with_sync=with_sync)

            def fun(*args, **kwargs):
                count = cls.names[func.__name__]['count']
                pure_inf_time = cls.names[func.__name__]['pure_inf_time']
                log_interval = cls.names[func.__name__]['log_interval']
                warmup_interval = cls.names[func.__name__]['warmup_interval']
                with_sync = cls.names[func.__name__]['with_sync']

                count += 1
                cls.names[func.__name__]['count'] = count

                if with_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                result = func(*args, **kwargs)

                if with_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time

                if count >= warmup_interval:
                    pure_inf_time += elapsed
                    cls.names[func.__name__]['pure_inf_time'] = pure_inf_time

                    if count % log_interval == 0:
                        times_per_count = 1000 * pure_inf_time / (
                            count - warmup_interval + 1)
                        print(
                            f'[{func.__name__}]-{count} times per count: '
                            f'{times_per_count:.1f} ms',
                            flush=True)

                return result

            return fun

        return _register

    @classmethod
    @contextmanager
    def profile_time(cls,
                     func_name,
                     log_interval=1,
                     warmup_interval=1,
                     with_sync=True):
        assert warmup_interval >= 1
        warnings.warn('func_name must be globally unique if you call '
                      'profile_time multiple times')

        if func_name in cls.names:
            count = cls.names[func_name]['count']
            pure_inf_time = cls.names[func_name]['pure_inf_time']
            log_interval = cls.names[func_name]['log_interval']
            warmup_interval = cls.names[func_name]['warmup_interval']
            with_sync = cls.names[func_name]['with_sync']
        else:
            count = 0
            pure_inf_time = 0
            cls.names[func_name] = dict(
                count=count,
                pure_inf_time=pure_inf_time,
                log_interval=log_interval,
                warmup_interval=warmup_interval,
                with_sync=with_sync)

        count += 1
        cls.names[func_name]['count'] = count

        if with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        yield

        if with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if count >= warmup_interval:
            pure_inf_time += elapsed
            cls.names[func_name]['pure_inf_time'] = pure_inf_time

            if count % log_interval == 0:
                times_per_count = 1000 * pure_inf_time / (
                    count - warmup_interval + 1)
                print(
                    f'[{func_name}]-{count} times per count: '
                    f'{times_per_count:.1f} ms',
                    flush=True)


def demo_count_time():
    @TimeCounter.count_time()
    def fun1():
        time.sleep(2)

    @TimeCounter.count_time()
    def fun2():
        time.sleep(1)

    for _ in range(20):
        fun1()
        for _ in range(2):
            fun2()


def demo_profile_time():
    # 第一种用法，需要对代码进行缩进
    for _ in range(20):
        with TimeCounter.profile_time('sleep1'):
            print('start test profile_time')
            time.sleep(2)
            print('end test profile_time')

        #     with TimeCounter.profile_time('sleep2', warmup_interval=5):
        #         print('start test profile_time')
        #         time.sleep(1)
        #         print('end test profile_time')
        #
        # # 第二种用法：直接在代码前后插入上下文，不需要对代码进行缩进
        # time_counter = TimeCounter.profile_time('sleep3')
        # time_counter.__enter__()
        #
        # # your code
        # time.sleep(2.5)
        #
        # time_counter.__exit__(None, None, None)

        # 注意：当 TimeCounter.profile_time 应用在多处时候，func_name 必须全局唯一，否则统计时间是不对的
        # 暂时没有办法 assert，只能用户自行保证

        # 以下写法是错误的，因为前面已经有了一个 sleep2 函数名,实际打印的时间为两个上下文函数执行总时间的平均值
        # with TimeCounter.profile_time('sleep2'):
        #     print('start test profile_time 1')
        #     time.sleep(3)
        #     print('end test profile_time 1 ')


if __name__ == '__main__':
    # demo_count_time()
    demo_profile_time()