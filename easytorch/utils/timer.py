import time


class Timer:
    """Timer with multiple record

    Examples:
        >>> timer = Timer()
        >>> time.sleep(1)
        >>> timer.record('one')
        >>> time.sleep(2)
        >>> timer.record('two')
        >>> timer.print()
        Start:: [diff: 0.000000, total: 0.000000]
        one:: [diff: 1.002618, total: 1.002618]
        two:: [diff: 2.003077, total: 3.005695]
        >>> print(timer.get(2))
        (2.0030770301818848, 3.005695104598999)
        >>> print(timer.get(1))
        (1.0026180744171143, 1.0026180744171143)
        >>> print(timer.get(2, 0))
        (3.005695104598999, 3.005695104598999)
    """

    def __init__(self):
        self._record_dict = {'Start': time.time()}
        self._record_names = ['Start']

    def record(self, name: str = None):
        """Record a checkpoint

        Args:
            name (str): checkpoint name (default is Record_i, i is index)
        """

        if name is None:
            name = 'Record_{:d}'.format(len(self._record_names))
        elif self._record_dict.get(name) is not None:
            raise ValueError('Name \'{}\' already exists'.format(name))

        self._record_dict[name] = time.time()
        self._record_names.append(name)

    def print(self):
        """Print all checkpoints of this timer
        """
        start_time_record = last_time_record = self._record_dict['Start']
        for name in self._record_names:
            time_record = self._record_dict[name]
            time_diff = time_record - last_time_record
            time_total = time_record - start_time_record
            last_time_record = time_record
            print('{}:: [diff: {:2f}, total: {:2f}]'.format(name, time_diff, time_total))

    def get(self, end: str or int, start: str or int = None):
        """Get the time from the ```start``` to the```end```(diff),
        and the time from timer initialization to the ```end```(total).

        Notes:
            If start is none, default is the previous one of the ```end```.

        Args:
            end (str or int): end checkpoint name or index
            start (str or int): start checkpoint name or index

        Returns:
            (diff, total)
        """

        # end
        if isinstance(end, int):
            end_record_index = end
            end_record_name = self._record_names[end_record_index]
        else:
            end_record_name = end
            end_record_index = self._record_names.index(end_record_name)
        end_record_time = self._record_dict[end_record_name]

        # start
        if start is None:
            start_record_index = max(end_record_index - 1, 0)
            start_record_name = self._record_names[start_record_index]
        elif isinstance(start, int):
            start_record_name = self._record_names[start]
        else:
            start_record_name = start
        start_record_time = self._record_dict[start_record_name]

        return end_record_time - start_record_time, end_record_time - self._record_dict['Start']


class TimePredictor:
    def __init__(self, start_step: int, end_step: int):
        self.start_step = start_step
        self.end_step = end_step
        self.start_time = time.time()

    def get_remaining_time(self, step: int) -> float:
        now_time = time.time()
        return (now_time - self.start_time) * (self.end_step - self.start_step) / (step - self.start_step)

    def get_expected_end_time(self, step: int) -> float:
        return self.start_time + self.get_remaining_time(step)
