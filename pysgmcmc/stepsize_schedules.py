class StepsizeSchedule(object):
    def update(self, params, cost):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()


class ConstantStepsizeSchedule(object):
    def __init__(self, initial_value):
        super(ConstantStepsizeSchedule, self).__init__()
        self.initial_value = initial_value

    def __next__(self):
        return self.initial_value

    def update(self, *args, **kwargs):
        pass
