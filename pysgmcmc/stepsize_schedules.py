from abc import ABCMeta, abstractmethod


class StepsizeSchedule(object):
    """ Generic base class for all stepsize schedules. """
    __metaclass__ = ABCMeta

    def __init__(self, initial_value):
        self.initial_value = initial_value

    @abstractmethod
    def __next__(self):
        """
        Compute and return the next stepsize according to this schedule.

        Returns
        ----------
        next_stepsize : float
            Next stepsize to use according to this schedule.
        """

    def __iter__(self):
        return self

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update this schedule with new information. What information
        will be relevant depends on the type of schedule used.
        Information may e.g. include cost values for the last step size
        used, effective sample sizes of a sampler, values of other
        hyperparameters etc.

        """


class ConstantStepsizeSchedule(StepsizeSchedule):
    """ Trivial schedule that keeps the stepsize at a constant value.  """

    def __next__(self):
        """
        Calling `next(schedule)` on a constant stepsize schedule
        will always return the schedules initial value.

        Returns
        ----------
        constant_value : float
            Constant value associated with this schedule.

        Examples
        ----------
        Proof of concept:

        >>> schedule = ConstantStepsizeSchedule(0.01)
        >>> schedule.initial_value
        0.01
        >>> next(schedule)
        0.01
        >>> from itertools import islice
        >>> list(islice(schedule, 4))
        [0.01, 0.01, 0.01, 0.01]

        """
        return self.initial_value

    def __str__(self):
        """ Pretty string representation of `ConstantStepsizeSchedule`.

        Returns
        ----------
        schedule_str : string
            String representation of this schedule.

        Examples
        ----------
        Proof of concept:

        >>> schedule = ConstantStepsizeSchedule(0.01)
        >>> str(schedule)
        'ConstantStepsizeSchedule(stepsize=0.01)'

        >>> schedule = ConstantStepsizeSchedule(0.1)
        >>> str(schedule)
        'ConstantStepsizeSchedule(stepsize=0.1)'

        """
        return "ConstantStepsizeSchedule(stepsize={})".format(self.initial_value)

    def update(self, *args, **kwargs):
        """ Updating a constant stepsize schedule is a no-op. """
        pass
