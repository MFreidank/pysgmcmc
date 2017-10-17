# vim: foldmethod=marker
from enum import Enum


class Sampler(Enum):
    """ Enumeration type for all samplers we support. """

    SGHMC = "SGHMC"
    RelativisticSGHMC = "RelativisticSGHMC"
    SGLD = "SGLD"

    @staticmethod
    def is_burn_in_mcmc(sampling_method):
        """
        Static method that returns true if `sampling_method` is a
        burn_in sampler (e.g. there is an entry for it in `Sampler` enum).

        Examples
        ----------

        Burn-in sampling methods give `True`:

        >>> Sampler.is_burn_in_mcmc(Sampler.SGHMC)
        True

        Other sampling methods give `False`:

        >>> Sampler.is_burn_in_mcmc(Sampler.RelativisticSGHMC)
        False

        Other input types give `False`:

        >>> Sampler.is_burn_in_mcmc(0)
        False
        >>> Sampler.is_burn_in_mcmc("test")
        False

        """
        return sampling_method in (Sampler.SGHMC, Sampler.SGLD)

    @staticmethod
    def is_supported(sampling_method):
        """
        Static method that returns true if `sampling_method` is a
        supported sampler (e.g. there is an entry for it in `Sampler` enum).

        Examples
        ----------

        Supported sampling methods give `True`:

        >>> Sampler.is_supported(Sampler.SGHMC)
        True

        Other input types give `False`:

        >>> Sampler.is_supported(0)
        False
        >>> Sampler.is_supported("test")
        False

        """
        return sampling_method in (Sampler.SGHMC, Sampler.SGLD)

    @classmethod
    def get_sampler(cls, sampling_method, **sampler_args):
        """ Return a sampler object for supported `sampling_method`, where all
            default values for parameters in keyword dictionary `sampler_args`
            are overwritten.

        Parameters
        ----------
        sampling_method : Sampler
            Enum corresponding to sampling method to return a sampler for.

        **sampler_args : dict
            Keyword arguments that contain all input arguments to the desired
            the constructor of the sampler for the specified `sampling_method`.

        Returns
        ----------
        sampler : Subclass of `sampling.MCMCSampler`
            A sampler instance that implements the specified `sampling_method`
            and is initialized with inputs `sampler_args`.

        Examples
        ----------
        We can use this method to construct a sampler for a given
        sampling method and override default values by providing them as
        keyword arguments:

        >>> import tensorflow as tf
        >>> params = [tf.Variable(0.)]
        >>> cost_fun = lambda params: tf.reduce_sum(params) # dummy cost function
        >>> session=tf.Session()
        >>> with tf.Session() as session: sampler = Sampler.get_sampler(Sampler.SGHMC, session=session, params=params, cost_fun=cost_fun, dtype=tf.float32)
        >>> type(sampler)
        <class 'pysgmcmc.samplers.sghmc.SGHMCSampler'>
        >>> sampler.dtype
        tf.float32
        >>> session.close()

        Construction of SGLD sampler:

        >>> import tensorflow as tf
        >>> params = [tf.Variable(0.)]
        >>> cost_fun = lambda params: tf.reduce_sum(params) # dummy cost function
        >>> session=tf.Session()
        >>> with tf.Session() as session: sampler = Sampler.get_sampler(Sampler.SGLD, session=session, params=params, cost_fun=cost_fun, dtype=tf.float32)
        >>> type(sampler)
        <class 'pysgmcmc.samplers.sgld.SGLDSampler'>
        >>> sampler.dtype
        tf.float32
        >>> session.close()

        Construction of Relativistic SGHMC sampler:

        >>> import tensorflow as tf
        >>> params = [tf.Variable(0.)]
        >>> cost_fun = lambda params: tf.reduce_sum(params) # dummy cost function
        >>> session=tf.Session()
        >>> with tf.Session() as session: sampler = Sampler.get_sampler(Sampler.RelativisticSGHMC, session=session, params=params, cost_fun=cost_fun, dtype=tf.float32)
        >>> type(sampler)
        <class 'pysgmcmc.samplers.relativistic_sghmc.RelativisticSGHMCSampler'>
        >>> sampler.dtype
        tf.float32
        >>> session.close()

        Sampler arguments that do not have a default *must* be provided as keyword
        argument, otherwise this method will raise an exception:

        >>> sampler = Sampler.get_sampler(Sampler.SGHMC, dtype=tf.float32)
        Traceback (most recent call last):
          ...
        ValueError: sampling.Sampler.get_sampler: params was not overwritten as sampler argument in `sampler_args` and does not have any default value in SGHMCSampler.__init__Please pass an explicit value for this parameter.

        If an **optional** argument is not provided as keyword argument,
        the corresponding default value is used.
        If we do not provide/overwrite the `dtype` keyword argument,
        the samplers default value of `tf.float64` is used:

        >>> import tensorflow as tf
        >>> params = [tf.Variable(0., dtype=tf.float64)]
        >>> cost_fun = lambda params: tf.reduce_sum(params) # dummy cost function
        >>> with tf.Session() as session: sampler = Sampler.get_sampler(Sampler.SGHMC, session=session, params=params, cost_fun=cost_fun)
        >>> sampler.dtype
        tf.float64

        If a keyword argument that is provided does not represent a valid
        parameter of the corresponding `sampling_method`, a `ValueError` is
        raised:

        >>> import tensorflow as tf
        >>> params = [tf.Variable(0., dtype=tf.float64)]
        >>> cost_fun = lambda params: tf.reduce_sum(params) # dummy cost function
        >>> with tf.Session() as session: sampler = Sampler.get_sampler(Sampler.SGHMC, unknown_argument=None, session=session, params=params, cost_fun=cost_fun)
        Traceback (most recent call last):
          ...
        ValueError: sampling.Sampler.get_sampler: 'SGHMCSampler' does not take any parameter with name 'unknown_argument' which was specified as argument to this sampler. Please ensure, that you only specify sampler arguments that fit the corresponding sampling method.
        For your choice of sampling method ('Sampler.SGHMC'), supported parameters are:
        -params
        -cost_fun
        -batch_generator
        -stepsize_schedule
        -burn_in_steps
        -mdecay
        -scale_grad
        -session
        -dtype
        -seed

        """

        if sampling_method == cls.SGHMC:
            from pysgmcmc.samplers.sghmc import SGHMCSampler as Get_Sampler
        elif sampling_method == cls.SGLD:
            from pysgmcmc.samplers.sgld import SGLDSampler as Get_Sampler
        elif sampling_method == cls.RelativisticSGHMC:
            from pysgmcmc.samplers.relativistic_sghmc import (
                RelativisticSGHMCSampler as Get_Sampler
            )
        else:
            raise ValueError(
                "Sampling method {sampler} is supported, but function "
                "'pysgmcmc.sampling.get_sampler' is missing an `import` "
                "statement for the corresponding sampler object. "
                "Please add an import in the appropriate location."
            )

        from inspect import signature, _empty

        # look up all initializer parameters with their (potential)
        # default values
        all_sampler_parameters = signature(Get_Sampler.__init__).parameters

        # Check if any invalid sampler arguments were passed
        # (sampler arguments that are not actually parameters of the specified)
        # sampling method
        try:
            undefined_parameter = next(
                parameter_name for parameter_name in sampler_args
                if parameter_name not in all_sampler_parameters
            )
        except StopIteration:
            pass
        else:
            raise ValueError(
                "sampling.Sampler.get_sampler: '{sampler_name}' "
                "does not take any parameter with name '{parameter}' "
                "which was specified as argument to this sampler. "
                "Please ensure, that you only specify sampler arguments "
                "that fit the corresponding sampling method.\n"
                "For your choice of sampling method ('{sampler}'), supported parameters are:\n"
                "{valid_parameters}".format(
                    sampler_name=Get_Sampler.__name__,
                    sampler=sampling_method,
                    parameter=undefined_parameter,
                    valid_parameters="\n".join(
                        ["-{}".format(parameter_name)
                         for parameter_name in all_sampler_parameters
                         if parameter_name != "self"]
                    )
                )
            )

        def parameter_value(parameter_name):
            """ Determine the value to assign to the parameter
                with name `parameter_name`.
                If `parameter_name` is overwritten (if it is a key in
                `sampler_args`) use the value provided in `sampler_args`.
                Otherwise, fall back to the default value provided in
                the samplers `init` method.

            Parameters
            ----------
            parameter_name : string
                Name of the parameter that we want to determine the value for.

            Returns
            -------
            value : object
                Value of sampler parameter with name `parameter_name` that
                will be passed to the initializer of the sampler.

            """

            default_value = all_sampler_parameters[parameter_name].default

            if parameter_name not in sampler_args and default_value is _empty:
                raise ValueError(
                    "sampling.Sampler.get_sampler: "
                    "{param_name} was not overwritten as sampler argument "
                    "in `sampler_args` and does not have any default value "
                    "in {sampler}.__init__"
                    "Please pass an explicit value for this parameter.".format(
                        param_name=parameter_name, sampler=Get_Sampler.__name__
                    )
                )

            return sampler_args.get(parameter_name, default_value)

        sampler_args = {
            parameter_name: parameter_value(parameter_name)
            for parameter_name in all_sampler_parameters
            if parameter_name != "self"  # never pass `self` during construction
        }

        return Get_Sampler(**sampler_args)
