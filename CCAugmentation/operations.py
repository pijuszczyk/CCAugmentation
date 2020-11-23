import random

import numpy as np


class Operation:
    """
    Abstract base class for all types of operations that may be put into a pipeline.
    """
    def __init__(self):
        """
        Create the Operation.

        args member contains the arguments that were used while creating the operation. It is necessary for loading
        and saving pipelines.

        By default, `requires_full_dataset_in_memory` member variable is set to False to tell that
        the operation works with generators (may process just one sample at a time) and doesn't require the full dataset
        to be stored in the memory. It may be overriden in the subclasses.
        """
        self.args = self._prepare_args(locals())
        self.requires_full_dataset_in_memory = False

    def __str__(self):
        """ Stringify the operation """
        return self.__class__.__name__

    @staticmethod
    def _prepare_args(local_vars):
        """ Simple method that removes unwanted 'self' variable from the set that will be stored for loading and saving pipelines"""
        return {k: v for k, v in local_vars.items() if k != 'self'}

    def to_json(self):
        """ Serialize operation configuration to JSON-compatible dict """
        return {'name': self.__class__.__name__, 'args': self.args}

    def get_output_samples_number_multiplier(self):
        """ Return how many img+DM pairs are on average returned as output from a single pair """
        return 1

    def execute(self, images_and_density_maps):
        """ Abstract method that must be implemented in the subclasses, should take and return an iterable of img+DM pairs """
        raise NotImplementedError("execute method not implemented in the child class")


class Duplicate(Operation):
    """
    Duplicates each sample in a dataset a specified number of times. One of the most helpful operations when it comes
    to data augmentation.
    """
    def __init__(self, duplicates_num):
        """
        Define duplication.

        :param duplicates_num: Each sample will be repeated that number of times.
        """
        Operation.__init__(self)
        self.args = self._prepare_args(locals())
        self.duplicates_num = duplicates_num

    def get_output_samples_number_multiplier(self):
        """ Return how many img+DM pairs are on average returned as output from a single pair """
        return self.duplicates_num

    def execute(self, images_and_density_maps):
        """ Duplicates samples """
        for image_and_density_map in images_and_density_maps:
            for i in range(self.duplicates_num):
                yield image_and_density_map


class Dropout(Operation):
    """
    Drops out samples with a given probability.
    """
    def __init__(self, probability):
        """
        Define dropout.

        :param probability: Each sample will be dropped out with this probability, meaning that the estimated number of output images for a dataset with `N` samples is `N*(1-probability)`.
        """
        Operation.__init__(self)
        self.args = self._prepare_args(locals())
        self.probability = probability

    def get_output_samples_number_multiplier(self):
        """ Return how many img+DM pairs are on average returned as output from a single pair """
        return self.probability

    def execute(self, images_and_density_maps):
        """ Drops out samples """
        for image_and_density_map in images_and_density_maps:
            if random.random() >= self.probability:
                yield image_and_density_map


class RandomArgs(Operation):
    """
    Allows running operations with randomized numeral arguments.
    """
    def __init__(self, operation, constargs, randomargs):
        """
        Specify randomization by providing the operation to invoke, const arguments that must be passed as they are and
        random arguments that will be randomized. Only standard, defined in the project, operations are allowed and
        unique names are assumed.

        :param operation: Type of operation that will be invoked. Class or name of class. Must come from this, .outputs or .transformations module.
        :param constargs: Dictionary of constant arguments, i.e. ones whose values (nor names) won't change.
        :param randomargs: Dictionary of randomized arguments specified as (min, max) tuple for each arg name. Values are taken from uniform distribution and are either floats or ints, depending on the types of provided min and max values.
        """
        Operation.__init__(self)
        self.operation = operation
        self.constargs = constargs
        self.randomargs = randomargs
        if type(operation) is str:
            import CCAugmentation.outputs as cca_out
            import CCAugmentation.transformations as cca_trans
            self.operation = eval(self._get_op_str())
        self.args = {'operation': self.operation.__name__, 'constargs': constargs, 'randomargs': randomargs}

    def get_output_samples_number_multiplier(self):
        """ Return how many img+DM pairs are on average returned as output from a single pair """
        if self.operation is Duplicate:
            if "duplicates_num" in self.constargs:
                return self.constargs["duplicates_num"]
            elif "duplicates_num" in self.randomargs:
                return np.mean(self.randomargs["duplicates_num"])
            else:
                raise KeyError("Duplicate operation missing duplicates_num")
        elif self.operation is Dropout:
            if "probability" in self.constargs:
                return self.constargs["probability"]
            elif "probability" in self.randomargs:
                return np.mean(self.randomargs["probability"])
            else:
                raise KeyError("Dropout operation missing probability")
        else:
            return 1

    def _get_op_str(self):
        """
        Retrieve string representation of the operation to invoke, in a form that is ready to be computed in eval. That
        means, a prefix with module name is added when necessary. Only operations from this, .outputs and
        .transformations modules are supported.

        :return: Potentially prefixed operation name.
        """
        import CCAugmentation.outputs as cca_out
        import CCAugmentation.transformations as cca_trans

        if type(self.operation) is str:
            op_name_str = self.operation
        else:
            op_name_str = self.operation.__name__

        try:
            getattr(cca_trans, op_name_str)
            op_str = f"cca_trans.{op_name_str}"
        except AttributeError:
            try:
                getattr(cca_out, op_name_str)
                op_str = f"cca_out.{op_name_str}"
            except AttributeError:
                op_str = op_name_str

        return op_str

    def _get_const_str(self):
        """
        Transform dictionary of constant arguments to string that can be used inside parentheses in eval().

        :return: String representing constant arguments.
        """
        const_components = []
        for k, v in self.constargs.items():
            v_str = f"'{v}'" if type(v) is str else str(v)
            const_components.append(f"{k}={v_str}")
        return ",".join(const_components)

    def _get_rand_str(self):
        """
        Randomize, cast and transform to string the arguments that are to be randomized.

        :return: String representing randomized arguments.
        """
        rand_components = []
        for key, (min_val, max_val) in self.randomargs.items():
            val = random.uniform(min_val, max_val)
            if type(min_val) is int and type(max_val) is int:
                val = int(val)
            rand_components.append(f"{key}={str(val)}")
        return ",".join(rand_components)

    def execute(self, images_and_density_maps):
        """
        Create and execute the given operation with a set of constant and randomized arguments.

        :param images_and_density_maps: Iterable of img+DM pairs that are to be used in the operation.
        :return: Result img+DM pairs from the operation.
        """
        # these imports are used in eval(), don't remove them
        import CCAugmentation.outputs as cca_out
        import CCAugmentation.transformations as cca_trans
        _ = cca_out, cca_trans

        op_str = self._get_op_str()
        const_str = self._get_const_str()

        for image_and_density_map in images_and_density_maps:
            rand_str = self._get_rand_str()
            args_str = ",".join([const_str, rand_str]) if const_str and rand_str else const_str + rand_str
            op = eval(f"{op_str}({args_str})")
            for result in op.execute([image_and_density_map]):
                yield result
