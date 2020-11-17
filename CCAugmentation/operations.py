import random


class Operation:
    """
    Abstract base class for all types of operations that may be put into a pipeline.
    """
    def __init__(self):
        """
        Create the Operation.
        By default, `requires_full_dataset_in_memory` member variable is set to False to tell that
        the operation works with generators (may process just one sample at a time) and doesn't require the full dataset
        to be stored in the memory. It may be overriden in the subclasses.
        """
        self.requires_full_dataset_in_memory = False

    def __str__(self):
        """ Stringify the operation """
        return self.__class__.__name__

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
        self.duplicates_num = duplicates_num

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
        self.probability = probability

    def execute(self, images_and_density_maps):
        """ Drops out samples """
        for image_and_density_map in images_and_density_maps:
            if random.random() >= self.probability:
                yield image_and_density_map


class RandomArgs(Operation):
    """
    Allows running operations with randomized numeral arguments.
    """
    def __init__(self, operation, const_args, random_args):
        """
        Specify randomization by providing the operation to invoke, const arguments that must be passed as they are and
        random arguments that will be randomized.

        :param operation: Type of operation that will be invoked. Must come from this, .outputs or .transformations module.
        :param const_args: Dictionary of constant arguments, i.e. ones whose values (nor names) won't change.
        :param random_args: Dictionary of randomized arguments specified as (min, max) tuple for each arg name. Values are taken from uniform distribution and are either floats or ints, depending on the types of provided min and max values.
        """
        Operation.__init__(self)
        self.operation = operation
        self.const_args = const_args
        self.random_args = random_args

    def _get_op_str(self):
        """
        Retrieve string representation of the operation to invoke, in a form that is ready to be computed in eval. That
        means, a prefix with module name is added when necessary. Only operations from this, .outputs and
        .transformations modules are supported.

        :return: Potentially prefixed operation name.
        """
        import CCAugmentation.outputs as cca_out
        import CCAugmentation.transformations as cca_trans

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
        for k, v in self.const_args.items():
            v_str = f"'{v}'" if type(v) is str else str(v)
            const_components.append(f"{k}={v_str}")
        return ",".join(const_components)

    def _get_rand_str(self):
        """
        Randomize, cast and transform to string the arguments that are to be randomized.

        :return: String representing randomized arguments.
        """
        rand_components = []
        for key, (min_val, max_val) in self.random_args.items():
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
