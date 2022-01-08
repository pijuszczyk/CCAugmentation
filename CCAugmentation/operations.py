import itertools as _itertools
import random as _random

import numpy as _np


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
        """ Stringify the operation. """
        return self.__class__.__name__

    @staticmethod
    def _prepare_args(local_vars):
        """ Simple method that removes unwanted 'self' variable from the set that will be stored for loading and saving pipelines. """
        return {k: v for k, v in local_vars.items() if k != 'self'}

    def to_json(self):
        """ Serialize operation configuration to JSON-compatible dict. """
        return {'name': self.__class__.__name__, 'args': self.args}

    def get_output_samples_number_multiplier(self):
        """ Return how many img+DM pairs are on average returned as output from a single pair. """
        return 1

    def execute(self, images_and_density_maps):
        """ Abstract method that must be implemented in the subclasses, should take and return an iterable of img+DM pairs. """
        raise NotImplementedError("execute method not implemented in the child class")


class Duplicate(Operation):
    """
    Duplicates each sample in a dataset a specified number of times. One of the most helpful operations when it comes
    to data augmentation.
    """
    def __init__(self, duplicates_num):
        """
        Define duplication.

        Args:
            duplicates_num: Each sample will be repeated that number of times.
        """
        Operation.__init__(self)
        self.args = self._prepare_args(locals())
        self.duplicates_num = duplicates_num

    def get_output_samples_number_multiplier(self):
        """ Return how many img+DM pairs are on average returned as output from a single pair. """
        return self.duplicates_num

    def execute(self, images_and_density_maps):
        """ Duplicates samples. """
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

        Args:
            probability: Each sample will be dropped out with this probability, meaning that the estimated number of output images for a dataset with `N` samples is `N*(1-probability)`.
        """
        Operation.__init__(self)
        self.args = self._prepare_args(locals())
        self.probability = probability

    def get_output_samples_number_multiplier(self):
        """ Return how many img+DM pairs are on average returned as output from a single pair. """
        return 1.0 - self.probability

    def execute(self, images_and_density_maps):
        """ Drops out samples. """
        for image_and_density_map in images_and_density_maps:
            if _random.random() >= self.probability:
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

        Args:
            operation: Type of operation that will be invoked. Class or name of class. Must come from this, .outputs or .transformations module.
            constargs: Dictionary of constant arguments, i.e. ones whose values (nor names) won't change.
            randomargs: Dictionary of randomized arguments specified as (min, max) tuple for each arg name. Values are taken from uniform distribution and are either floats or ints, depending on the types of provided min and max values.
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
                return _np.mean(self.randomargs["duplicates_num"])
            else:
                raise KeyError("Duplicate operation missing duplicates_num")
        elif self.operation is Dropout:
            if "probability" in self.constargs:
                return self.constargs["probability"]
            elif "probability" in self.randomargs:
                return _np.mean(self.randomargs["probability"])
            else:
                raise KeyError("Dropout operation missing probability")
        else:
            return 1

    def _get_op_str(self):
        """
        Retrieve string representation of the operation to invoke, in a form that is ready to be computed in eval. That
        means, a prefix with module name is added when necessary. Only operations from this, .outputs and
        .transformations modules are supported.

        Returns:
            Potentially prefixed operation name.
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

        Returns:
            String representing constant arguments.
        """
        const_components = []
        for k, v in self.constargs.items():
            v_str = f"'{v}'" if type(v) is str else str(v)
            const_components.append(f"{k}={v_str}")
        return ",".join(const_components)

    def _get_rand_str(self):
        """
        Randomize, cast and transform to string the arguments that are to be randomized.

        Returns:
            String representing randomized arguments.
        """
        rand_components = []
        for key, (min_val, max_val) in self.randomargs.items():
            val = _random.uniform(min_val, max_val)
            if type(min_val) is int and type(max_val) is int:
                val = int(val)
            rand_components.append(f"{key}={str(val)}")
        return ",".join(rand_components)

    def execute(self, images_and_density_maps):
        """
        Create and execute the given operation with a set of constant and randomized arguments.

        Args:
            images_and_density_maps: Iterable of img+DM pairs that are to be used in the operation.

        Returns:
            Result img+DM pairs from the operation.
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


class OptimizeBatch(Operation):
    """
    Generally, it is required for batches to contain only data samples of the same shape. This is an operation that
    lays out input tuples in a way that maximizes possible batch size. To deal with the non-matching tuples when
    the batch still hasn't been completed, a temporary buffer is used to store the non-matching tuples. The user can
    limit the buffer size so that from time to time it will be cleaned up and the memory usage will be below some
    threshold; however, when used with too restrictive limit, this can lead to suboptimal batches.
    """
    def __init__(self, target_batch_size, max_buffer_size=None):
        """
        Construct a batch optimizer that tries to lays out input tuples in a way to allow easy construction of batches
        of target size at a later stage. The temporary buffer can be limited in size.

        Args:
            target_batch_size: Number of img+DM pairs that the optimizer tries to lay out one after another, preserving shape consistency.
            max_buffer_size: Maximum number of img+DM pairs that can rest in the temporary buffer, waiting for a better moment to be put into a batch. If None, buffer size is unlimited.
        """
        Operation.__init__(self)
        self.args = self._prepare_args(locals())
        self.target_batch_size = target_batch_size
        self.max_buffer_size = float("inf") if max_buffer_size is None else max_buffer_size

    def execute(self, images_and_density_maps):
        """
        Optimizes layout of input samples.

        Args:
            images_and_density_maps: Img+DM pairs that will be kind of sorted for batch construction efficiency.

        Returns:
            Img+DM pairs in optimized order.
        """
        if self.max_buffer_size == float("inf"):
            for image, density_map in sorted(images_and_density_maps, key=lambda t: t[0].shape):
                yield image, density_map
            return

        batch = []
        batch_size = 0
        batched_image_shape = None
        buffer = []
        buffer_size = 0

        loop_cutoff = 1000000
        for _ in range(loop_cutoff):
            stop = True  # condition that assures that there is still work to be done; if True, the loop is exited

            for image, density_map in images_and_density_maps:
                stop = False
                if image.shape != batched_image_shape:
                    if buffer_size < self.max_buffer_size and batched_image_shape is not None:
                        buffer.append((image, density_map))
                        buffer_size += 1
                    else:
                        for batched_image, batched_density_map in batch:
                            yield batched_image, batched_density_map
                        batch = [(image, density_map)]
                        batch_size = 1
                        batched_image_shape = image.shape
                        break  # try going through buffer, clearing it
                else:
                    batch.append((image, density_map))
                    batch_size += 1

                if batch_size == self.target_batch_size:
                    for batched_image, batched_density_map in batch:
                        yield batched_image, batched_density_map
                    batch = []
                    batch_size = 0

            buffer.sort(key=lambda t: t[0].shape)
            matching_shape_found_in_buffer = False
            used_images_start, used_images_end = None, buffer_size
            for i, (image, density_map) in enumerate(buffer):
                if image.shape == batched_image_shape:
                    stop = False  # set stop to False only if there are images to take from the buffer to avoid (semi)infinite loops
                    if not matching_shape_found_in_buffer:
                        matching_shape_found_in_buffer = True
                        used_images_start = i
                    batch.append((image, density_map))
                    batch_size += 1
                    if batch_size == self.target_batch_size:
                        for batched_image, batched_density_map in batch:
                            yield batched_image, batched_density_map
                        batch = []
                        batch_size = 0
                else:
                    if matching_shape_found_in_buffer:  # there were images to use in the buffer but there are no more (buffer is sorted so it is known)
                        used_images_end = i
                        break
            if matching_shape_found_in_buffer:
                buffer = list(_itertools.chain(buffer[:used_images_start], buffer[used_images_end:buffer_size]))
                buffer_size = len(buffer)

            if stop:
                break

        # return the remains
        for image, density_map in _itertools.chain(batch, buffer):
            yield image, density_map
