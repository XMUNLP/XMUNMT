# variable_scope.py
# modified from Tensorflow's tf.variable_scope
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import six
import theano
import contextlib

from variable import variable
from regularizer import add_regularization_loss
from name_scope import name_scope as name_scope_op
from dtype import is_integer_dtype, is_floating_dtype
from collection import add_to_collection, get_collection, get_collection_ref
from initializer import uniform_unit_scaling_initializer, zeros_initializer


__all__ = ["variable_scope", "get_variable_scope"]


_VARSTORE_KEY = ("__variable_store",)
_VARSCOPE_KEY = ("__varscope",)


def is_compatible_shape(shape1, shape2):
    if len(shape1) != len(shape2):
        return False

    for item1, item2 in zip(shape1, shape2):
        if item1 != item2:
            return False

    return True


# store variables
class variable_store(object):

    def __init__(self):
        self._vars = {}
        self._variable_scopes_count = {}

    def open_variable_scope(self, scope_name):
        if scope_name in self._variable_scopes_count:
            self._variable_scopes_count[scope_name] += 1
        else:
            self._variable_scopes_count[scope_name] = 1

    def close_variable_subscopes(self, scope_name):
        for k in self._variable_scopes_count:
            if not scope_name or k.startswith(scope_name + "/"):
                self._variable_scopes_count[k] = 0

    def variable_scope_count(self, scope_name):
        return self._variable_scopes_count.get(scope_name, 0)

    def get_variable(self, name, shape=None, dtype=None, initializer=None,
                     regularizer=None, reuse=None, trainable=True):
        initializing_from_value = False

        if initializer is not None and not callable(initializer):
            initializing_from_value = True

        if shape is not None and initializing_from_value:
            raise ValueError("if initializer is a constant, "
                             "do not specify shape.")

        should_check = reuse is not None
        dtype = dtype or theano.config.floatX

        # name already defined
        if name in self._vars:
            if should_check and not reuse:
                raise ValueError("variable %s already exists, disallowed." %
                                 name)

            found_var = self._vars[name]
            found_shape = found_var.get_value().shape

            if not is_compatible_shape(shape, found_shape):
                raise ValueError("trying to share variable %s, "
                                 "but specified shape %s and found shape %s."
                                 % (name, shape, found_shape))

            if dtype and dtype != found_var.dtype:
                raise ValueError("trying to share variable %s, but specified "
                                 "dtype %s and found dtype %s." %
                                 (name, dtype, found_var.dtype))
            return found_var

        # creating a new variable
        if should_check and reuse:
            raise ValueError("variable %s does not exist, or was not created "
                             "with get_variable()."  % name)

        # get default initializer
        if initializer is None:
            if is_floating_dtype(dtype):
                initializer = uniform_unit_scaling_initializer()
                initializing_from_value = False

            elif is_integer_dtype(dtype):
                initializer = zeros_initializer()
                initializer = initializer(shape=shape, dtype=dtype)
                initializing_from_value = True
            else:
                raise ValueError("a initializer for variable %s of %s "
                                 "is required" % (name, dtype))

        if initializing_from_value:
            init_val = initializer
        else:
            init_val = lambda: initializer(shape, dtype=dtype)

        # create variable
        v = variable(initial_value=init_val, name=name, trainable=trainable,
                     dtype=dtype)

        self._vars[name] = v

        if regularizer:
            with name_scope_op(name + "/regularizer/"):
                loss = regularizer(v)
                if loss is not None:
                    add_regularization_loss(loss)

        return v


class var_scope(object):

    def __init__(self, reuse, name="", initializer=None, regularizer=None,
                 name_scope="", dtype=None):
        self._name = name
        self._initializer = initializer
        self._regularizer = regularizer
        self._reuse = reuse
        self._name_scope = name_scope
        self._dtype = dtype

    @property
    def name(self):
        return self._name

    @property
    def original_name_scope(self):
        return self._name_scope

    @property
    def reuse(self):
        return self._reuse

    @property
    def initializer(self):
        return self._initializer

    @property
    def dtype(self):
        return self._dtype

    @property
    def regularizer(self):
        return self._regularizer

    def reuse_variables(self):
        self._reuse = True

    def set_initializer(self, initializer):
        self._initializer = initializer

    def set_dtype(self, dtype):
        self._dtype = dtype

    def set_regularizer(self, regularizer):
        self._regularizer = regularizer

    def get_variable(self, var_store, name, shape=None, dtype=None,
                     initializer=None, regularizer=None, trainable=True):
        if regularizer is None:
            regularizer = self._regularizer

        full_name = self.name + "/" + name if self.name else name

        with name_scope_op(None):
            if (dtype is not None and initializer is not None and
                not callable(initializer)):
                # initializer is a numpy object
                init_dtype = initializer.dtype
                if init_dtype != dtype:
                    raise ValueError("nitializer type '%s' and explicit"
                                     " dtype '%s'  don't match." %
                                     (init_dtype, dtype))

            if initializer is None:
                initializer = self._initializer
            if dtype is None:
                dtype = self._dtype

            return var_store.get_variable(full_name, shape=shape, dtype=dtype,
                                          reuse=self.reuse, trainable=trainable,
                                          initializer=initializer,
                                          regularizer=regularizer)


def get_variable_scope():
    # get_collection returns a list
    scope = get_collection(_VARSCOPE_KEY)
    if scope:
        # only 1 element in the list
        return scope[0]
    # create a new scope
    scope = var_scope(False)
    add_to_collection(_VARSCOPE_KEY, scope)

    return scope


def _get_default_variable_store():
    store = get_collection(_VARSTORE_KEY)
    if store:
        return store[0]
    # create a new store
    store = variable_store()
    add_to_collection(_VARSTORE_KEY, store)

    return store


def get_variable(name, shape=None, dtype=None, initializer=None,
                 regularizer=None, trainable=True):
    return get_variable_scope().get_variable(_get_default_variable_store(),
                                             name, shape=shape, dtype=dtype,
                                             initializer=initializer,
                                             regularizer=regularizer,
                                             trainable=trainable)



@contextlib.contextmanager
def _pure_variable_scope(name_or_scope, reuse=None, initializer=None,
                         regularizer=None, old_name_scope=None, dtype=None):
    # create a variable scope if not exsit
    get_variable_scope()
    # get var_scope object
    default_varscope = get_collection_ref(_VARSCOPE_KEY)

    old = default_varscope[0]

    # create a variable_store object if not exsit
    var_store = _get_default_variable_store()

    if isinstance(name_or_scope, var_scope):
        new_name = name_or_scope.name
    else:
        if old.name:
            new_name = old.name + "/" + name_or_scope
        else:
            new_name = name_or_scope

    try:
        var_store.open_variable_scope(new_name)

        if isinstance(name_or_scope, var_scope):
            scope = name_or_scope
            name_scope = scope._name_scope
            reuse = scope.reuse if reuse is None else reuse
            scope_initializer = scope.initializer
            scope_regularizer = scope.regularizer

            # set list element, create a new var_scope object
            default_varscope[0] = var_scope(reuse, name=new_name,
                                            initializer=scope_initializer,
                                            regularizer=scope_regularizer,
                                            dtype=scope.dtype,
                                            name_scope=name_scope)
        else:
            reuse = reuse or old.reuse
            old_initializer = old.initializer
            old_regularizer = old.regularizer
            name_scope = old_name_scope or name_or_scope
            default_varscope[0] = var_scope(reuse, name=new_name,
                                            initializer=old_initializer,
                                            regularizer=old_regularizer,
                                            dtype=old.dtype,
                                            name_scope=name_or_scope)
        # update
        if initializer is not None:
            default_varscope[0].set_initializer(initializer)
        if regularizer is not None:
            default_varscope[0].set_regularizer(regularizer)
        if dtype is not None:
            default_varscope[0].set_dtype(dtype)

        yield default_varscope[0]
    finally:
        var_store.close_variable_subscopes(new_name)
        default_varscope[0] = old


def _get_unique_variable_scope(prefix):
    var_store = _get_default_variable_store()
    current_scope = get_variable_scope()
    name = current_scope.name + "/" + prefix if current_scope.name else prefix
    if var_store.variable_scope_count(name) == 0:
        return prefix
    idx = 1
    while var_store.variable_scope_count(name + ("_%d" % idx)) > 0:
        idx += 1
    return prefix + ("_%d" % idx)


@contextlib.contextmanager
def variable_scope(name_or_scope, default_name=None, values=None,
                   initializer=None, regularizer=None, reuse=None, dtype=None):

    if default_name is None and name_or_scope is None:
      raise TypeError("if default_name is None then name_or_scope is required")

    if values is None:
        values = []

    if name_or_scope is not None:
        if not isinstance(name_or_scope, (var_scope,) + six.string_types):
            raise TypeError("variable_scope: name_or_scope must be a string or"
                            " var_scope.")
        if isinstance(name_or_scope, six.string_types):
            name_scope = name_or_scope
        else:
            name_scope = name_or_scope.name.split("/")[-1]

        if name_scope:
            with name_scope_op(name_scope) as cur_name_scope:
                if isinstance(name_or_scope, six.string_types):
                    old_name_scope = cur_name_scope
                else:
                    old_name_scope = name_or_scope.original_name_scope
            with _pure_variable_scope(name_or_scope, reuse=reuse,
                                      initializer=initializer,
                                      regularizer=regularizer,
                                      old_name_scope=old_name_scope,
                                      dtype=dtype) as vs:
                yield vs
        else:
            with _pure_variable_scope(name_or_scope, reuse=reuse,
                                    initializer=initializer,
                                    regularizer=regularizer,
                                    dtype=dtype) as vs:
                yield vs
    else:
        if reuse:
            raise ValueError("reuse=True cannot be used without a "
                             "name_or_scope")
        with name_scope_op(default_name) as scope:
            unique_default_name = _get_unique_variable_scope(default_name)
            with _pure_variable_scope(unique_default_name,
                                      initializer=initializer,
                                      regularizer=regularizer,
                                      old_name_scope=scope,
                                      dtype=dtype) as vs:
                yield vs
