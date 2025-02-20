""" WIP """
from types import SimpleNamespace
import jax
from jax.tree_util import register_pytree_node
import equinox as eqx

class Tape(dict):
    def __init__(self, *args, **kwargs):
        super(Tape, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

register_pytree_node(Tape, lambda tree: tuple(tree.__dict__.values()), lambda _, args: Tape(*args, **kwargs))

def new_tape_type(keys):
    vars_abstract = {k: None for k in keys}
    Tape = type('Tape', (eqx.Module,), vars_abstract | {"__annotations__": vars_abstract }| {"__getitem__": lambda self, k: getattr(self, k)})
    return Tape

if __name__ == "__main__":
    tape = Tape(a=3, b=7)
    tape.a
    tape.b

    tape = jax.tree.map(lambda x: x + 1, tape)
