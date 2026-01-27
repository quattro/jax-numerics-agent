import abc
import equinox as eqx
import jax.numpy as jnp
from typing import Generic, TypeVar
from jaxtyping import Array, Int, PyTree


_State = TypeVar("_State")


class AbstractSolver(eqx.Module, Generic[_State]):
    rtol: eqx.AbstractVar[float]

    @abc.abstractmethod
    def init(self, y: PyTree[Array]) -> _State: ...

    @abc.abstractmethod
    def step(self, y: PyTree[Array], state: _State) -> tuple[PyTree[Array], _State]: ...


class SolverState(eqx.Module):
    count: Int[Array, ""]


class MySolver(AbstractSolver[SolverState]):
    rtol: float

    def init(self, y: PyTree[Array]) -> SolverState:
        return SolverState(count=jnp.array(0))

    def step(self, y: PyTree[Array], state: SolverState) -> tuple[PyTree[Array], SolverState]:
        return y, SolverState(count=state.count + 1)
