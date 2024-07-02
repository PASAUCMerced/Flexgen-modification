import asyncio
import contextlib
import multiprocessing as mp
import sys
from enum import Enum
from itertools import chain
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from async_timeout import timeout

from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import amap_in_executor, anext
from hivemind import (
    DHT,
    MSGPackSerializer,
    P2PContext,
    PeerID,
    deserialize_tensor_stream,
    deserialize_torch_tensor,
    nested_flatten,
    nested_pack,
    serialize_torch_tensor,
)
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE
from hivemind.utils.streaming import split_for_streaming
from flexgen.backend import DistOptLMBackend
from abc import ABC, abstractmethod
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, TorchTensor)

ModuleUID = str

class TaskPrioritizerBase(ABC):
    """Abstract class for TaskPrioritizer whose responsibility is to evaluate task priority"""
    @abstractmethod
    def prioritize(self, *input: TorchTensor, points: float = 0.0, **kwargs) -> float:
        """Evaluates task value by the amount of points given, task input and additional kwargs. Lower priority is better"""
        pass

class DummyTaskPrioritizer(TaskPrioritizerBase):
    def prioritize(self, *input: torch.Tensor, points: float = 0.0, **kwargs) -> float:
        # Inference steps (especially short ones) go first since they are more latency-sensitive
        if kwargs.get("type") == "short_inference":
            return 1.0
        if kwargs.get("type") == "inference":
            return 2.0
        return 3.0  # Forward, backward
    

class TransformerConnectionHandler(ConnectionHandler):
    """Handles three request types: forward, backward and forward-incremental (inference)"""

    module_backends: Dict[ModuleUID, DistOptLMBackend]

    def __init__(
        self,
        dht: DHT,
        module_backends: Dict[str, DistOptLMBackend],
        *,
        adapters: Optional[Sequence[str]],
        dht_prefix: str,
        handler_event_queues: Sequence[mp.Queue],
        handler_index: int,
        inference_max_length: int,
        request_timeout: float,
        session_timeout: float,
        step_timeout: float,
        task_prioritizer: TaskPrioritizerBase = DummyTaskPrioritizer(),
    ):
        super().__init__(dht, module_backends)
        for module_backend in self.module_backends.values():
            assert isinstance(module_backend, DistOptLMBackend)
        self.dht_prefix = dht_prefix
        self.adapters = adapters
        self._handler_event_queues = handler_event_queues
        self._handler_index = handler_index
        self._own_event_queue = handler_event_queues[handler_index]
        self._listener_task: Optional[asyncio.Task] = None
        self._session_queues: Dict[str, asyncio.Queue] = {}
        self._session_handlers: Dict[str, int] = {}

        self.inference_max_length = inference_max_length
        self.request_timeout = request_timeout
        self.session_timeout, self.step_timeout = session_timeout, step_timeout
        self._prioritizer = task_prioritizer

    async def add_p2p_handlers(self, *args, **kwargs) -> None:
        if self._listener_task is None:
            # Start listening to our own event queue before we accept any requests
            self._listener_task = asyncio.create_task(self._listen_to_event_queue())
        await super().add_p2p_handlers(*args, **kwargs)

    async def rpc_generate(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        async with timeout(self.request_timeout):
            # Parse request and prepare backends
            flat_inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
            requested_uids = self._check_uids(request.uid)
            self._log_request("rpc_generate", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
            active_adapter = self._get_active_adapter(metadata)
            points = metadata.get("points", 0)
            assert isinstance(
                points, (float, int)
            ), f"rpc_generate should have number of points as number or None, got {points}"

            hidden_states = await run_rpc_generation_task(
                *flat_inputs,
                requested_backends=requested_backends,
                prioritizer=self._prioritizer,
                active_adapter=active_adapter,
                points=points,
            )
            return runtime_pb2.ExpertResponse(
                tensors=self._serialize_outputs(hidden_states, requested_backends, metadata)
            )

    async def rpc_generate_stream(self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        async with timeout(self.request_timeout):
            # Parse requests and prepare backends
            uid_str, flat_inputs, metadata = await self._gather_inputs(requests, context)
            requested_uids = self._check_uids(uid_str)
            self._log_request("rpc_forward_stream", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            active_adapter = self._get_active_adapter(metadata)
            points = metadata.get("points", 0)
            assert isinstance(
                points, (float, int)
            ), f"rpc_forward_stream should have number of points as number or None, got {points}"

            hidden_states = await run_rpc_generation_task(
                *flat_inputs,
                requested_backends=requested_backends,
                prioritizer=self._prioritizer,
                active_adapter=active_adapter,
                points=points,
            )

            # Split the serialized_output for streaming and respond to client
            for tensor in self._serialize_outputs(hidden_states, requested_backends, metadata):
                for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE):
                    yield runtime_pb2.ExpertResponse(tensors=[part]) 


## from Petals block_functions.py
## Run generation pass on deserialized inputs and prompts, 
## used by rpc_forward and rpc_forward_stream
async def run_rpc_generation_task(
    *flat_tensors: TorchTensor,
    requested_backends: Sequence[DistOptLMBackend],
    active_adapter: str = "",
    prioritizer: TaskPrioritizerBase,
    points: int = 0,
    args_structure: Any = None,
) -> TorchTensor:
    # if args_structure is not None:
    # # TODO: kwargs currently is unused, it can be used later for peft-like adaptation
    #     flat_tensors, kwargs = unpack_args_kwargs(flat_tensors, args_structure)
    hidden_states, prompts, *_ = flat_tensors

    dtype = requested_backends[0].dtype
    # check parse input tensors and cast dtypes
    hidden_states = hidden_states.to(dtype)
    # assert hidden_states.ndim == 3
    # if prompts is None or is_dummy(prompts):
    #     prompts = [DUMMY] * len(requested_backends)
    # else:
    #     prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]
    prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]

    # Run a chain of requested backends
    for backend, prompt in zip(requested_backends, prompts):
        # if not is_dummy(prompt):
        #     hidden_states[:, : prompt.shape[1]] += prompt
        hidden_states[:, : prompt.shape[1]] += prompt

        # assert isinstance(backend.generation_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            hidden_states, points=points / len(requested_backends), backend=backend, type="generation"
        )
        (hidden_states,) = await backend.generation_pool.submit_task(
            hidden_states,
            active_adapter,
            priority=priority,
        )
        assert isinstance(hidden_states, TorchTensor)
        assert (
            hidden_states.ndim == 3
        ), f"inputs to {type(backend)} must be a list with a single 3d tensor of hidden states"

    return hidden_states