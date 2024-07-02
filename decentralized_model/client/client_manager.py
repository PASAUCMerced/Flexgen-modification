from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
import random
import threading
import time
from typing import Any, Collection, Dict, List, Optional, Sequence, Union
from weakref import WeakMethod

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional, Union

from flexgen.config import ClientConfig
from flexgen.handler import TransformerConnectionHandler
# from flexgen.utils.ping import PingAggregator
from flexgen.share_utils.ping import PingAggregator
from flexgen.share_utils.sequence_info import RemoteSequenceInfo

#from config import ClientConfig
#from handler import TransformerConnectionHandler

import dijkstar
import numpy as np
from torch import nn
from hivemind import DHT, P2P, MSGPackSerializer, PeerID
from hivemind.dht.node import Blacklist
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

ModuleUID = str
UID_DELIMITER = "."  # delimits parts of one module uid, e.g. "bloom.transformer.h.4.self_attention"
CHAIN_DELIMITER = " "  # delimits multiple uids in a sequence, e.g. "bloom.layer3 bloom.layer4"

# PUBLIC_INITIAL_PEERS = [
#     # IPv4 DNS addresses
#     "ec2-54-153-80-127.us-west-1.compute.amazonaws.com",
#     "ec2-54-193-136-27.us-west-1.compute.amazonaws.com",
#     # Reserved IPs
#     "/ip4/54.193.136.27/",
#     "/ip4/54.153.80.127/"]

PUBLIC_INITIAL_PEERS = ["/ip4/172.31.26.141/tcp/32977/p2p/12D3KooWFRwayM6VrNFwAhyA4MGbxsZW6yvQgmVCxwRVkxrSRsgS", 
                        "/ip4/172.31.26.141/udp/54124/quic/p2p/12D3KooWFRwayM6VrNFwAhyA4MGbxsZW6yvQgmVCxwRVkxrSRsgS"]


@dataclasses.dataclass
class SequenceManagerConfig:
    initial_peers: Sequence[str] = tuple(PUBLIC_INITIAL_PEERS)  # a list of initial peers for hivemind DHT
    dht_prefix: Optional[str] = None  # a prefix for all dht keys that correspond to this model (default: model name)
    daemon_startup_timeout: int = 60  # timeout for the libp2p daemon connecting to initial peers

    show_route: Union[str, bool] = "inference"  # show chosen route through servers. one of [False, "inference", True]
    allowed_servers: Optional[Collection[Union[PeerID, str]]] = None  # if defined, send requests only to these servers
    use_server_to_server: bool = True  # Use direct server-to-server communication

    connect_timeout: float = 5  # timeout for opening a connection
    request_timeout: float = 3 * 60  # timeout for forward/backward/inference requests
    update_period: float = 60  # refresh DHT information once in this many seconds

    max_retries: Optional[int] = None  # max number retries before the client raises an exception (default: inf)
    min_backoff: float = 1  # after a repeated failure, sleep for this many seconds times 2 ** (num_failures - 1)
    max_backoff: float = 60  # limit maximal sleep time between retries to this value
    ban_timeout: float = 15  # when a remote peer fails to respond, prevent routing to that peer for this many seconds
    active_adapter: Optional[str] = None  # name of active LoRA adapter (usually, Hugging Face repo)

    max_pinged: int = 5  # max servers to ping from each sequence side, per update
    ping_timeout: float = 2  # max time to wait for pings, per update

class SequenceManagerState:
    p2p: P2P = None
    sequence_info: Optional[RemoteSequenceInfo] = None
    rpc_info: Optional[dict] = None
    banned_peers: Optional[Blacklist] = None

    def __getitem__(self, ix: Union[int, slice]) -> SequenceManagerState:
        return dataclasses.replace(self, sequence_info=self.sequence_info[ix])

    def __len__(self) -> int:
        return len(self.sequence_info)
      
class RemoteSequenceManager:
    def __init__(
        self,
        config: SequenceManagerConfig,
        block_uids: Sequence[ModuleUID],
        *,
        dht: Optional[DHT] = None,
        state: Optional[SequenceManagerState] = None,
    ):
        assert config.initial_peers or dht is not None, "Please specify `config.initial_peers` or `dht`"
        assert config.dht_prefix, "Could not find dht_prefix in config, please create model with dht_prefix=..."
        assert len(block_uids) > 0, "Sequences must contain at least one block"

        self.config = config
        if state is None:
            state = SequenceManagerState()
        self.state = state

        if dht is None:
            dht = DHT(
                initial_peers=config.initial_peers,
                client_mode=True,
                num_workers=32,
                startup_timeout=config.daemon_startup_timeout,
                start=True,
            )
        assert isinstance(dht, DHT) and dht.is_alive(), "`dht` must be a running hivemind.DHT instance"
        self.dht = dht

        if state.p2p is None:
            state.p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p())

        self.lock_changes = threading.Lock()
        self._thread = _SequenceManagerUpdateThread(config.update_period, WeakMethod(self._update))
        self._thread_start_lock = threading.Lock()
        # self.policy = NoSpendingPolicy()

        self.ping_aggregator = PingAggregator(dht)

        if state.banned_peers is None:
            state.banned_peers = Blacklist(base_time=config.ban_timeout, backoff_rate=2.0)
        if state.sequence_info is None:
            state.sequence_info = RemoteSequenceInfo.make_empty(block_uids)

        if state.sequence_info.last_updated_time is not None:
            assert block_uids == state.sequence_info.block_uids
            self._thread.ready.set()  # no need to await the first dht fetch
            self._need_latest_infos = True

    def make_sequence(
        self,
        start_index: int = 0,
        end_index: Optional[int] = None,
        *,
        mode: str,
        cache_tokens_needed: Optional[int] = None,
    ) -> List[RemoteSpanInfo]:
        """
        Form a sequence of remote servers that collectively serve all consecutive layers

        :param start_index: optional index of the first module in a sequence, default = the first of block_uids
        :param end_index: optional index of the last module (non-inclusive), default = after last of block uids
        :param mode: one of ["max_throughput", "min_latency"]
        """
        with self._thread_start_lock:
            if not self.is_alive():
                self._thread.start()
        if not self.ready.is_set():
            self.update(wait=True)  # this will await an existing update or trigger a new one (if not updating)

        end_index = end_index if end_index is not None else len(self)

        if mode == "min_latency":
            span_sequence = self._make_sequence_with_min_latency(
                start_index, end_index, cache_tokens_needed=cache_tokens_needed
            )
        elif mode == "max_throughput":
            span_sequence = self._make_sequence_with_max_throughput(start_index, end_index)
        else:
            raise RuntimeError(f"Unexpected mode {mode}")

        if self.config.show_route is True or (mode == "min_latency" and self.config.show_route == "inference"):
            route_repr = " => ".join(
                [f"{span.start}:{span.end} via …{str(span.peer_id)[-6:]}" for span in span_sequence]
            )
            logger.info(f"Route found: {route_repr}")
        return span_sequence

    def __getitem__(self, ix: Union[int, slice]) -> RemoteSequenceManager:
        """Get a RemoteSequenceManager for a sub-sequence of blocks"""
        assert isinstance(ix, (int, slice))
        if not isinstance(ix, slice):
            ix = slice(int(ix), int(ix) + 1, 1)
        return type(self)(self.config, self.block_uids[ix], dht=self.dht, state=self.state[ix])

    def update(self, *, wait: bool):
        """Run an asynchronous update in background as soon as possible"""
        self.ready.clear()
        self._thread.trigger.set()
        if wait:
            self.ready.wait()

    def _update(self):
        """Perform an immediate and synchronous refresh, may take time"""

        new_block_infos = petals.dht_utils.get_remote_module_infos(
            self.dht, self.block_uids, active_adapter=self.config.active_adapter, latest=True
        )

        for block_info in new_block_infos:
            if not block_info:
                continue

            # Apply whitelist, if defined
            if self.config.allowed_servers is not None:
                block_info.servers = {
                    peer_id: server_info
                    for peer_id, server_info in block_info.servers.items()
                    if peer_id in self.config.allowed_servers or str(peer_id) in self.config.allowed_servers
                }

            # Remove temporarily banned peers, unless there are no peers left
            valid_servers = {
                peer_id: server_info
                for peer_id, server_info in block_info.servers.items()
                if peer_id not in self.state.banned_peers
            }
            if len(valid_servers) < len(block_info.servers):
                if valid_servers:
                    logger.debug(
                        f"Kept {len(valid_servers)} out of {len(block_info.servers)} servers holding {block_info.uid}"
                    )
                    block_info.servers = valid_servers

                else:
                    # If we blacklisted all servers, the error may actually be client-caused
                    logger.debug(f"All servers holding {block_info.uid} are blacklisted, ignoring blacklist")

        with self.lock_changes:
            self.state.sequence_info.update_(new_block_infos)

            first_servers = [span.peer_id for span in self.state.sequence_info.spans_containing_block[0]]
            last_servers = [span.peer_id for span in self.state.sequence_info.spans_containing_block[-1]]

        pinged_servers = set(sample_up_to(first_servers, self.config.max_pinged))
        pinged_servers |= set(sample_up_to(last_servers, self.config.max_pinged))
        self.ping_aggregator.ping(list(pinged_servers), wait_timeout=self.config.ping_timeout)

        self.ready.set()

    def on_request_failure(self, peer_id: Optional[PeerID]):
        """remove a given peer from the routing table. If the routing is no longer possible, trigger an update"""
        if peer_id is not None:
            logger.debug(f"Peer {peer_id} did not respond, banning it temporarily")
            self.state.banned_peers.register_failure(peer_id)
        with self.lock_changes:
            should_update = False
            for info in self.state.sequence_info.block_infos:
                info.servers.pop(peer_id, None)
                if not info.servers:
                    should_update = True
            if should_update:
                self.ready.clear()
                self.update(wait=False)

    def on_request_success(self, peer_id: PeerID):
        """if peer has a failure streak, clear that streak"""
        self.state.banned_peers.register_success(peer_id)

    def __len__(self):
        return len(self.block_uids)

    @property
    def is_alive(self):
        return self._thread.is_alive

    @property
    def ready(self) -> threading.Event:
        return self._thread.ready

    @property
    def block_uids(self):
        return self.state.sequence_info.block_uids

    @property
    def rpc_info(self):
        """Return the rpc_info queried from one of the servers that hold the first block"""
        if self.state.rpc_info is not None:
            return self.state.rpc_info

        with self._thread_start_lock:
            if not self.is_alive():
                self._thread.start()

        for attempt_no in itertools.count():
            peer_id = None
            try:
                if not self.ready.is_set():
                    self.update(wait=True)

                active_servers = [
                    peer_id
                    for peer_id, server in self.state.sequence_info.block_infos[0].servers.items()
                    if server.state == ServerState.ONLINE
                ]
                if not active_servers:
                    raise MissingBlocksError(0)
                peer_id = random.choice(active_servers)

                stub = TransformerConnectionHandler.get_stub(self.state.p2p, peer_id)
                outputs = RemoteExpertWorker.run_coroutine(
                    stub.rpc_info(runtime_pb2.ExpertUID(uid=self.block_uids[0]), timeout=self.config.request_timeout)
                )
                self.state.rpc_info = MSGPackSerializer.loads(outputs.serialized_info)
                self.on_request_success(peer_id)
                break
            except Exception as e:
                self.on_request_failure(peer_id)
                if attempt_no + 1 == self.config.max_retries:
                    raise
                delay = self.get_retry_delay(attempt_no)
                logger.warning(
                    f"Caught exception when gathering information from peer {peer_id} "
                    f"(retry in {delay:.0f} sec): {repr(e)}"
                )
                maybe_log_traceback(e)
                time.sleep(delay)

        return self.state.rpc_info

    def get_retry_delay(self, attempt_no: int) -> float:
        if attempt_no == 0:
            return 0
        return min(self.config.min_backoff * 2 ** (attempt_no - 1), self.config.max_backoff)


    def get_request_metadata(self, protocol: str, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        :param protocol: one of "rpc_forward", "rpc_backward" or "rpc_inference"
        :param args: request-specific inputs, typically block uids and input tensors
        :param kwargs: additional request context, such as remote peer ID
        :returns: msgpack-serialized metadata dict that will be passed alongside a given request
        """
        # return dict(points=self.policy.get_points(protocol, *args, **kwargs), active_adapter=self.config.active_adapter)
        return dict(0.0, active_adapter=self.config.active_adapter)

    def shutdown(self):
        self._thread.shutdown()


class RemoteSequential(nn.Module):
    """
    A sequence of transformer blocks hosted by the swarm.
    """

    def __init__(
        self,
        config: ClientConfig,
        *,
        sequence_manager: Optional[RemoteSequenceManager] = None,
        dht: Optional[DHT] = None,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        assert sequence_manager is None or (
            dht is None and start_block is None and end_block is None
        ), "`dht`, `start_block`, and `end_block` have no effect when you provide a custom `sequence_manager`"
        if sequence_manager is None:
            if start_block is None:
                start_block = 0
            if end_block is None:
                end_block = self.config.num_hidden_layers
            block_uids = tuple(f"{config.dht_prefix}{UID_DELIMITER}{i}" for i in range(start_block, end_block))
            sequence_manager = RemoteSequenceManager(config, block_uids, dht=dht, **kwargs)
        self.sequence_manager = sequence_manager

        self._active_session = ContextVar("active_session", default=None)

    def forward(self, inputs: torch.Tensor, prompts: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        assert inputs.ndim == 3, "inputs must be a tensor of shape [batch_size, seq_length, hidden_size]"
        if self.active_session is None:
            assert all(v is None for v in kwargs.values()), f"Extra kwargs are not supported in forward: {kwargs}"
            return _RemoteSequentialAutogradFunction.apply(inputs, prompts, self.sequence_manager)
        else:
            return self.active_session.step(inputs, prompts, **kwargs)

    @property
    def active_session(self) -> Optional[InferenceSession]:
        """
        If called inside `with model.inference_session(...):` or `with model.use_session(...):`,
        returns an active InferenceSession. Otherwise, returns None.
        """

        return self._active_session.get()

    @property
    def position(self) -> int:
        """Returns the prefix length (in tokens) in the active inference session or zero if no session is active."""

        return self.active_session.position if self.active_session is not None else 0

    @contextmanager
    def use_session(self, session: Optional[InferenceSession]) -> InferenceSession:
        """Inside this context, forward() will use an _existing_ InferenceSession provided as the argument."""

        token = self._active_session.set(session)
        try:
            yield session
        finally:
            self._active_session.reset(token)

    @contextmanager
    def inference_session(self, **kwargs) -> InferenceSession:
        """
        Inside this context, forward() will use a _new_ InferenceSession created with given parameters.

        :param max_length: Maximal expected length of inference results. Servers use this parameter
                           to calculate the size of attention caches allocated to this client.
        """

        with InferenceSession(self.sequence_manager, **kwargs) as session, self.use_session(session):
            yield session

    def __getitem__(self, ix: Union[int, slice]) -> RemoteSequential:
        return RemoteSequential(
            self.config,
            sequence_manager=self.sequence_manager[ix],
        )

    def __iter__(self):
        for block_index in range(len(self)):
            yield self[block_index]

    def __len__(self):
        return len(self.sequence_manager)

    def extra_repr(self) -> str:
        return f"modules={self.sequence_manager.block_uids[0]}..{self.sequence_manager.block_uids[-1]}"

class _SequenceManagerUpdateThread(threading.Thread):
    def __init__(self, update_period: float, ref_update_manager: WeakMethod):
        super().__init__(daemon=True)
        self.ref_update_manager = ref_update_manager
        self.ready = threading.Event()
        self.trigger = threading.Event()
        self.update_period = update_period
        self.should_shutdown = False

    def run(self) -> None:
        while not self.should_shutdown:
            update_manager = self.ref_update_manager()
            if update_manager is None:
                logger.debug(f"{self.__class__.__name__} exited because the sequence manager no longer exists")
                break

            try:
                self.trigger.clear()
                update_manager()
            except Exception as e:
                logger.exception(e)
            finally:
                del update_manager

            self.trigger.wait(self.update_period)

        logger.debug(f"{self.__class__.__name__} thread exited")

    def shutdown(self, timeout: Optional[float] = None):
        self.should_shutdown = True
        self.trigger.set()
        if self.is_alive():
            self.join(timeout)

    def __del__(self):
        self.shutdown()
