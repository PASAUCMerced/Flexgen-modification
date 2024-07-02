import torch
from hivemind import DHT, MAX_DHT_TIME_DISCREPANCY_SECONDS, BatchTensorDescriptor, get_dht_time
from hivemind.moe.server.layers import add_custom_models_from_file
from hivemind.moe.server.runtime import Runtime
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import get_logger
from transformers import PretrainedConfig

from flexgen.backend import DistOptLMBackend
from flexgen.handler import TransformerConnectionHandler

class ModuleContainer(threading.Thread):
    """Serves a set of specific Bloom layers for inference, forward, and backward. Announces itself over the DHT."""

    # noinspection PyMethodOverriding
    @classmethod
    def create(
        cls,
        *,
        dht: DHT,
        dht_prefix: str,
        converted_model_name_or_path: str,
        block_config: PretrainedConfig,
        attn_cache_bytes: int,
        alloc_timeout: float,
        server_info: ServerInfo,
        block_indices: List[int],
        min_batch_size: int,
        max_batch_size: int,
        max_chunk_size_bytes: int,
        torch_dtype: torch.dtype,
        cache_dir: str,
        max_disk_space: int,
        device: Union[str, torch.device],
        compression: CompressionType,
        update_period: float,
        expiration: Optional[float],
        revision: Optional[str],
        token: Optional[Union[str, bool]],
        quant_type: QuantType,
        tensor_parallel_devices: Sequence[torch.device],
        should_validate_reachability: bool,
        **kwargs,
    ) -> ModuleContainer:
        module_uids = [f"{dht_prefix}{UID_DELIMITER}{block_index}" for block_index in block_indices]
        memory_cache = MemoryCache(attn_cache_bytes, alloc_timeout)

        server_info.state = ServerState.JOINING
        dht_announcer = ModuleAnnouncerThread(
            module_uids,
            dht,
            server_info,
            block_config=block_config,
            memory_cache=memory_cache,
            update_period=update_period,
            expiration=expiration,
            daemon=True,
        )
        dht_announcer.start()
        logger.info(f"Announced that blocks {block_indices} are joining")

        assert len(tensor_parallel_devices) >= 1 and all(isinstance(d, torch.device) for d in tensor_parallel_devices)

        blocks = {}
        try:
            for module_uid, block_index in zip(module_uids, block_indices):
                block = load_pretrained_block(
                    converted_model_name_or_path,
                    block_index,
                    config=block_config,
                    torch_dtype=torch_dtype,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    max_disk_space=max_disk_space,
                )
                block = convert_block(
                    block,
                    block_index,
                    block_config,
                    tensor_parallel_devices,
                    device,
                    quant_type,
                    adapters=server_info.adapters,
                    freeze=True,
                    token=token,
                    cache_dir=cache_dir,
                    max_disk_space=max_disk_space,
                )
                blocks[module_uid] = TransformerBackend(
                    module_uid,
                    block,
                    config=block_config,
                    memory_cache=memory_cache,
                    backend_dtype=torch_dtype,
                    max_chunk_size_bytes=max_chunk_size_bytes,
                    args_schema=(
                        BatchTensorDescriptor(
                            1, 2048, block_config.hidden_size, dtype=torch_dtype, compression=compression
                        ),
                    ),
                    kwargs_schema={},
                    outputs_schema=(
                        BatchTensorDescriptor(
                            1, 2048, block_config.hidden_size, dtype=torch_dtype, compression=compression
                        ),
                    ),
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                )

            merge_inference_pools_inplace(blocks)

            if should_validate_reachability:
                validate_reachability(dht.peer_id)
        except:
            logger.debug("Shutting down backends")
            for backend in blocks.values():
                backend.shutdown()

            dht_announcer.announce(ServerState.OFFLINE)
            logger.info(f"Announced that blocks {module_uids} are offline")
            raise

        return cls(
            dht,
            dht_prefix,
            blocks,
            dht_announcer=dht_announcer,
            server_info=server_info,
            update_period=update_period,
            expiration=expiration,
            **kwargs,
        )

    def __init__(
        self,
        dht: DHT,
        dht_prefix: str,
        module_backends: Dict[str, TransformerBackend],
        *,
        inference_max_length: int,
        num_handlers: int,
        dht_announcer: ModuleAnnouncerThread,
        server_info: ServerInfo,
        update_period: float,
        expiration: Optional[float] = None,
        request_timeout: float,
        session_timeout: float,
        step_timeout: float,
        start: bool,
        **kwargs,
    ):
        super().__init__()

        self.dht, self.module_backends = dht, module_backends
        self.server_info, self.update_period, self.expiration = server_info, update_period, expiration

        handler_event_queues = [mp.Queue() for _ in range(num_handlers)]
        self.conn_handlers = [
            TransformerConnectionHandler(
                dht,
                self.module_backends,
                adapters=server_info.adapters,
                dht_prefix=dht_prefix,
                handler_event_queues=handler_event_queues,
                handler_index=i,
                inference_max_length=inference_max_length,
                request_timeout=request_timeout,
                session_timeout=session_timeout,
                step_timeout=step_timeout,
                quant_type=QuantType[server_info.quant_type.upper()],
            )
            for i in range(num_handlers)
        ]

        self.runtime = RuntimeWithDeduplicatedPools(self.module_backends, device=None, **kwargs)
        # note: We set device=None in runtime to avoid moving all modules to device 0 in runtime.run(). tensor_parallel has already moved it as needed.

        dht_announcer.announce(ServerState.ONLINE)
        self.dht_announcer = dht_announcer

        if start:
            self.run_in_background(await_ready=True)

    def run(self):
        """
        Runs ModuleContainer in the current thread. Initializes dht if necessary, starts connection handlers,
        runs Runtime (self.runtime) to process incoming requests.
        """
        for handler in self.conn_handlers:
            handler.run_in_background()

        self.runtime.run()

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts ModuleContainer in a background thread. if await_ready, this method will wait until the container
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError("ModuleContainer didn't notify .ready in {timeout} seconds")

    @property
    def ready(self) -> mp.synchronize.Event:
        """
        An event (multiprocessing.Event) that is set when the container is ready to process requests.

        Example
        =======
        >>> container.start()
        >>> container.ready.wait(timeout=10)
        >>> print("Container ready" if container.ready.is_set() else "Container didn't start in 10 seconds")
        """
        return self.runtime.ready  # mp.Event that is true if self is ready to process batches

    def is_healthy(self) -> bool:
        return all(handler.is_alive() for handler in self.conn_handlers) and all(
            pool.is_alive() for pool in self.runtime.pools
        )

    def shutdown(self):
        """
        Gracefully terminate the container, process-safe.
        Please note that terminating container otherwise (e.g. by killing processes) may result in zombie processes.
        If you did already cause a zombie outbreak, your only option is to kill them with -9 (SIGKILL).
        """
        self.dht_announcer.announce(ServerState.OFFLINE)
        logger.info(f"Announced that blocks {list(self.module_backends.keys())} are offline")

        self.ready.clear()

        logger.debug("Shutting down connection handlers")
        for handler in self.conn_handlers:
            handler.shutdown()

        logger.debug(f"Shutting down pools")
        for pool in self.runtime.pools:
            if pool.is_alive():
                pool.shutdown()

        logger.debug(f"Shutting down runtime")
        self.runtime.shutdown()

        logger.debug("Shutting down backends")
        for backend in self.module_backends.values():
            backend.shutdown()

        logger.info("Module container shut down successfully")