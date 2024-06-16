
import global_config
import compression


def fix_recursive_import():
    global_config.general_copy_compressed = compression.general_copy_compressed
    global_config.TorchCompressedDevice = compression.TorchCompressedDevice
