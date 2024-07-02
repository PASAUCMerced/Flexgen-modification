

import hivemind
import logging

# # 设置日志级别为DEBUG
# logging.basicConfig(level=logging.DEBUG)

# # 创建初始对等点A，使用不同的端口
# dht_A = hivemind.DHT(
#     host_maddrs=["/ip4/0.0.0.0/tcp/50475", "/ip4/0.0.0.0/udp/58446/quic"],
#     start=True
# )

# # 打印初始对等点A的可见地址
# print("对等点A的地址:")
# print('\n'.join(str(addr) for addr in dht_A.get_visible_maddrs()))
# print("公共IP地址:", hivemind.utils.networking.choose_ip_address(dht_A.get_visible_maddrs()))

# import hivemind
# import logging

# # 设置日志级别为DEBUG
# logging.basicConfig(level=logging.DEBUG)
#         host_maddrs=["/ip4/0.0.0.0/tcp/40375", "/ip4/0.0.0.0/udp/48446/quic"],
initial_peers=[
   # IPv4 DNS addresses
   "chi-dyn-129-114-108-6.tacc.chameleoncloud.org",
   "chi-dyn-129-114-109-60.tacc.chameleoncloud.org",
   # Reserved IPs
   "/ip4/129.114.108.6/",
   "/ip4/129.114.109.60/"
]

try:
#     # 创建初始对等点B
    dht_B = hivemind.DHT(
        host_maddrs=["/ip4/0.0.0.0/tcp/40375", "/ip4/0.0.0.0/udp/48446/quic"],
        initial_peers=initial_peers,
        start=True
    )

#     # 打印初始对等点A的可见地址
    print("对等点B的地址:")
    print('\n'.join(str(addr) for addr in dht_B.get_visible_maddrs()))
#     print("公共IP地址:", hivemind.utils.networking.choose_ip_address(dht_B.get_visible_maddrs()))
except Exception as e:
    logging.error(f"Error starting DHT: {e}")


