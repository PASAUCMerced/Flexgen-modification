import hivemind
from time import sleep

INITIAL_PEERS=["/ip4/127.0.0.1/tcp/37857/p2p/12D3KooWQtkEzJU3VosFutKLVHcXwSo45XhgiMbZjME8pKDjS3Fr",]
# Start a DHT node
dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    initial_peers=INITIAL_PEERS,
    start=True,
    # use_ipfs=True,
)

# Print the peer ID
print(f"Peer ID: {dht.peer_id}")
# print(dht.get_visible_maddrs())
print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))

sleep(60)
# sleep(6000)



# cuurent node
# Peer ID: 12D3KooWMFFECWyC8jbLKWxBPNcUTLhsDXks9NMjVSaLBaJdcKdS
# /ip4/10.52.2.175/tcp/36431/p2p/12D3KooWMFFECWyC8jbLKWxBPNcUTLhsDXks9NMjVSaLBaJdcKdS
# /ip4/127.0.0.1/tcp/36431/p2p/12D3KooWMFFECWyC8jbLKWxBPNcUTLhsDXks9NMjVSaLBaJdcKdS
# /ip4/10.52.2.175/udp/50716/quic/p2p/12D3KooWMFFECWyC8jbLKWxBPNcUTLhsDXks9NMjVSaLBaJdcKdS
# /ip4/127.0.0.1/udp/50716/quic/p2p/12D3KooWMFFECWyC8jbLKWxBPNcUTLhsDXks9NMjVSaLBaJdcKdS
# Global IP: 10.52.2.175

# peer node
# /ip4/10.52.3.142/tcp/43063/p2p/12D3KooWL13ymiiTLC9E7xWBgs1EzQySDxBt4LARz4HhXXBL6pZu
# /ip4/127.0.0.1/tcp/43063/p2p/12D3KooWL13ymiiTLC9E7xWBgs1EzQySDxBt4LARz4HhXXBL6pZu
# /ip4/10.52.3.142/udp/47912/quic/p2p/12D3KooWL13ymiiTLC9E7xWBgs1EzQySDxBt4LARz4HhXXBL6pZu
# /ip4/127.0.0.1/udp/47912/quic/p2p/12D3KooWL13ymiiTLC9E7xWBgs1EzQySDxBt4LARz4HhXXBL6pZu
# Global IP: 10.52.3.142

# Run the Script on Each Node
# on node ip address is 129.114.109.60
# INITIAL_PEERS = [
#     "/ip4/129.114.108.6/tcp/38461/p2p/12D3KooWL13ymiiTLC9E7xWBgs1EzQySDxBt4LARz4HhXXBL6pZu",
#     "/ip4/129.114.108.6/udp/55744/quic/p2p/12D3KooWL13ymiiTLC9E7xWBgs1EzQySDxBt4LARz4HhXXBL6pZu"
# ]


# on node ip address is 129.114.109.6
# INITIAL_PEERS = [
#     "/ip4/129.114.109.60/tcp/38461/p2p/12D3KooWMFFECWyC8jbLKWxBPNcUTLhsDXks9NMjVSaLBaJdcKdS",
#     "/ip4/129.114.109.60/udp/55744/quic/p2p/12D3KooWMFFECWyC8jbLKWxBPNcUTLhsDXks9NMjVSaLBaJdcKdS",
# ]