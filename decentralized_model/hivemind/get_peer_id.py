import hivemind

# Start a DHT node
dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    start=True
)

# Print the peer ID
print(f"Peer ID: {dht.peer_id}")
print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))

# Peer ID: 12D3KooWHLeoUQJv4j9PNKKLwXtet8X28WK2DHCZ8kPGNyre8er2
# /ip4/10.52.2.175/tcp/46049/p2p/12D3KooWHLeoUQJv4j9PNKKLwXtet8X28WK2DHCZ8kPGNyre8er2
# /ip4/127.0.0.1/tcp/46049/p2p/12D3KooWHLeoUQJv4j9PNKKLwXtet8X28WK2DHCZ8kPGNyre8er2
# /ip4/10.52.2.175/udp/59179/quic/p2p/12D3KooWHLeoUQJv4j9PNKKLwXtet8X28WK2DHCZ8kPGNyre8er2
# /ip4/127.0.0.1/udp/59179/quic/p2p/12D3KooWHLeoUQJv4j9PNKKLwXtet8X28WK2DHCZ8kPGNyre8er2
# Global IP: 10.52.2.175


# Run the Script on Each Node
# INITIAL_PEERS = [
#     "/ip4/129.114.109.60/tcp/38461/p2p/12D3KooWHLeoUQJv4j9PNKKLwXtet8X28WK2DHCZ8kPGNyre8er2",
#     "/ip4/129.114.109.60/udp/55744/quic/p2p/12D3KooWHLeoUQJv4j9PNKKLwXtet8X28WK2DHCZ8kPGNyre8er2",
#     "/ip4/129.114.108.6/tcp/38461/p2p/12D3KooWQiEzzdfQWB5hSuqQHoWWwVKUzPp951mUQtxGEvFWCVrv",
#     "/ip4/129.114.108.6/udp/55744/quic/p2p/12D3KooWQiEzzdfQWB5hSuqQHoWWwVKUzPp951mUQtxGEvFWCVrv"
# ]