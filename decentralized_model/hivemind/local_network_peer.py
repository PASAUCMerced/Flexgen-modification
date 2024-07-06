import hivemind
import asyncio
from hivemind import DHT

# Define peer behavior
async def run_peer(node_ip, peer_id, port, initial_peers):
    # Start the DHT node
    dht_1 = DHT(start=True)
    print('\n'.join(str(addr) for addr in dht_1.get_visible_maddrs()))
    print("private IP:", hivemind.utils.networking.choose_ip_address(dht_1.get_visible_maddrs()))
    
    dht = await hivemind.dht.DHT(initial_peers=initial_peers,start=True)
    # Example function to run on the peer
    async def hello():
        return f"Hello from peer {peer_id}"

    # Declare the function in the DHT
    await dht.declare_function(hello, 'hello_function')

    # Call the function from another peer
    result = await dht.run_coroutine('hello_function')
    print(result)

# Main function to start the node
async def main():
    node_ip = '10.52.2.175'  # IP of the current node
    peer_id = 2
    port = 1235
    initial_peers = ['10.52.3.142:1234']  # IP and port of the other node

    await run_peer(node_ip, peer_id, port, initial_peers)

if __name__ == '__main__':
    asyncio.run(main())
