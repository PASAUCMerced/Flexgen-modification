import hivemind
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def run_node(is_initial_node, ip_address, initial_peer=None):
    host_maddrs = [f"/ip4/{ip_address}/tcp/0", f"/ip4/0.0.0.0/tcp/0"]
    
    if is_initial_node:
        # Node A: Initial node
        dht = await hivemind.DHT.create(
            host_maddrs=host_maddrs,
            start=True
        )
        print("Node A started")
    else:
        # Node B: Connecting to Node A
        dht = await hivemind.DHT.create(
            host_maddrs=host_maddrs,
            initial_peers=[initial_peer] if initial_peer else None,
            start=True
        )
        print("Node B started and attempting to connect to Node A")

    visible_maddrs = dht.get_visible_maddrs()
    print("Node address:", visible_maddrs)
    return dht, visible_maddrs

async def main(is_initial_node, ip_address, initial_peer=None):
    try:
        dht, visible_maddrs = await run_node(is_initial_node, ip_address, initial_peer)
        
        if is_initial_node:
            print("For Node B, use one of these addresses:")
            for addr in visible_maddrs:
                if not addr.startswith("/ip4/127.0.0.1") and not addr.startswith("/ip4/192.168."):
                    print(addr)
        
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down node...")
    finally:
        if 'dht' in locals():
            await dht.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a Hivemind DHT node")
    parser.add_argument("--initial", action="store_true", help="Run as the initial node")
    parser.add_argument("--ip", required=True, help="IP address of this node")
    parser.add_argument("--peer", help="Address of the initial peer (for non-initial nodes)")
    args = parser.parse_args()

    asyncio.run(main(is_initial_node=args.initial, ip_address=args.ip, initial_peer=args.peer))