import socket

def find_free_port():
    # Create a socket with the address family and socket type
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to a random port
    sock.bind(('localhost', 0))
    
    # Get the allocated port
    _, port = sock.getsockname()
    
    # Close the socket
    sock.close()
    
    return port

# Usage:
free_port = find_free_port()
print(f"Free port: {free_port}")
