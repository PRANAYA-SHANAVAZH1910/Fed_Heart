import socket

HOST = "0.0.0.0"
PORT = 8080

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Listening on {HOST}:{PORT}")

    conn, addr = server.accept()
    print("Connected by", addr)

    while True:
        data = conn.recv(1024)
        if not data:
            break
        print("Client:", data.decode())
        conn.sendall(b"Message received")

    conn.close()
finally:
    server.close()