import paramiko
import sys

server, username, key, command = sys.argv[1:]


print(command)
# command = "pwd;whoami"

with paramiko.SSHClient() as client:
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, username=username, key_filename=key)
    transport = client.get_transport()
    transport.set_keepalive(1)
    stdin, stdout, stderr = client.exec_command(command)

    line = []
    while not stdout.channel.exit_status_ready():
        char = stdout.read(1).decode('ascii')
        if char == '\n':
            print(''.join(line))
            line = []
        else:
            line.append(char)

    print(''.join(line))

    exit(stdout.channel.recv_exit_status())
