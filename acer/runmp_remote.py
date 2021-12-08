import paramiko
import sys

_, server, command = sys.argv


with paramiko.SSHClient() as client:
    client.load_system_host_keys()
    client.connect(server)
    stdin, stdout, stderr = client.exec_command(command)

    for line in stdout.readline():
        print(line)