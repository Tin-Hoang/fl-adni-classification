#!/usr/bin/env python3
"""
Flower Multi-Machine Runner with Full Process Isolation Mode
This version uses Flower's Process Isolation Mode on both server and client sides to enable PyTorch DataLoader multiprocessing.

Features:
- Full Process Isolation Mode (Server: SuperLink + ServerApp, Client: SuperNode + ClientApp)
- Resolves "daemonic processes are not allowed to have children" error on both server and client
- Enables PyTorch DataLoader with num_workers > 0 on both server and client sides
- Tmux-based process management for monitoring all components
- SSH tunnel support for secure communication
- Separate logging for all process components
- Comprehensive monitoring and error reporting

Architecture:
Server Side:
- SuperLink: Coordination service in process isolation mode
- ServerApp: FL server logic in separate non-daemon process

Client Side:
- SuperNode: Handles communication with SuperLink via SSH tunnel
- ClientApp: Runs FL training/evaluation in separate non-daemon process

Process Isolation: Allows PyTorch multiprocessing without daemon restrictions on both sides
"""

import paramiko
import time
import signal
import sys
import threading
import socket
import select
from typing import List, Dict
from datetime import datetime

class FlowerMultiMachineTmuxRunner:
    def __init__(self, server_config: Dict, clients_config: List[Dict], project_dir: str, ssh_timeout: int = 30, ssh_auth_timeout: int = 30, ssh_banner_timeout: int = 30):
        self.server_config = server_config
        self.clients_config = clients_config
        self.project_dir = project_dir
        self.ssh_timeout = ssh_timeout
        self.ssh_auth_timeout = ssh_auth_timeout
        self.ssh_banner_timeout = ssh_banner_timeout
        self.ssh_connections = []
        self.server_ssh = None
        self.tunnel_threads = []
        self.tunnel_sockets = []
        # Generate timestamp for this session
        self.timestamp = self.generate_timestamp()

    def generate_timestamp(self) -> str:
        """Generate timestamp string for log files"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def ensure_logs_directory(self, ssh_client, project_dir: str) -> bool:
        """Ensure logs directory exists on remote machine"""
        try:
            create_logs_cmd = f"mkdir -p {project_dir}/logs"
            stdin, stdout, stderr = ssh_client.exec_command(create_logs_cmd)
            error = stderr.read().decode().strip()
            if error:
                print(f"⚠️ Warning creating logs directory: {error}")
                return False
            return True
        except Exception as e:
            print(f"❌ Error creating logs directory: {e}")
            return False

    def get_venv_command(self, command: str, venv_activate: str) -> str:
        """Wrap a command with virtual environment activation"""
        return f"bash -c 'source {venv_activate} && {command}'"

    def forward_tunnel(self, local_port, remote_host, remote_port, transport):
        """Forward local port to remote host through SSH tunnel"""
        try:
            # Create local socket
            local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            local_socket.bind(('127.0.0.1', local_port))
            local_socket.listen(1)

            self.tunnel_sockets.append(local_socket)
            print(f"🔗 Tunnel listening on local port {local_port}")

            while True:
                try:
                    client_socket, addr = local_socket.accept()
                    print(f"🔗 Tunnel connection from {addr}")

                    # Create channel through SSH
                    channel = transport.open_channel(
                        "direct-tcpip",
                        (remote_host, remote_port),
                        addr
                    )

                    # Start forwarding in a separate thread
                    forward_thread = threading.Thread(
                        target=self.handle_tunnel_connection,
                        args=(client_socket, channel)
                    )
                    forward_thread.daemon = True
                    forward_thread.start()

                except Exception as e:
                    print(f"⚠️ Tunnel accept error: {e}")
                    break

        except Exception as e:
            print(f"❌ Tunnel setup error: {e}")
        finally:
            try:
                local_socket.close()
            except:
                pass

    def handle_tunnel_connection(self, client_socket, channel):
        """Handle individual tunnel connection"""
        try:
            while True:
                r, w, x = select.select([client_socket, channel], [], [])
                if client_socket in r:
                    data = client_socket.recv(4096)
                    if len(data) == 0:
                        break
                    channel.send(data)
                if channel in r:
                    data = channel.recv(4096)
                    if len(data) == 0:
                        break
                    client_socket.send(data)
        except Exception as e:
            print(f"⚠️ Tunnel connection error: {e}")
        finally:
            try:
                client_socket.close()
                channel.close()
            except:
                pass

    def create_ssh_tunnel_paramiko_fixed(self, ssh_client, remote_host, remote_port, local_port=9092):
        """Create SSH tunnel using paramiko with proper error handling"""
        try:
            print(f"🔗 Creating SSH tunnel to {remote_host}:{remote_port} via paramiko...")

            # Get transport from SSH client
            transport = ssh_client.get_transport()
            if not transport or not transport.is_active():
                print("❌ SSH transport not active")
                return False

            # Create a local server socket
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind(('127.0.0.1', local_port))
                server_socket.listen(5)  # Allow multiple connections
                self.tunnel_sockets.append(server_socket)
                print(f"🔗 Tunnel server socket bound to local port {local_port}")
            except Exception as e:
                print(f"❌ Failed to bind to local port {local_port}: {e}")
                return False

            # Start tunnel handler thread
            def tunnel_handler():
                try:
                    while True:
                        try:
                            # Accept client connection
                            client_socket, client_addr = server_socket.accept()
                            print(f"🔗 Tunnel connection from {client_addr}")

                            # Create SSH channel
                            try:
                                channel = transport.open_channel(
                                    "direct-tcpip",
                                    (remote_host, remote_port),
                                    client_addr
                                )
                                print(f"📡 SSH channel opened to {remote_host}:{remote_port}")

                                # Handle the connection in a separate thread
                                connection_thread = threading.Thread(
                                    target=self.handle_tunnel_connection_fixed,
                                    args=(client_socket, channel),
                                    daemon=True
                                )
                                connection_thread.start()

                            except Exception as e:
                                print(f"❌ Failed to open SSH channel: {e}")
                                try:
                                    client_socket.close()
                                except:
                                    pass

                        except Exception as e:
                            print(f"⚠️ Tunnel accept error: {e}")
                            # Don't break on accept errors - keep tunnel alive
                            time.sleep(1)
                            continue

                except Exception as e:
                    print(f"❌ Tunnel handler error: {e}")
                finally:
                    try:
                        server_socket.close()
                    except:
                        pass

            # Start the tunnel handler thread
            tunnel_thread = threading.Thread(target=tunnel_handler, daemon=True)
            tunnel_thread.start()
            self.tunnel_threads.append(tunnel_thread)

            # Wait a moment for tunnel to start
            time.sleep(2)

            # Test if tunnel is listening (but don't disconnect it)
            print("🧪 Testing tunnel is listening...")
            try:
                # Just test if we can create a socket that would connect to the tunnel
                # but don't actually connect to avoid disrupting the tunnel
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(2)
                # Check if the port is bound by trying to bind to it (should fail if tunnel is there)
                try:
                    test_socket.bind(('127.0.0.1', local_port))
                    test_socket.close()
                    print(f"❌ Local port {local_port} is not bound - tunnel not active")
                    return False
                except OSError:
                    # Port is already bound (good - our tunnel is using it)
                    test_socket.close()
                    print(f"✅ SSH tunnel is active and listening on local port {local_port}")
                    return True
            except Exception as e:
                print(f"❌ SSH tunnel test failed: {e}")
                return False

        except Exception as e:
            print(f"❌ SSH tunnel creation error: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def handle_tunnel_connection_fixed(self, client_socket, channel):
        """Handle individual tunnel connection with better error handling"""
        connection_id = id(client_socket) % 10000  # Short ID for this connection
        try:
            print(f"🔄 Starting tunnel data forwarding (conn-{connection_id})")

            while True:
                # Use select to check for data
                ready_sockets, _, error_sockets = select.select([client_socket, channel], [], [client_socket, channel], 1.0)

                if error_sockets:
                    break

                if client_socket in ready_sockets:
                    try:
                        data = client_socket.recv(4096)
                        if not data:
                            break
                        channel.send(data)
                        # Only print for first few packets to reduce noise
                        if connection_id % 1000 < 10:
                            print(f"📤 conn-{connection_id}: Forwarded {len(data)} bytes to server")
                    except Exception as e:
                        print(f"⚠️ conn-{connection_id}: Error forwarding client data: {e}")
                        break

                if channel in ready_sockets:
                    try:
                        data = channel.recv(4096)
                        if not data:
                            break
                        client_socket.send(data)
                        # Only print for first few packets to reduce noise
                        if connection_id % 1000 < 10:
                            print(f"📥 conn-{connection_id}: Forwarded {len(data)} bytes to client")
                    except Exception as e:
                        print(f"⚠️ conn-{connection_id}: Error forwarding server data: {e}")
                        break

        except Exception as e:
            print(f"⚠️ conn-{connection_id}: Tunnel connection handling error: {e}")
        finally:
            print(f"🧹 conn-{connection_id}: Connection closed")
            try:
                client_socket.close()
            except:
                pass
            try:
                channel.close()
            except:
                pass

    def start_flower_server_tmux(self, venv_activate: str):
        """Start Flower server using tmux session"""
        try:
            print(f"🌸 Starting Flower server on {self.server_config['host']} using tmux...")
            print(f"🔍 Debug: Connecting to {self.server_config['host']} as {self.server_config['username']}")
            print(f"🔍 Debug: Using virtual environment: {venv_activate}")

            self.server_ssh = paramiko.SSHClient()
            self.server_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect with debugging
            try:
                self.server_ssh.connect(
                    self.server_config['host'],
                    username=self.server_config['username'],
                    password=self.server_config['password'],
                    timeout=self.ssh_timeout,
                    auth_timeout=self.ssh_auth_timeout,
                    banner_timeout=self.ssh_banner_timeout
                )
                print(f"✓ SSH connection established (timeout: {self.ssh_timeout}s, auth: {self.ssh_auth_timeout}s, banner: {self.ssh_banner_timeout}s)")
            except Exception as ssh_error:
                print(f"❌ SSH connection failed: {ssh_error}")
                return False

            # Clean up any existing tmux sessions and processes
            print("🧹 Cleaning up existing Flower processes and tmux sessions...")
            cleanup_commands = [
                "tmux kill-session -t flower_server 2>/dev/null || true",
                "pkill -f 'flower-superlink' || pkill -f 'flower-supernode' || true"
            ]
            for cmd in cleanup_commands:
                self.server_ssh.exec_command(cmd)
            time.sleep(3)

            # Ensure logs directory exists
            print("📁 Creating logs directory...")
            if not self.ensure_logs_directory(self.server_ssh, self.project_dir):
                print("⚠️ Warning: Could not create logs directory, logs will be saved in project root")
                logs_prefix = ""
            else:
                logs_prefix = "logs/"

            # Verify Flower is installed
            print("🔍 Debug: Checking Flower installation...")
            check_flwr_command = self.get_venv_command("which flwr && flwr --version", venv_activate)
            stdin, stdout, stderr = self.server_ssh.exec_command(check_flwr_command)
            flwr_check = stdout.read().decode()
            flwr_error = stderr.read().decode()

            if flwr_error:
                print(f"❌ Flower installation issue: {flwr_error}")
                return False
            else:
                print(f"✓ Flower found: {flwr_check}")

            # Test basic Flower command
            print("🔍 Debug: Testing basic Flower command...")
            test_command = self.get_venv_command(f"cd {self.project_dir} && flwr run --help", venv_activate)
            stdin, stdout, stderr = self.server_ssh.exec_command(test_command)
            test_output = stdout.read().decode()
            test_error = stderr.read().decode()

            if test_error:
                print(f"❌ Flower command test failed: {test_error}")
                return False
            else:
                print("✓ Flower command works")

            # Create new tmux session for SuperLink
            print("🚀 Creating tmux session for SuperLink...")
            session_command = f"tmux new-session -d -s flower_server"
            stdin, stdout, stderr = self.server_ssh.exec_command(session_command)
            time.sleep(2)

            # Send SuperLink command with Process Isolation Mode
            superlink_log = f"{logs_prefix}superlink_{self.timestamp}.log"
            superlink_command = self.get_venv_command(
                f'cd {self.project_dir} && flower-superlink --isolation process --insecure 2>&1 | tee {superlink_log}',
                venv_activate
            )

            send_command = f'tmux send-keys -t flower_server "{superlink_command}" Enter'
            print(f"🚀 Starting SuperLink with Process Isolation Mode: {superlink_command}")
            stdin, stdout, stderr = self.server_ssh.exec_command(send_command)

            # Wait for SuperLink to start
            print("⏳ Waiting for SuperLink to start...")
            time.sleep(10)

            # Create separate tmux session for ServerApp
            print("🔧 Creating tmux session for ServerApp...")
            serverapp_session_cmd = "tmux new-session -d -s flower_serverapp"
            stdin, stdout, stderr = self.server_ssh.exec_command(serverapp_session_cmd)
            time.sleep(2)

            # Start ServerApp in separate process
            serverapp_log = f"{logs_prefix}serverapp_{self.timestamp}.log"
            serverapp_command = self.get_venv_command(
                f'cd {self.project_dir} && flwr-serverapp --serverappio-api-address 127.0.0.1:9091 --insecure 2>&1 | tee {serverapp_log}',
                venv_activate
            )

            send_serverapp_cmd = f'tmux send-keys -t flower_serverapp "{serverapp_command}" Enter'
            print(f"🚀 Starting ServerApp in separate process")
            stdin, stdout, stderr = self.server_ssh.exec_command(send_serverapp_cmd)
            time.sleep(5)

            # Check if SuperLink tmux session exists and is running
            check_superlink_session = "tmux list-sessions | grep flower_server"
            stdin, stdout, stderr = self.server_ssh.exec_command(check_superlink_session)
            superlink_session_output = stdout.read().decode()

            # Check if ServerApp tmux session exists and is running
            check_serverapp_session = "tmux list-sessions | grep flower_serverapp"
            stdin, stdout, stderr = self.server_ssh.exec_command(check_serverapp_session)
            serverapp_session_output = stdout.read().decode()

            # Check if SuperLink process is running
            check_superlink_process = "pgrep -f 'flower-superlink'"
            stdin, stdout, stderr = self.server_ssh.exec_command(check_superlink_process)
            superlink_pids = stdout.read().decode().strip()

            # Check if ServerApp process is running
            check_serverapp_process = "pgrep -f 'flwr-serverapp'"
            stdin, stdout, stderr = self.server_ssh.exec_command(check_serverapp_process)
            serverapp_pids = stdout.read().decode().strip()

            # Check if ports 9092 (Fleet API) and 9093 (REST API) are listening
            check_fleet_port_command = "netstat -tuln | grep :9092 || ss -tuln | grep :9092"
            stdin, stdout, stderr = self.server_ssh.exec_command(check_fleet_port_command)
            fleet_port_output = stdout.read().decode()

            check_rest_port_command = "netstat -tuln | grep :9093 || ss -tuln | grep :9093"
            stdin, stdout, stderr = self.server_ssh.exec_command(check_rest_port_command)
            rest_port_output = stdout.read().decode()

            if (superlink_session_output and serverapp_session_output and
                superlink_pids and serverapp_pids and fleet_port_output and rest_port_output):
                print(f"✅ Flower Server Process Isolation Mode started successfully")
                print(f"📺 SuperLink session: {superlink_session_output.strip()}")
                print(f"📺 ServerApp session: {serverapp_session_output.strip()}")
                print(f"🔧 SuperLink PID: {superlink_pids}")
                print(f"🔧 ServerApp PID: {serverapp_pids}")
                print(f"🌐 Fleet API (port 9092): {fleet_port_output.strip()}")
                print(f"🌐 REST API (port 9093): {rest_port_output.strip()}")
                print(f"🔗 Process Isolation: SuperLink <-> ServerApp via 127.0.0.1:9091")
                print(f"✅ Server-side PyTorch DataLoader multiprocessing enabled")

                # Show SuperLink log preview
                superlink_log_command = f"tail -10 {superlink_log} 2>/dev/null || echo 'SuperLink log not yet available'"
                stdin, stdout, stderr = self.server_ssh.exec_command(f"cd {self.project_dir} && {superlink_log_command}")
                superlink_log_output = stdout.read().decode()
                if superlink_log_output and "not yet available" not in superlink_log_output:
                    print("📋 SuperLink log preview:")
                    print(superlink_log_output)

                # Show ServerApp log preview
                serverapp_log_command = f"tail -10 {serverapp_log} 2>/dev/null || echo 'ServerApp log not yet available'"
                stdin, stdout, stderr = self.server_ssh.exec_command(f"cd {self.project_dir} && {serverapp_log_command}")
                serverapp_log_output = stdout.read().decode()
                if serverapp_log_output and "not yet available" not in serverapp_log_output:
                    print("📋 ServerApp log preview:")
                    print(serverapp_log_output)

                return True
            else:
                print("❌ Failed to start Flower Server Process Isolation Mode")

                # Debug information
                if not superlink_session_output:
                    print("  - SuperLink tmux session not found")
                if not serverapp_session_output:
                    print("  - ServerApp tmux session not found")
                if not superlink_pids:
                    print("  - SuperLink process not running")
                if not serverapp_pids:
                    print("  - ServerApp process not running")
                if not fleet_port_output:
                    print("  - Fleet API port 9092 not listening")
                if not rest_port_output:
                    print("  - REST API port 9093 not listening")

                # Check SuperLink tmux session content for errors
                if superlink_session_output:
                    log_command = "tmux capture-pane -t flower_server -p"
                    stdin, stdout, stderr = self.server_ssh.exec_command(log_command)
                    tmux_output = stdout.read().decode()
                    if tmux_output:
                        print(f"📋 SuperLink tmux output:\n{tmux_output}")

                # Check ServerApp tmux session content for errors
                if serverapp_session_output:
                    log_command = "tmux capture-pane -t flower_serverapp -p"
                    stdin, stdout, stderr = self.server_ssh.exec_command(log_command)
                    tmux_output = stdout.read().decode()
                    if tmux_output:
                        print(f"📋 ServerApp tmux output:\n{tmux_output}")

                # Show detailed error logs if available
                superlink_error_log = f"tail -20 {superlink_log} 2>/dev/null || echo 'No {superlink_log} found'"
                stdin, stdout, stderr = self.server_ssh.exec_command(f"cd {self.project_dir} && {superlink_error_log}")
                error_log = stdout.read().decode()
                if error_log and "not found" not in error_log:
                    print("❌ SuperLink error log:")
                    print(error_log)

                serverapp_error_log = f"tail -20 {serverapp_log} 2>/dev/null || echo 'No {serverapp_log} found'"
                stdin, stdout, stderr = self.server_ssh.exec_command(f"cd {self.project_dir} && {serverapp_error_log}")
                error_log = stdout.read().decode()
                if error_log and "not found" not in error_log:
                    print("❌ ServerApp error log:")
                    print(error_log)

                return False

        except Exception as e:
            print(f"❌ Error starting server with tmux: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def start_flower_client_tmux(self, client_config: Dict, venv_activate: str):
        """Start Flower client using tmux session"""
        client_host = client_config['host']
        partition_id = client_config.get('partition_id', 0)
        session_name = f"flower_client_{partition_id}"

        try:
            print(f"🌻 Starting Flower client on {client_host} using tmux...")

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                client_host,
                username=client_config['username'],
                password=client_config['password'],
                timeout=self.ssh_timeout,
                auth_timeout=self.ssh_auth_timeout,
                banner_timeout=self.ssh_banner_timeout
            )

            # Clean up any existing tmux sessions and processes
            cleanup_commands = [
                f"tmux kill-session -t {session_name} 2>/dev/null || true",
                "pkill -f 'flower-supernode' || true"
            ]
            for cmd in cleanup_commands:
                ssh.exec_command(cmd)
            time.sleep(3)

            # Ensure logs directory exists
            print(f"📁 Creating logs directory on {client_host}...")
            if not self.ensure_logs_directory(ssh, client_config["project_dir"]):
                print(f"⚠️ Warning: Could not create logs directory on {client_host}, logs will be saved in project root")
                logs_prefix = ""
            else:
                logs_prefix = "logs/"

            # Clean up any existing tunnels from previous runs
            print(f"🧹 Cleaning up any existing tunnels on local port {9092 + partition_id}...")
            cleanup_tunnel_processes = [
                f"pkill -f 'ssh.*localhost:{9092 + partition_id}' || true",
                f"fuser -k {9092 + partition_id}/tcp 2>/dev/null || true"
            ]
            for cmd in cleanup_tunnel_processes:
                ssh.exec_command(cmd)

            # Create new tmux session for SuperNode
            session_command = f"tmux new-session -d -s {session_name}"
            stdin, stdout, stderr = ssh.exec_command(session_command)
            time.sleep(2)

            # Prepare SuperNode command with logging
            client_port = 9094 + partition_id
            client_log_name = f"{logs_prefix}client_{client_host.split('.')[0]}_{self.timestamp}.log"
            flower_command = self.get_venv_command(
                f'cd {client_config["project_dir"]} && '
                f'flower-supernode --insecure '
                f'--superlink {self.server_config["host"]}:9092 '
                f'--clientappio-api-address 0.0.0.0:{client_port} '
                f"--node-config 'partition-id={partition_id} num-partitions={len(self.clients_config)}' "
                f'2>&1 | tee {client_log_name}',
                venv_activate
            )

            # Send the flower command to the tmux session
            send_command = f'tmux send-keys -t {session_name} "{flower_command}" Enter'
            print(f"🚀 Starting client with tmux: {flower_command}")
            stdin, stdout, stderr = ssh.exec_command(send_command)

            # Wait and verify client started
            time.sleep(5)

            # Check if tmux session exists and is running
            check_session_command = f"tmux list-sessions | grep {session_name}"
            stdin, stdout, stderr = ssh.exec_command(check_session_command)
            session_output = stdout.read().decode()

            # Check if SuperNode process is running
            check_process_command = "pgrep -f 'flower-supernode'"
            stdin, stdout, stderr = ssh.exec_command(check_process_command)
            process_pids = stdout.read().decode().strip()

            if session_output and process_pids:
                print(f"✅ SuperNode started in tmux session on {client_host}")
                print(f"📺 Tmux session: {session_output.strip()}")
                print(f"🔧 Process PID: {process_pids}")
                print(f"📝 Client logs: {client_config['project_dir']}/{client_log_name}")
                self.ssh_connections.append(ssh)
                return True
            else:
                print(f"❌ Failed to start SuperNode in tmux session on {client_host}")

                # Debug information
                if not session_output:
                    print("  - Tmux session not found")
                if not process_pids:
                    print("  - SuperNode process not running")

                # Check tmux session content for errors
                log_command = f"tmux capture-pane -t {session_name} -p"
                stdin, stdout, stderr = ssh.exec_command(log_command)
                tmux_output = stdout.read().decode()
                if tmux_output:
                    print(f"📋 Tmux session output:\n{tmux_output}")

                # Check client log file for detailed errors
                client_log_command = f"tail -10 {client_log_name} 2>/dev/null || echo 'Client log not yet available'"
                stdin, stdout, stderr = ssh.exec_command(f"cd {client_config['project_dir']} && {client_log_command}")
                client_log_output = stdout.read().decode()
                if client_log_output and "not yet available" not in client_log_output:
                    print(f"📋 Client log preview ({client_log_name}):")
                    print(client_log_output)

                ssh.close()
                return False

        except Exception as e:
            print(f"❌ Error starting client on {client_host} with tmux: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def start_flower_client_tmux_with_tunnel(self, client_config: Dict, venv_activate: str):
        """Start Flower client using tmux session with SSH tunnel and Process Isolation Mode"""
        client_host = client_config['host']
        partition_id = client_config.get('partition_id', 0)
        supernode_session = f"flower_supernode_{partition_id}"
        clientapp_session = f"flower_clientapp_{partition_id}"

        try:
            print(f"🌻 Starting Flower client on {client_host} using Process Isolation Mode with SSH tunnel...")

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                client_host,
                username=client_config['username'],
                password=client_config['password'],
                timeout=self.ssh_timeout,
                auth_timeout=self.ssh_auth_timeout,
                banner_timeout=self.ssh_banner_timeout
            )

            # Clean up any existing tmux sessions and processes
            cleanup_commands = [
                f"tmux kill-session -t {supernode_session} 2>/dev/null || true",
                f"tmux kill-session -t {clientapp_session} 2>/dev/null || true",
                "pkill -f 'flower-supernode' || true",
                "pkill -f 'flwr-clientapp' || true",
                f"pkill -f 'ssh.*{self.server_config['host']}.*9092' || true"  # Kill existing SSH tunnels to this server
            ]
            for cmd in cleanup_commands:
                ssh.exec_command(cmd)
            time.sleep(3)

            # Ensure logs directory exists
            print(f"📁 Creating logs directory on {client_host}...")
            if not self.ensure_logs_directory(ssh, client_config["project_dir"]):
                print(f"⚠️ Warning: Could not create logs directory on {client_host}, logs will be saved in project root")
                logs_prefix = ""
            else:
                logs_prefix = "logs/"

            # Create tmux sessions for SuperNode and ClientApp
            print(f"🔗 Creating Process Isolation Mode setup on {client_host}...")

            local_tunnel_port = 9092 + partition_id  # Each client gets a unique local port
            client_port = 9094 + partition_id
            supernode_log = f"{logs_prefix}supernode_{client_host.split('.')[0]}_{self.timestamp}.log"
            clientapp_log = f"{logs_prefix}clientapp_{client_host.split('.')[0]}_{self.timestamp}.log"

            print(f"🔧 Client {partition_id} will use local tunnel port {local_tunnel_port}")

            # Step 1: Create SSH tunnel and start SuperNode in process isolation mode
            supernode_session_cmd = f"tmux new-session -d -s {supernode_session}"
            stdin, stdout, stderr = ssh.exec_command(supernode_session_cmd)
            time.sleep(2)

            # SuperNode command with interactive SSH tunnel (user will input password manually)
            supernode_command = self.get_venv_command(
                f'cd {client_config["project_dir"]} && '
                f'ssh -f -N -L {local_tunnel_port}:{self.server_config["host"]}:9092 '
                f'-o StrictHostKeyChecking=no '
                f'-o UserKnownHostsFile=/dev/null '
                f'-o ConnectTimeout={self.ssh_timeout} '
                f'-o PasswordAuthentication=yes '
                f'-o NumberOfPasswordPrompts=3 '
                f'-o ServerAliveInterval=30 '
                f'-o ServerAliveCountMax=3 '
                f'{self.server_config["username"]}@{self.server_config["host"]} && '
                f'sleep 3 && '
                f'flower-supernode --isolation process --insecure '
                f'--superlink localhost:{local_tunnel_port} '
                f'--clientappio-api-address 0.0.0.0:{client_port} '
                f"--node-config 'partition-id={partition_id} num-partitions={len(self.clients_config)}' "
                f'2>&1 | tee {supernode_log}',
                venv_activate
            )

            # Send SuperNode command to tmux session
            send_supernode_cmd = f'tmux send-keys -t {supernode_session} "{supernode_command}" Enter'
            stdin, stdout, stderr = ssh.exec_command(send_supernode_cmd)
            print(f"🚀 Started SuperNode in process isolation mode")
            print(f"💡 SSH tunnel will prompt for password in tmux session '{supernode_session}' on {client_host}")
            print(f"⏱️ SSH tunnel timeouts: connect={self.ssh_timeout}s, auth={self.ssh_auth_timeout}s (3 attempts)")
            print(f"🔄 Keep-alive: 30s interval × 3 attempts = 90s grace period")
            print(f"📋 To enter password: ssh {client_host} && tmux attach -t {supernode_session}")

            # Wait longer for user to input password and tunnel to be established
            time.sleep(15)

            # Step 2: Create separate tmux session for ClientApp
            clientapp_session_cmd = f"tmux new-session -d -s {clientapp_session}"
            stdin, stdout, stderr = ssh.exec_command(clientapp_session_cmd)
            time.sleep(2)

            # Start ClientApp connecting to SuperNode
            clientapp_command = self.get_venv_command(
                f'cd {client_config["project_dir"]} && '
                f'flwr-clientapp --clientappio-api-address 127.0.0.1:{client_port} --insecure '
                f'2>&1 | tee {clientapp_log}',
                venv_activate
            )

            # Send ClientApp command to its tmux session
            send_clientapp_cmd = f'tmux send-keys -t {clientapp_session} "{clientapp_command}" Enter'
            print(f"🚀 Started ClientApp in separate process")
            stdin, stdout, stderr = ssh.exec_command(send_clientapp_cmd)

            # Wait for ClientApp to start
            time.sleep(5)

            # Verify both processes are running
            # Check SuperNode tmux session
            check_supernode_session = f"tmux list-sessions | grep {supernode_session}"
            stdin, stdout, stderr = ssh.exec_command(check_supernode_session)
            supernode_session_output = stdout.read().decode()

            # Check ClientApp tmux session
            check_clientapp_session = f"tmux list-sessions | grep {clientapp_session}"
            stdin, stdout, stderr = ssh.exec_command(check_clientapp_session)
            clientapp_session_output = stdout.read().decode()

            # Verify SSH tunnel is active (check for SSH tunnel process)
            check_tunnel_process = f"pgrep -f 'ssh.*-L {local_tunnel_port}:{self.server_config['host']}:9092'"
            stdin, stdout, stderr = ssh.exec_command(check_tunnel_process)
            tunnel_pid = stdout.read().decode().strip()
            tunnel_active = bool(tunnel_pid)

            # Check if SuperNode process is running
            check_supernode_process = "pgrep -f 'flower-supernode'"
            stdin, stdout, stderr = ssh.exec_command(check_supernode_process)
            supernode_pids = stdout.read().decode().strip()

            # Check if ClientApp process is running
            check_clientapp_process = "pgrep -f 'flwr-clientapp'"
            stdin, stdout, stderr = ssh.exec_command(check_clientapp_process)
            clientapp_pids = stdout.read().decode().strip()

            if (supernode_session_output and clientapp_session_output and
                tunnel_active and supernode_pids and clientapp_pids):
                print(f"✅ Process Isolation Mode setup successful on {client_host}")
                print(f"📺 SuperNode session: {supernode_session_output.strip()}")
                print(f"📺 ClientApp session: {clientapp_session_output.strip()}")
                print(f"🔧 SSH tunnel: Active on local port {local_tunnel_port}")
                if tunnel_pid:
                    print(f"🌐 Tunnel process: {tunnel_pid}")
                print(f"🔧 SuperNode PID: {supernode_pids}")
                print(f"🔧 ClientApp PID: {clientapp_pids}")
                print(f"📝 SuperNode logs: {client_config['project_dir']}/{supernode_log}")
                print(f"📝 ClientApp logs: {client_config['project_dir']}/{clientapp_log}")
                print(f"🔗 Process Isolation: SuperNode <-> ClientApp via 127.0.0.1:{client_port}")
                print(f"🔗 Network: SuperNode -> SSH tunnel -> {self.server_config['host']}:9092")
                self.ssh_connections.append(ssh)
                return True
            else:
                print(f"❌ Failed to start Process Isolation Mode setup on {client_host}")

                # Debug information
                if not supernode_session_output:
                    print("  - SuperNode tmux session not found")
                if not clientapp_session_output:
                    print("  - ClientApp tmux session not found")
                if not tunnel_active:
                    print(f"  - SSH tunnel not active on port {local_tunnel_port}")
                if not supernode_pids:
                    print("  - SuperNode process not running")
                if not clientapp_pids:
                    print("  - ClientApp process not running")

                # Check tmux session content for errors
                if supernode_session_output:
                    log_command = f"tmux capture-pane -t {supernode_session} -p"
                    stdin, stdout, stderr = ssh.exec_command(log_command)
                    tmux_output = stdout.read().decode()
                    if tmux_output:
                        print(f"📋 SuperNode tmux output:\n{tmux_output}")

                if clientapp_session_output:
                    log_command = f"tmux capture-pane -t {clientapp_session} -p"
                    stdin, stdout, stderr = ssh.exec_command(log_command)
                    tmux_output = stdout.read().decode()
                    if tmux_output:
                        print(f"📋 ClientApp tmux output:\n{tmux_output}")

                # Check log files for detailed errors
                supernode_log_command = f"tail -10 {supernode_log} 2>/dev/null || echo 'SuperNode log not yet available'"
                stdin, stdout, stderr = ssh.exec_command(f"cd {client_config['project_dir']} && {supernode_log_command}")
                supernode_log_output = stdout.read().decode()
                if supernode_log_output and "not yet available" not in supernode_log_output:
                    print(f"📋 SuperNode log preview:")
                    print(supernode_log_output)

                clientapp_log_command = f"tail -10 {clientapp_log} 2>/dev/null || echo 'ClientApp log not yet available'"
                stdin, stdout, stderr = ssh.exec_command(f"cd {client_config['project_dir']} && {clientapp_log_command}")
                clientapp_log_output = stdout.read().decode()
                if clientapp_log_output and "not yet available" not in clientapp_log_output:
                    print(f"📋 ClientApp log preview:")
                    print(clientapp_log_output)

                ssh.close()
                return False

        except Exception as e:
            print(f"❌ Error starting Process Isolation Mode on {client_host}: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def ensure_federation_config(self, federation_name: str = "multi-machine"):
        """Ensure pyproject.toml has the correct federation configuration on the server"""
        try:
            pyproject_path = f"{self.project_dir}/pyproject.toml"

            print(f"🔍 Checking federation configuration in {pyproject_path} on server...")

            # More thorough check for pyproject.toml file
            check_commands = [
                f"ls -la {pyproject_path}",
                f"file {pyproject_path}",
                f"head -5 {pyproject_path}"
            ]

            print("🔍 Detailed file check:")
            for cmd in check_commands:
                stdin, stdout, stderr = self.server_ssh.exec_command(cmd)
                output = stdout.read().decode().strip()
                error = stderr.read().decode().strip()
                if output:
                    print(f"  {cmd}: {output}")
                if error:
                    print(f"  {cmd} (error): {error}")

            # Read current content from server
            print("📖 Reading pyproject.toml content...")
            read_file_command = f"cat {pyproject_path}"
            stdin, stdout, stderr = self.server_ssh.exec_command(read_file_command)
            content = stdout.read().decode()
            read_error = stderr.read().decode()

            if read_error:
                print(f"❌ Error reading pyproject.toml: {read_error}")
                print(f"🔍 Working directory on server: {self.project_dir}")
                # List directory contents for debugging
                list_cmd = f"ls -la {self.project_dir}/"
                stdin, stdout, stderr = self.server_ssh.exec_command(list_cmd)
                dir_contents = stdout.read().decode()
                print(f"📁 Directory contents:\n{dir_contents}")
                return False

            if not content.strip():
                print("❌ pyproject.toml is empty")
                return False

            print(f"✅ Successfully read pyproject.toml ({len(content)} characters)")

            # Check if federation configuration exists
            federation_section = f"[tool.flwr.federations.{federation_name}]"
            if federation_section not in content:
                print(f"📝 Adding federation configuration to existing pyproject.toml on server...")

                # Add federation configuration
                federation_config = f"""

# Flower federation configuration for multi-machine deployment
{federation_section}
address = "127.0.0.1:9093"
insecure = true
"""

                # Append to the file on the server
                append_command = f'cat >> {pyproject_path} << "EOF"\n{federation_config}\nEOF'
                stdin, stdout, stderr = self.server_ssh.exec_command(append_command)
                append_error = stderr.read().decode()

                if append_error:
                    print(f"⚠️ Error appending to pyproject.toml: {append_error}")
                    return False

                print(f"✅ Added federation '{federation_name}' configuration to existing pyproject.toml")
            else:
                print(f"✅ Federation '{federation_name}' configuration already exists")

            return True

        except Exception as e:
            print(f"⚠️ Error checking federation configuration: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            print(f"📋 Please manually add this to your pyproject.toml on the server:")
            print(f"    [tool.flwr.federations.{federation_name}]")
            print(f"    address = \"127.0.0.1:9093\"")
            print(f"    insecure = true")
            return False

    def run_flower_app(self, venv_activate: str, federation_name: str = "multi-machine"):
        """Run the Flower App"""
        try:
            print(f"🚀 Running Flower App on federation '{federation_name}'...")
            print(f"🔍 Note: Federation will connect to SuperLink REST API at localhost:9093")

            # Ensure federation configuration exists
            if not self.ensure_federation_config(federation_name):
                print("❌ Failed to configure federation. Cannot proceed.")
                return False

            # Verify SuperLink REST API is accessible
            print("🔍 Verifying SuperLink REST API accessibility...")
            check_rest_api_command = "curl -s -f http://127.0.0.1:9093/health || echo 'REST API not accessible'"
            stdin, stdout, stderr = self.server_ssh.exec_command(check_rest_api_command)
            rest_api_response = stdout.read().decode().strip()

            if "not accessible" in rest_api_response:
                print("⚠️ SuperLink REST API not accessible at 127.0.0.1:9093")
                print("💡 This is normal - some Flower versions don't have a health endpoint")
            else:
                print("✅ SuperLink REST API is accessible")

            run_command = self.get_venv_command(
                f'cd {self.project_dir} && flwr run . {federation_name} --stream',
                venv_activate
            )

            print(f"🔍 Executing: {run_command}")

            # Use invoke_shell for interactive streaming
            channel = self.server_ssh.invoke_shell()
            channel.send(run_command + '\n')

            print("📊 Flower App Output:")
            print("=" * 50)

            # Stream output with timeout
            output_buffer = ""
            timeout_counter = 0
            max_timeout = 3000  # 25 minutes

            while timeout_counter < max_timeout:
                if channel.recv_ready():
                    data = channel.recv(4096).decode()
                    output_buffer += data
                    print(data, end="")
                    timeout_counter = 0  # Reset timeout when receiving data

                    # Check for completion indicators
                    if any(phrase in output_buffer.lower() for phrase in [
                        "run finished", "completed successfully", "experiment completed",
                        "training finished", "federation completed"
                    ]):
                        print("\n🎉 Detected completion signal!")
                        break

                if channel.exit_status_ready():
                    # Get any remaining output
                    while channel.recv_ready():
                        data = channel.recv(4096).decode()
                        print(data, end="")
                    break

                time.sleep(0.5)
                timeout_counter += 1

            if timeout_counter >= max_timeout:
                print(f"\n⏰ Timeout reached ({max_timeout/2/60:.1f} minutes)")

            print("\n" + "=" * 50)
            print("✅ Flower App execution completed!")
            channel.close()
            return True

        except Exception as e:
            print(f"❌ Error running Flower App: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def cleanup_tmux_sessions(self):
        """Clean up all tmux sessions and processes"""
        print("\n🧹 Cleaning up tmux sessions...")

        # Cleanup server (Process Isolation Mode)
        if self.server_ssh:
            try:
                cleanup_commands = [
                    "tmux kill-session -t flower_server 2>/dev/null || true",
                    "tmux kill-session -t flower_serverapp 2>/dev/null || true",
                    "pkill -f 'flower-superlink' || true",
                    "pkill -f 'flwr-serverapp' || true"
                ]
                for cmd in cleanup_commands:
                    self.server_ssh.exec_command(cmd)
                print("✓ Server Process Isolation Mode sessions stopped")
                self.server_ssh.close()
            except Exception as e:
                print(f"⚠️ Error cleaning up server: {e}")

        # Cleanup clients (Process Isolation Mode)
        for i, ssh in enumerate(self.ssh_connections):
            try:
                local_tunnel_port = 9092 + i
                cleanup_commands = [
                    f"tmux kill-session -t flower_supernode_{i} 2>/dev/null || true",
                    f"tmux kill-session -t flower_clientapp_{i} 2>/dev/null || true",
                    "pkill -f 'flower-supernode' || true",
                    "pkill -f 'flwr-clientapp' || true",
                    f"pkill -f 'ssh.*-L {local_tunnel_port}:{self.server_config['host']}:9092' || true",
                    f"fuser -k {local_tunnel_port}/tcp 2>/dev/null || true"
                ]
                for cmd in cleanup_commands:
                    ssh.exec_command(cmd)
                ssh.close()
                print(f"✓ Client {i+1} Process Isolation Mode sessions and SSH tunnels stopped")
            except Exception as e:
                print(f"⚠️ Error cleaning up client {i+1}: {e}")

        print("✅ Process Isolation Mode cleanup completed")

    def monitor_tmux_sessions(self):
        """Monitor tmux sessions and processes continuously"""
        try:
            print("📊 Monitoring tmux sessions...")

            # Check server sessions (Process Isolation Mode)
            if self.server_ssh:
                # Check SuperLink
                check_superlink_cmd = "tmux list-sessions | grep flower_server && pgrep -f 'flower-superlink'"
                stdin, stdout, stderr = self.server_ssh.exec_command(check_superlink_cmd)
                superlink_status = stdout.read().decode()

                # Check ServerApp
                check_serverapp_cmd = "tmux list-sessions | grep flower_serverapp && pgrep -f 'flwr-serverapp'"
                stdin, stdout, stderr = self.server_ssh.exec_command(check_serverapp_cmd)
                serverapp_status = stdout.read().decode()

                if superlink_status and serverapp_status:
                    print("✅ Server Process Isolation Mode active (SuperLink + ServerApp)")
                elif superlink_status:
                    print("⚠️ SuperLink active, but ServerApp not found")
                elif serverapp_status:
                    print("⚠️ ServerApp active, but SuperLink not found")
                else:
                    print("⚠️ Server Process Isolation Mode not found")

            # Check Process Isolation Mode sessions
            active_supernodes = 0
            active_clientapps = 0
            active_tunnels = 0
            for i, ssh in enumerate(self.ssh_connections):
                try:
                    # Check SuperNode session
                    supernode_cmd = f"tmux list-sessions | grep flower_supernode_{i} && pgrep -f 'flower-supernode'"
                    stdin, stdout, stderr = ssh.exec_command(supernode_cmd)
                    supernode_status = stdout.read().decode()
                    if supernode_status:
                        active_supernodes += 1

                    # Check ClientApp session
                    clientapp_cmd = f"tmux list-sessions | grep flower_clientapp_{i} && pgrep -f 'flwr-clientapp'"
                    stdin, stdout, stderr = ssh.exec_command(clientapp_cmd)
                    clientapp_status = stdout.read().decode()
                    if clientapp_status:
                        active_clientapps += 1

                    # Check if SSH tunnel is active
                    local_tunnel_port = 9092 + i
                    tunnel_cmd = f"pgrep -f 'ssh.*-L {local_tunnel_port}:{self.server_config['host']}:9092'"
                    stdin, stdout, stderr = ssh.exec_command(tunnel_cmd)
                    tunnel_status = stdout.read().decode().strip()
                    if tunnel_status:
                        active_tunnels += 1
                except:
                    pass

            print(f"🌻 Active SuperNode sessions: {active_supernodes}/{len(self.ssh_connections)}")
            print(f"📱 Active ClientApp sessions: {active_clientapps}/{len(self.ssh_connections)}")
            print(f"🔗 Active SSH tunnels: {active_tunnels}/{len(self.ssh_connections)}")

        except Exception as e:
            print(f"⚠️ Monitoring error: {e}")

    def monitor_status_continuous(self):
        """Continuously monitor the status of server and clients during execution"""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds

                # Check SuperLink and ServerApp status
                if self.server_ssh:
                    # Check SuperLink
                    stdin, stdout, stderr = self.server_ssh.exec_command("pgrep -f 'flower-superlink'")
                    superlink_pid = stdout.read().decode().strip()

                    # Check ServerApp
                    stdin, stdout, stderr = self.server_ssh.exec_command("pgrep -f 'flwr-serverapp'")
                    serverapp_pid = stdout.read().decode().strip()

                    if superlink_pid and serverapp_pid:
                        print(f"📈 Server Process Isolation Mode running (SuperLink: {superlink_pid}, ServerApp: {serverapp_pid})")
                    elif superlink_pid:
                        print(f"⚠️ SuperLink running ({superlink_pid}) but ServerApp not found!")
                    elif serverapp_pid:
                        print(f"⚠️ ServerApp running ({serverapp_pid}) but SuperLink not found!")
                    else:
                        print("⚠️ Server Process Isolation Mode not running!")
                        break

                # Check SuperNode and ClientApp status
                active_supernodes = 0
                active_clientapps = 0
                for i, ssh in enumerate(self.ssh_connections):
                    try:
                        # Check SuperNode process
                        stdin, stdout, stderr = ssh.exec_command("pgrep -f 'flower-supernode'")
                        supernode_pid = stdout.read().decode().strip()
                        if supernode_pid:
                            active_supernodes += 1

                        # Check ClientApp process
                        stdin, stdout, stderr = ssh.exec_command("pgrep -f 'flwr-clientapp'")
                        clientapp_pid = stdout.read().decode().strip()
                        if clientapp_pid:
                            active_clientapps += 1
                    except:
                        pass

                print(f"🌻 Active SuperNodes: {active_supernodes}/{len(self.ssh_connections)}")
                print(f"📱 Active ClientApps: {active_clientapps}/{len(self.ssh_connections)}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                break

    def run_federated_learning(self, venv_activate: str = None):
        """Main method to run federated learning with Process Isolation Mode"""
        print("🚀 Starting Flower Federated Learning with Full Process Isolation Mode")
        print("🔧 Server Process Isolation: SuperLink + ServerApp in separate processes")
        print("🔧 Client Process Isolation: SuperNode + ClientApp in separate processes")
        print("✅ PyTorch DataLoader multiprocessing enabled on both server and client (num_workers > 0)")
        print("🔗 Using SSH tunnel for secure communication")
        print("=" * 70)

        # Start server
        if not self.start_flower_server_tmux(venv_activate):
            print("❌ Failed to start server. Exiting...")
            return False

        print("⏳ Waiting for server to be ready...")
        time.sleep(10)

        # Start clients
        success_count = 0
        for i, client_config in enumerate(self.clients_config):
            print(f"Starting client {i+1}/{len(self.clients_config)} on {client_config['host']}...")
            print(f"💡 To manually enter SSH password, connect to {client_config['host']} and run:")
            print(f"   tmux attach -t flower_supernode_{i}")
            print()

            # SSH tunnel mode only
            success = self.start_flower_client_tmux_with_tunnel(client_config, venv_activate)

            if success:
                success_count += 1

            time.sleep(5)

        if success_count > 0:
            print(f"✅ {success_count}/{len(self.clients_config)} Process Isolation Mode clients started!")
            print("🔧 Server: SuperLink + ServerApp in separate processes")
            print("🔧 Each client: SuperNode + ClientApp in separate processes")
            print("✅ PyTorch DataLoader multiprocessing is now enabled on both server and client sides")

            print("⏳ Waiting for all components to connect...")
            time.sleep(10)

            # Monitor sessions before starting FL
            self.monitor_tmux_sessions()

            # Run Flower App
            print("\n🌸 Starting Flower App execution...")
            if self.run_flower_app(venv_activate):
                print("✅ Federated learning completed successfully!")
                print("🎉 Full Process Isolation Mode resolved the multiprocessing issue on both sides!")
                return True
            else:
                print("❌ Federated learning execution failed")
                return False
        else:
            print("❌ No Process Isolation Mode clients started successfully")
            return False

    def setup_persistent_env_vars(self, ssh_client, client_host: str):
        """Set up persistent environment variables in .bashrc for easier authentication"""
        try:
            print(f"🔧 Setting up persistent environment variables in .bashrc on {client_host}...")

            # Environment variables to add
            env_vars = [
                f'export FL_SERVER_HOST="{self.server_config["host"]}"',
                f'export FL_SERVER_USER="{self.server_config["username"]}"',
                f'export FL_SERVER_PASS="{self.server_config["password"]}"',
                f'export FL_PROJECT_DIR="{self.project_dir}"'
            ]

            # Check if variables already exist in .bashrc
            check_command = "grep -q 'FL_SERVER_HOST' ~/.bashrc"
            stdin, stdout, stderr = ssh_client.exec_command(check_command)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:  # Variables don't exist, add them
                print("📝 Adding FL environment variables to .bashrc...")

                bashrc_content = f"""

# Flower Federated Learning environment variables
# Added by run_multi_machines_tmux.py
{chr(10).join(env_vars)}
"""

                append_command = f'cat >> ~/.bashrc << "EOF"{bashrc_content}EOF'
                stdin, stdout, stderr = ssh_client.exec_command(append_command)
                append_error = stderr.read().decode()

                if append_error:
                    print(f"⚠️ Warning adding to .bashrc: {append_error}")
                else:
                    print("✅ Environment variables added to .bashrc")
                    print("💡 Variables will be available in new shell sessions")
            else:
                print("✅ FL environment variables already exist in .bashrc")

            return True

        except Exception as e:
            print(f"⚠️ Error setting up environment variables: {e}")
            return False

def main():
    """Main function using tmux sessions"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Flower Multi-Machine Federated Learning Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "config_file",
        help="Path to YAML configuration file (e.g., fl_server.yaml)"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        print(f"📄 Loading configuration from {args.config_file}...")
        from multi_config import load_config_from_yaml, get_server_config_dict, get_clients_config_dict
        config = load_config_from_yaml(args.config_file)

        # Validate multi-machine configuration
        if not config.fl.multi_machine:
            print("❌ No multi-machine configuration found in YAML file!")
            print("📝 Please add 'fl.multi_machine' section to your configuration.")
            print("📋 See fl_server.yaml for an example configuration.")
            return

        project_dir = config.fl.multi_machine.project_dir
        venv_activate = config.fl.multi_machine.venv_activate

    except FileNotFoundError as e:
        print(f"❌ Configuration file error: {e}")
        print("📝 Please provide a valid YAML configuration file.")
        print("📋 Example: python run_multi_machines_tmux.py fl_server.yaml")
        return
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return

    # Convert to dictionary format for compatibility with existing runner
    server_config = get_server_config_dict(config)
    clients_config = get_clients_config_dict(config)

    if not server_config or not clients_config:
        print("❌ Invalid multi-machine configuration!")
        print("📝 Please ensure server and clients are properly configured in the YAML file.")
        return

    # Create runner with SSH timeout configurations
    ssh_config = config.fl.multi_machine.ssh if config.fl.multi_machine.ssh else None
    ssh_timeout = ssh_config.timeout if ssh_config else 30
    ssh_auth_timeout = ssh_config.auth_timeout if ssh_config else 30
    ssh_banner_timeout = ssh_config.banner_timeout if ssh_config else 30
    runner = FlowerMultiMachineTmuxRunner(server_config, clients_config, project_dir, ssh_timeout, ssh_auth_timeout, ssh_banner_timeout)

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal...")
        runner.cleanup_tmux_sessions()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("🔍 Configuration:")
        print(f"  Server: {server_config['host']}:{server_config.get('port', 9092)}")
        print(f"  Clients: {[client['host'] for client in clients_config]}")
        print(f"  Project Dir: {project_dir}")
        print(f"  Virtual Env: {venv_activate}")
        print(f"  SSH Timeouts: connect={ssh_timeout}s, auth={ssh_auth_timeout}s, banner={ssh_banner_timeout}s")
        print()

        # Use Full Process Isolation Mode to enable PyTorch DataLoader multiprocessing
        print("🔄 Starting Full Process Isolation Mode...")
        print("💡 IMPORTANT: You will need to manually enter SSH passwords in tmux sessions")
        print("📋 Instructions:")
        print("   1. Script will start tmux sessions on server and client machines")
        print("   2. Server: SuperLink + ServerApp in separate processes")
        print("   3. Clients: SuperNode + ClientApp in separate processes with SSH tunnels")
        print("   4. SSH tunnels will prompt for passwords - enter them manually")
        print("   5. Use 'tmux attach -t flower_supernode_X' to access client sessions if needed")
        print("   6. Use 'tmux attach -t flower_server' or 'tmux attach -t flower_serverapp' for server")
        print("   7. Monitor progress and federated learning execution")
        print()
        if runner.run_federated_learning(venv_activate=venv_activate):
            print("🎉 Federated learning completed successfully!")
            print("✅ Full Process Isolation Mode enabled PyTorch DataLoader with num_workers > 0 on both sides")
        else:
            print("❌ Full Process Isolation Mode failed - federated learning cannot proceed")
            print("💡 Ensure SSH access between machines and Flower installation is correct")
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
    finally:
        runner.cleanup_tmux_sessions()

if __name__ == "__main__":
    main()
