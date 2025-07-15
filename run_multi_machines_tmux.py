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
import codecs
import os
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

    def update_config_file(self, ssh_client, config_file_path: str, train_csv_path: str, val_csv_path: str, experiment_idx: int):
        """Update a config file with new data paths for sequential experiments"""
        try:
            print(f"📝 Updating config file: {config_file_path}")
            print(f"   Train CSV: {train_csv_path}")
            print(f"   Val CSV: {val_csv_path}")

            # First, check if the file exists and get the absolute path
            check_command = f"cd {self.project_dir} && ls -la {config_file_path}"
            stdin, stdout, stderr = ssh_client.exec_command(check_command)
            ls_output = stdout.read().decode()
            ls_error = stderr.read().decode()

            if ls_error:
                print(f"❌ Config file not found: {config_file_path}")
                print(f"🔍 Error: {ls_error}")
                # Try to find similar files for debugging
                config_dir = "/".join(config_file_path.split("/")[:-1])
                if config_dir:
                    find_command = f"cd {self.project_dir} && find {config_dir} -name '*.yaml' -type f 2>/dev/null || echo 'Directory not found'"
                    stdin, stdout, stderr = ssh_client.exec_command(find_command)
                    find_output = stdout.read().decode()
                    print(f"🔍 Available config files in {config_dir}:")
                    print(find_output)
                return False

            print(f"✅ Config file found: {ls_output.strip()}")

            # Read current config file
            read_command = f"cd {self.project_dir} && cat {config_file_path}"
            stdin, stdout, stderr = ssh_client.exec_command(read_command)
            config_content = stdout.read().decode()
            error = stderr.read().decode()

            if error:
                print(f"❌ Error reading config file {config_file_path}: {error}")
                return False

            # Update data paths using sed to preserve comments and formatting
            # Extract seed ID from the train_csv_path for wandb naming
            seed_id = "unknown"
            try:
                # Look for pattern like "seed01", "seed10", "seed42", etc.
                import re
                seed_match = re.search(r'seed(\d+)', train_csv_path)
                if seed_match:
                    seed_id = f"seed{seed_match.group(1)}"
                    print(f"🔍 Extracted seed ID: {seed_id}")
                else:
                    print(f"⚠️ Could not extract seed ID from path: {train_csv_path}")
            except Exception as e:
                print(f"⚠️ Error extracting seed ID: {e}")

            # Use sed to update train_csv_path line
            train_update_command = f"cd {self.project_dir} && sed -i 's|^\\s*train_csv_path:.*|  train_csv_path: \"{train_csv_path}\"|' {config_file_path}"
            stdin, stdout, stderr = ssh_client.exec_command(train_update_command)
            train_error = stderr.read().decode()

            if train_error:
                print(f"❌ Error updating train_csv_path: {train_error}")
                return False

            # Use sed to update val_csv_path line
            val_update_command = f"cd {self.project_dir} && sed -i 's|^\\s*val_csv_path:.*|  val_csv_path: \"{val_csv_path}\"|' {config_file_path}"
            stdin, stdout, stderr = ssh_client.exec_command(val_update_command)
            val_error = stderr.read().decode()

            if val_error:
                print(f"❌ Error updating val_csv_path: {val_error}")
                return False

            # Update wandb run_name and notes with seed ID
            if seed_id != "unknown":
                print(f"🏷️ Updating wandb configuration with {seed_id}...")

                # Update wandb run_name: properly remove existing seed suffix and add new one
                # Use a more precise regex that captures everything before the -seed pattern
                run_name_update_command = f"cd {self.project_dir} && sed -i 's|^\\s*run_name:\\s*\"\\(.*\\)-seed[0-9]\\+\".*|  run_name: \"\\1-{seed_id}\"|' {config_file_path}"
                stdin, stdout, stderr = ssh_client.exec_command(run_name_update_command)
                run_name_error = stderr.read().decode()

                if run_name_error:
                    print(f"⚠️ Warning updating wandb run_name: {run_name_error}")

                # Update wandb notes: properly remove existing seed suffix and add new one
                notes_update_command = f"cd {self.project_dir} && sed -i 's|^\\s*notes:\\s*\"\\(.*\\)-seed[0-9]\\+\".*|  notes: \"\\1-{seed_id}\"|' {config_file_path}"
                stdin, stdout, stderr = ssh_client.exec_command(notes_update_command)
                notes_error = stderr.read().decode()

                if notes_error:
                    print(f"⚠️ Warning updating wandb notes: {notes_error}")

                # Extract numeric seed value from seed_id (e.g., "seed01" -> "1", "seed42" -> "42")
                try:
                    numeric_seed = str(int(seed_id.replace("seed", "")))
                    print(f"🔢 Updating training.seed with numeric value: {numeric_seed}")

                    # Update training.seed field
                    training_seed_command = f"cd {self.project_dir} && sed -i 's|^\\s*seed:\\s*[0-9]\\+.*|  seed: {numeric_seed}|' {config_file_path}"
                    stdin, stdout, stderr = ssh_client.exec_command(training_seed_command)
                    training_seed_error = stderr.read().decode()

                    if training_seed_error:
                        print(f"⚠️ Warning updating training.seed: {training_seed_error}")
                    else:
                        print(f"✅ Updated training.seed to: {numeric_seed}")

                except ValueError as e:
                    print(f"⚠️ Could not extract numeric seed from {seed_id}: {e}")

                print(f"✅ Updated wandb configuration with seed ID: {seed_id}")
            else:
                print("⚠️ Skipping wandb update due to unknown seed ID")

            # Verify the changes were made
            verify_command = f"cd {self.project_dir} && grep -E '(train_csv_path|val_csv_path|run_name|notes):' {config_file_path}"
            stdin, stdout, stderr = ssh_client.exec_command(verify_command)
            verify_output = stdout.read().decode()

            if verify_output:
                print(f"✅ Updated configuration verified:")
                for line in verify_output.strip().split('\n'):
                    if line.strip():
                        print(f"   {line.strip()}")
            else:
                print("⚠️ Could not verify configuration updates")

            print(f"✅ Successfully updated config file: {config_file_path}")
            return True

        except Exception as e:
            print(f"❌ Error updating config file {config_file_path}: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def update_server_config_for_experiment(self, experiment_idx: int, server_config: Dict):
        """Update server config file for current experiment"""
        try:
            if not server_config.get("sequential_experiment", False):
                return True

            config_file = server_config.get("config_file")
            train_labels = server_config.get("train_sequential_labels", [])
            val_labels = server_config.get("val_sequential_labels", [])

            if not config_file or experiment_idx >= len(train_labels) or experiment_idx >= len(val_labels):
                print(f"❌ Invalid server config for experiment {experiment_idx}")
                return False

            train_csv_path = train_labels[experiment_idx]
            val_csv_path = val_labels[experiment_idx]

            return self.update_config_file(
                self.server_ssh,
                config_file,
                train_csv_path,
                val_csv_path,
                experiment_idx
            )

        except Exception as e:
            print(f"❌ Error updating server config for experiment {experiment_idx}: {e}")
            return False

    def update_client_configs_for_experiment(self, experiment_idx: int, clients_config: List[Dict]):
        """Update all client config files for current experiment"""
        success_count = 0

        for i, client_config in enumerate(clients_config):
            try:
                if not client_config.get("sequential_experiment", False):
                    success_count += 1
                    continue

                config_file = client_config.get("config_file")
                train_labels = client_config.get("train_sequential_labels", [])
                val_labels = client_config.get("val_sequential_labels", [])

                if not config_file or experiment_idx >= len(train_labels) or experiment_idx >= len(val_labels):
                    print(f"❌ Invalid client config for experiment {experiment_idx}, client {i}")
                    continue

                train_csv_path = train_labels[experiment_idx]
                val_csv_path = val_labels[experiment_idx]

                # Get SSH connection for this client
                ssh_client = self.ssh_connections[i] if i < len(self.ssh_connections) else None
                if not ssh_client:
                    print(f"❌ No SSH connection for client {i}")
                    continue

                if self.update_config_file(
                    ssh_client,
                    config_file,
                    train_csv_path,
                    val_csv_path,
                    experiment_idx
                ):
                    success_count += 1

            except Exception as e:
                print(f"❌ Error updating client {i} config for experiment {experiment_idx}: {e}")

        return success_count == len(clients_config)

    def update_pyproject_toml(self, server_config: Dict, clients_config: List[Dict]) -> bool:
        """Update pyproject.toml with correct config files, app components, and number of supernodes"""
        try:
            print("🔧 Updating pyproject.toml with current experiment configuration...")

            pyproject_path = f"{self.project_dir}/pyproject.toml"

            # Determine strategy from server config to set correct app components
            strategy = "fedavg"  # default
            if server_config.get("config_file"):
                # Try to read strategy from server config file
                try:
                    read_server_config_cmd = f"cd {self.project_dir} && grep -E 'strategy:' {server_config['config_file']} | head -1"
                    stdin, stdout, stderr = self.server_ssh.exec_command(read_server_config_cmd)
                    strategy_line = stdout.read().decode().strip()
                    if strategy_line:
                        # Extract strategy value from YAML line like "strategy: fedavg" or "strategy: \"secagg+\""
                        import re
                        strategy_match = re.search(r'strategy:\s*["\']?([^"\']+)["\']?', strategy_line)
                        if strategy_match:
                            strategy = strategy_match.group(1).strip().lower()
                            print(f"🔍 Detected strategy: {strategy}")
                except Exception as e:
                    print(f"⚠️ Could not detect strategy, using default: {e}")

            # Determine app components based on strategy
            if strategy in ["secagg+", "secaggplus"]:
                serverapp_component = "adni_flwr.server_app:secagg_plus_app"
                clientapp_component = "adni_flwr.client_app:secagg_plus_app"
                print(f"🔒 Using SecAgg+ components for strategy: {strategy}")
            else:
                serverapp_component = "adni_flwr.server_app:app"
                clientapp_component = "adni_flwr.client_app:app"
                print(f"📊 Using standard components for strategy: {strategy}")

            # Read current pyproject.toml
            read_command = f"cat {pyproject_path}"
            stdin, stdout, stderr = self.server_ssh.exec_command(read_command)
            content = stdout.read().decode()
            error = stderr.read().decode()

            if error:
                print(f"❌ Error reading pyproject.toml: {error}")
                return False

            if not content.strip():
                print("❌ pyproject.toml is empty")
                return False

            lines = content.split('\n')
            updated_lines = []
            in_app_components = False
            in_app_config = False
            in_federation_config = False

            for line in lines:
                # Track sections
                if line.strip() == "[tool.flwr.app.components]":
                    in_app_components = True
                    in_app_config = False
                    in_federation_config = False
                elif line.strip() == "[tool.flwr.app.config]":
                    in_app_components = False
                    in_app_config = True
                    in_federation_config = False
                elif line.strip() == "[tool.flwr.federations.multi-machine]":
                    in_app_components = False
                    in_app_config = False
                    in_federation_config = True
                elif line.strip().startswith("[") and line.strip().endswith("]"):
                    in_app_components = False
                    in_app_config = False
                    in_federation_config = False

                # Update app components based on strategy
                if in_app_components and line.strip().startswith("serverapp"):
                    updated_lines.append(f'serverapp = "{serverapp_component}"')
                    print(f"📝 Updated serverapp: {serverapp_component}")
                elif in_app_components and line.strip().startswith("clientapp"):
                    updated_lines.append(f'clientapp = "{clientapp_component}"')
                    print(f"📝 Updated clientapp: {clientapp_component}")

                # Update client-config-files
                elif in_app_config and line.strip().startswith("client-config-files"):
                    client_config_files = []
                    for client_config in clients_config:
                        config_file = client_config.get("config_file")
                        if config_file:
                            client_config_files.append(config_file)

                    if client_config_files:
                        client_config_str = ",".join(client_config_files)
                        updated_lines.append(f'client-config-files = "{client_config_str}"')
                        print(f"📝 Updated client-config-files: {client_config_str}")
                    else:
                        updated_lines.append(line)

                # Update server-config-file
                elif in_app_config and line.strip().startswith("server-config-file"):
                    server_config_file = server_config.get("config_file")
                    if server_config_file:
                        updated_lines.append(f'server-config-file = "{server_config_file}"')
                        print(f"📝 Updated server-config-file: {server_config_file}")
                    else:
                        updated_lines.append(line)

                # Update options.num-supernodes
                elif in_federation_config and line.strip().startswith("options.num-supernodes"):
                    num_clients = len(clients_config)
                    updated_lines.append(f'options.num-supernodes = {num_clients}')
                    print(f"📝 Updated options.num-supernodes: {num_clients}")

                else:
                    updated_lines.append(line)

            # Write updated content back
            updated_content = '\n'.join(updated_lines)
            write_command = f"cat > {pyproject_path} << 'EOF'\n{updated_content}\nEOF"
            stdin, stdout, stderr = self.server_ssh.exec_command(write_command)
            error = stderr.read().decode()

            if error:
                print(f"❌ Error writing pyproject.toml: {error}")
                return False

            print("✅ Successfully updated pyproject.toml")
            return True

        except Exception as e:
            print(f"❌ Error updating pyproject.toml: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def get_num_experiments(self, server_config: Dict, clients_config: List[Dict]) -> int:
        """Get the number of sequential experiments based on config"""
        try:
            # Check if sequential experiment is enabled
            if not server_config.get("sequential_experiment", False):
                return 1  # Single experiment

            # Get number of experiments from server config
            server_train_labels = server_config.get("train_sequential_labels", [])
            server_val_labels = server_config.get("val_sequential_labels", [])

            if not server_train_labels or not server_val_labels:
                return 1

            server_experiments = min(len(server_train_labels), len(server_val_labels))

            # Validate that all clients have the same number of experiments
            for i, client_config in enumerate(clients_config):
                if not client_config.get("sequential_experiment", False):
                    continue

                client_train_labels = client_config.get("train_sequential_labels", [])
                client_val_labels = client_config.get("val_sequential_labels", [])

                if not client_train_labels or not client_val_labels:
                    print(f"❌ Client {i} has no sequential labels")
                    return 1

                client_experiments = min(len(client_train_labels), len(client_val_labels))

                if client_experiments != server_experiments:
                    print(f"❌ Client {i} has {client_experiments} experiments but server has {server_experiments}")
                    return 1

            print(f"✅ Found {server_experiments} sequential experiments")
            return server_experiments

        except Exception as e:
            print(f"❌ Error determining number of experiments: {e}")
            return 1

    def stream_channel_output(self, channel, output_buffer_ref, timeout_counter_ref, max_timeout):
        """
        Robust UTF-8 streaming from SSH channel with proper handling of partial characters.
        Returns (should_continue, completion_detected)
        """
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        try:
            if not channel.recv_ready():
                return True, False

            raw_data = channel.recv(4096)
            if not raw_data:
                return True, False

            # Use incremental decoder to handle partial UTF-8 sequences
            text_data = decoder.decode(raw_data, final=False)

            if text_data:  # Only process if we got actual text
                output_buffer_ref[0] += text_data
                print(text_data, end="")
                timeout_counter_ref[0] = 0  # Reset timeout when receiving data

                # Check for completion indicators
                if any(phrase in output_buffer_ref[0].lower() for phrase in [
                    "run finished", "completed successfully", "experiment completed",
                    "training finished", "federation completed"
                ]):
                    print("\n🎉 Detected completion signal!")
                    return False, True  # Stop processing, completion detected

            return True, False  # Continue processing

        except Exception as e:
            print(f"\n⚠️ Stream processing error: {e}")
            return True, False  # Continue despite error

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

    def start_flower_superlink_only(self, venv_activate: str):
        """Start only SuperLink (infrastructure) - ServerApp will be started per experiment"""
        try:
            print(f"🌸 Starting Flower SuperLink on {self.server_config['host']} using tmux...")
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
                "tmux kill-session -t flower_serverapp 2>/dev/null || true",
                "pkill -f 'flower-superlink' || true",
                "pkill -f 'flwr-serverapp' || true"
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
            print(f"🚀 Starting SuperLink with Process Isolation Mode")
            stdin, stdout, stderr = self.server_ssh.exec_command(send_command)

            # Wait for SuperLink to start
            print("⏳ Waiting for SuperLink to start...")
            time.sleep(10)

            # Check if SuperLink is running
            check_superlink_session = "tmux list-sessions | grep flower_server"
            stdin, stdout, stderr = self.server_ssh.exec_command(check_superlink_session)
            superlink_session_output = stdout.read().decode()

            check_superlink_process = "pgrep -f 'flower-superlink'"
            stdin, stdout, stderr = self.server_ssh.exec_command(check_superlink_process)
            superlink_pids = stdout.read().decode().strip()

            check_fleet_port_command = "netstat -tuln | grep :9092 || ss -tuln | grep :9092"
            stdin, stdout, stderr = self.server_ssh.exec_command(check_fleet_port_command)
            fleet_port_output = stdout.read().decode()

            if superlink_session_output and superlink_pids and fleet_port_output:
                print(f"✅ SuperLink infrastructure started successfully")
                print(f"📺 SuperLink session: {superlink_session_output.strip()}")
                print(f"🔧 SuperLink PID: {superlink_pids}")
                print(f"🌐 Fleet API (port 9092): {fleet_port_output.strip()}")
                print(f"🔧 ServerApp will be started fresh for each experiment")
                return True
            else:
                print("❌ Failed to start SuperLink infrastructure")
                return False

        except Exception as e:
            print(f"❌ Error starting SuperLink: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def start_serverapp_for_experiment(self, venv_activate: str, experiment_idx: int) -> bool:
        """Start fresh ServerApp for a specific experiment"""
        try:
            print(f"🔧 Starting fresh ServerApp for experiment {experiment_idx + 1}...")

            # Kill any existing ServerApp
            self.server_ssh.exec_command("tmux kill-session -t flower_serverapp 2>/dev/null || true")
            self.server_ssh.exec_command("pkill -f 'flwr-serverapp' || true")
            time.sleep(2)

            # Create new tmux session for ServerApp
            serverapp_session_cmd = "tmux new-session -d -s flower_serverapp"
            stdin, stdout, stderr = self.server_ssh.exec_command(serverapp_session_cmd)
            time.sleep(2)

            # Start fresh ServerApp
            logs_prefix = "logs/" if self.ensure_logs_directory(self.server_ssh, self.project_dir) else ""
            serverapp_log = f"{logs_prefix}serverapp_exp{experiment_idx + 1}_{self.timestamp}.log"
            serverapp_command = self.get_venv_command(
                f'cd {self.project_dir} && flwr-serverapp --serverappio-api-address 127.0.0.1:9091 --insecure 2>&1 | tee {serverapp_log}',
                venv_activate
            )

            send_serverapp_cmd = f'tmux send-keys -t flower_serverapp "{serverapp_command}" Enter'
            stdin, stdout, stderr = self.server_ssh.exec_command(send_serverapp_cmd)
            time.sleep(3)

            # Verify ServerApp is running
            check_serverapp_process = "pgrep -f 'flwr-serverapp'"
            stdin, stdout, stderr = self.server_ssh.exec_command(check_serverapp_process)
            serverapp_pids = stdout.read().decode().strip()

            if serverapp_pids:
                print(f"✅ Fresh ServerApp started for experiment {experiment_idx + 1} (PID: {serverapp_pids})")
                return True
            else:
                print(f"❌ Failed to start ServerApp for experiment {experiment_idx + 1}")
                return False

        except Exception as e:
            print(f"❌ Error starting ServerApp for experiment {experiment_idx + 1}: {e}")
            return False

    def stop_serverapp(self):
        """Stop ServerApp after experiment"""
        try:
            self.server_ssh.exec_command("tmux kill-session -t flower_serverapp 2>/dev/null || true")
            self.server_ssh.exec_command("pkill -f 'flwr-serverapp' || true")
            time.sleep(1)
            print("🛑 ServerApp stopped")
        except Exception as e:
            print(f"⚠️ Error stopping ServerApp: {e}")

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

    def start_flower_supernode_only(self, client_config: Dict, venv_activate: str):
        """Start only SuperNode with SSH tunnel (infrastructure) - ClientApp will be started per experiment"""
        client_host = client_config['host']
        partition_id = client_config.get('partition_id', 0)
        supernode_session = f"flower_supernode_{partition_id}"

        try:
            print(f"🌻 Starting SuperNode infrastructure on {client_host} with SSH tunnel...")

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
                f"tmux kill-session -t flower_clientapp_{partition_id} 2>/dev/null || true",
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

            local_tunnel_port = 9092 + partition_id  # Each client gets a unique local port
            client_port = 9094 + partition_id
            supernode_log = f"{logs_prefix}supernode_{client_host.split('.')[0]}_{self.timestamp}.log"

            print(f"🔧 Client {partition_id} will use local tunnel port {local_tunnel_port}")

            # Create SSH tunnel and start SuperNode
            supernode_session_cmd = f"tmux new-session -d -s {supernode_session}"
            stdin, stdout, stderr = ssh.exec_command(supernode_session_cmd)
            time.sleep(2)

            # SuperNode command with SSH tunnel
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
            print(f"🚀 Started SuperNode infrastructure")
            print(f"💡 SSH tunnel will prompt for password in tmux session '{supernode_session}' on {client_host}")
            print(f"📋 To enter password: ssh {client_host} && tmux attach -t {supernode_session}")

            # Wait longer for user to input password and tunnel to be established
            time.sleep(15)

            # Verify SuperNode is running
            check_supernode_session = f"tmux list-sessions | grep {supernode_session}"
            stdin, stdout, stderr = ssh.exec_command(check_supernode_session)
            supernode_session_output = stdout.read().decode()

            check_tunnel_process = f"pgrep -f 'ssh.*-L {local_tunnel_port}:{self.server_config['host']}:9092'"
            stdin, stdout, stderr = ssh.exec_command(check_tunnel_process)
            tunnel_pid = stdout.read().decode().strip()

            check_supernode_process = "pgrep -f 'flower-supernode'"
            stdin, stdout, stderr = ssh.exec_command(check_supernode_process)
            supernode_pids = stdout.read().decode().strip()

            if supernode_session_output and tunnel_pid and supernode_pids:
                print(f"✅ SuperNode infrastructure started on {client_host}")
                print(f"📺 SuperNode session: {supernode_session_output.strip()}")
                print(f"🔧 SSH tunnel: Active on local port {local_tunnel_port} (PID: {tunnel_pid})")
                print(f"🔧 SuperNode PID: {supernode_pids}")
                print(f"🔧 ClientApp will be started fresh for each experiment")
                self.ssh_connections.append(ssh)
                return True
            else:
                print(f"❌ Failed to start SuperNode infrastructure on {client_host}")
                ssh.close()
                return False

        except Exception as e:
            print(f"❌ Error starting SuperNode infrastructure on {client_host}: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def start_clientapp_for_experiment(self, client_idx: int, venv_activate: str, experiment_idx: int) -> bool:
        """Start fresh ClientApp for a specific experiment"""
        try:
            client_config = self.clients_config[client_idx]
            partition_id = client_config.get('partition_id', 0)
            client_host = client_config['host']
            client_port = 9094 + partition_id
            clientapp_session = f"flower_clientapp_{partition_id}"

            print(f"🔧 Starting fresh ClientApp for experiment {experiment_idx + 1} on {client_host}...")

            ssh = self.ssh_connections[client_idx]

            # Kill any existing ClientApp
            ssh.exec_command(f"tmux kill-session -t {clientapp_session} 2>/dev/null || true")
            ssh.exec_command("pkill -f 'flwr-clientapp' || true")
            time.sleep(2)

            # Create new tmux session for ClientApp
            clientapp_session_cmd = f"tmux new-session -d -s {clientapp_session}"
            stdin, stdout, stderr = ssh.exec_command(clientapp_session_cmd)
            time.sleep(2)

            # Start fresh ClientApp
            logs_prefix = "logs/" if self.ensure_logs_directory(ssh, client_config["project_dir"]) else ""
            clientapp_log = f"{logs_prefix}clientapp_{client_host.split('.')[0]}_exp{experiment_idx + 1}_{self.timestamp}.log"
            clientapp_command = self.get_venv_command(
                f'cd {client_config["project_dir"]} && '
                f'flwr-clientapp --clientappio-api-address 127.0.0.1:{client_port} --insecure '
                f'2>&1 | tee {clientapp_log}',
                venv_activate
            )

            send_clientapp_cmd = f'tmux send-keys -t {clientapp_session} "{clientapp_command}" Enter'
            stdin, stdout, stderr = ssh.exec_command(send_clientapp_cmd)
            time.sleep(3)

            # Verify ClientApp is running
            check_clientapp_process = "pgrep -f 'flwr-clientapp'"
            stdin, stdout, stderr = ssh.exec_command(check_clientapp_process)
            clientapp_pids = stdout.read().decode().strip()

            if clientapp_pids:
                print(f"✅ Fresh ClientApp started for experiment {experiment_idx + 1} on {client_host} (PID: {clientapp_pids})")
                return True
            else:
                print(f"❌ Failed to start ClientApp for experiment {experiment_idx + 1} on {client_host}")
                return False

        except Exception as e:
            print(f"❌ Error starting ClientApp for experiment {experiment_idx + 1} on client {client_idx}: {e}")
            return False

    def stop_clientapps(self):
        """Stop all ClientApps after experiment"""
        try:
            for i, ssh in enumerate(self.ssh_connections):
                partition_id = self.clients_config[i].get('partition_id', 0)
                clientapp_session = f"flower_clientapp_{partition_id}"
                ssh.exec_command(f"tmux kill-session -t {clientapp_session} 2>/dev/null || true")
                ssh.exec_command("pkill -f 'flwr-clientapp' || true")
            time.sleep(1)
            print("🛑 All ClientApps stopped")
        except Exception as e:
            print(f"⚠️ Error stopping ClientApps: {e}")

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
        """Run the Flower App with intelligent timeout and process monitoring"""
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

            # Enhanced streaming with intelligent timeout and process monitoring
            output_buffer = ""
            silence_counter = 0
            total_runtime = 0
            max_silence_time = 1800  # 15 minutes of silence before checking processes
            max_total_time = 14400   # 4 hours maximum total runtime
            check_interval = 30      # Check every 30 seconds

            while total_runtime < max_total_time:
                data_received = False

                # Check for new output
                if channel.recv_ready():
                    try:
                        data = channel.recv(4096).decode('utf-8', errors='replace')
                        if data:
                            output_buffer += data
                            print(data, end="")
                            silence_counter = 0  # Reset silence counter when data received
                            data_received = True

                            # Check for completion indicators
                            completion_phrases = [
                                "run finished", "completed successfully", "experiment completed",
                                "training finished", "federation completed", "fl training completed",
                            ]
                            if any(phrase in output_buffer.lower() for phrase in completion_phrases):
                                print("\n🎉 Detected completion signal in output!")
                                break

                    except UnicodeDecodeError as decode_error:
                        print(f"\n⚠️ UTF-8 decode warning: {decode_error}")
                        silence_counter = 0  # Reset on any activity

                # Check if command completed
                if channel.exit_status_ready():
                    print("\n🏁 Command completed - collecting remaining output...")
                    # Get any remaining output
                    remaining_output = ""
                    while channel.recv_ready():
                        try:
                            remaining_data = channel.recv(4096).decode('utf-8', errors='replace')
                            remaining_output += remaining_data
                            print(remaining_data, end="")
                        except UnicodeDecodeError:
                            pass

                    # Check exit status
                    exit_status = channel.recv_exit_status()
                    print(f"\n📊 Process exit status: {exit_status}")

                    if exit_status == 0:
                        print("✅ Flower App completed successfully!")
                        break
                    else:
                        print(f"⚠️ Flower App exited with status {exit_status}")
                        print("🔍 This might be normal depending on your FL configuration")
                        break

                # Wait and update counters
                time.sleep(1)
                total_runtime += 1
                if not data_received:
                    silence_counter += 1

                # Periodic status updates and process checks
                if total_runtime % check_interval == 0:
                    print(f"\n⏱️ Runtime: {total_runtime//60}m{total_runtime%60}s, Silence: {silence_counter//60}m{silence_counter%60}s")

                    # Check if FL processes are still running
                    processes_running = self.check_fl_processes_running()
                    if not processes_running:
                        print("🛑 FL processes no longer running - experiment completed or failed")
                        break

                    # If silent for too long, do detailed process check
                    if silence_counter > max_silence_time:
                        print(f"⚠️ No output for {silence_counter//60} minutes - checking process status...")

                        # Check if flwr run process is still active
                        check_flwr_process = "pgrep -f 'flwr run' || echo 'no-process'"
                        stdin, stdout, stderr = self.server_ssh.exec_command(check_flwr_process)
                        flwr_process = stdout.read().decode().strip()

                        if flwr_process == "no-process":
                            print("🛑 Flower App process no longer running - assuming completion")
                            break
                        else:
                            print(f"✅ Flower App still running (PID: {flwr_process}) - continuing to monitor...")
                            silence_counter = 0  # Reset silence counter after verification

            # Handle timeout scenarios
            if total_runtime >= max_total_time:
                print(f"\n⏰ Maximum runtime reached ({max_total_time//3600}h {(max_total_time%3600)//60}m)")
                print("🔍 Checking if processes completed successfully...")

                # Final process check
                if not self.check_fl_processes_running():
                    print("✅ FL processes completed - experiment finished")
                else:
                    print("⚠️ FL processes still running after timeout - manual intervention may be needed")

            print("\n" + "=" * 50)
            print("✅ Flower App monitoring completed!")
            channel.close()
            return True

        except Exception as e:
            print(f"❌ Error running Flower App: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def check_fl_processes_running(self) -> bool:
        """Check if FL processes (ServerApp and ClientApps) are still running"""
        try:
            # Check ServerApp
            stdin, stdout, stderr = self.server_ssh.exec_command("pgrep -f 'flwr-serverapp'")
            serverapp_running = bool(stdout.read().decode().strip())

            # Check ClientApps
            clientapps_running = 0
            for ssh in self.ssh_connections:
                try:
                    stdin, stdout, stderr = ssh.exec_command("pgrep -f 'flwr-clientapp'")
                    if stdout.read().decode().strip():
                        clientapps_running += 1
                except:
                    pass

            # Consider FL running if ServerApp is running and at least one ClientApp is running
            fl_active = serverapp_running and clientapps_running > 0

            if fl_active:
                print(f"🔄 FL processes active: ServerApp={serverapp_running}, ClientApps={clientapps_running}/{len(self.ssh_connections)}")

            return fl_active

        except Exception as e:
            print(f"⚠️ Error checking FL processes: {e}")
            return True  # Assume running on error to avoid premature termination

    def cleanup_tmux_sessions(self):
        """Clean up all tmux sessions and processes"""
        print("\n🧹 Cleaning up infrastructure and FL components...")

        # First stop any running FL components
        try:
            self.stop_serverapp()
            self.stop_clientapps()
        except:
            pass

        # Cleanup server infrastructure (SuperLink)
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
                print("✓ SuperLink infrastructure stopped")
                self.server_ssh.close()
            except Exception as e:
                print(f"⚠️ Error cleaning up server: {e}")

        # Cleanup client infrastructure (SuperNode + SSH tunnels)
        for i, ssh in enumerate(self.ssh_connections):
            try:
                partition_id = self.clients_config[i].get('partition_id', 0)
                local_tunnel_port = 9092 + partition_id
                cleanup_commands = [
                    f"tmux kill-session -t flower_supernode_{partition_id} 2>/dev/null || true",
                    f"tmux kill-session -t flower_clientapp_{partition_id} 2>/dev/null || true",
                    "pkill -f 'flower-supernode' || true",
                    "pkill -f 'flwr-clientapp' || true",
                    f"pkill -f 'ssh.*-L {local_tunnel_port}:{self.server_config['host']}:9092' || true",
                    f"fuser -k {local_tunnel_port}/tcp 2>/dev/null || true"
                ]
                for cmd in cleanup_commands:
                    ssh.exec_command(cmd)
                ssh.close()
                print(f"✓ SuperNode infrastructure and SSH tunnel stopped on client {i+1}")
            except Exception as e:
                print(f"⚠️ Error cleaning up client {i+1}: {e}")

        print("✅ Infrastructure cleanup completed")

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
        """Main method to run federated learning with Process Isolation Mode and Sequential Experiments"""
        print("🚀 Starting Flower Federated Learning with Full Process Isolation Mode")
        print("🔧 Server Process Isolation: SuperLink + ServerApp in separate processes")
        print("🔧 Client Process Isolation: SuperNode + ClientApp in separate processes")
        print("✅ PyTorch DataLoader multiprocessing enabled on both server and client (num_workers > 0)")
        print("🔗 Using SSH tunnel for secure communication")
        print("=" * 70)

        # Debug: Show configuration paths
        print("🔍 Debug: Configuration file paths:")
        print(f"   Server config file: {self.server_config.get('config_file', 'NOT SET')}")
        for i, client_config in enumerate(self.clients_config):
            print(f"   Client {i+1} config file: {client_config.get('config_file', 'NOT SET')}")
        print("=" * 70)

        # Determine number of experiments
        num_experiments = self.get_num_experiments(self.server_config, self.clients_config)
        is_sequential = num_experiments > 1

        if is_sequential:
            print(f"🔄 Sequential Experiment Mode: {num_experiments} experiments")
            print("📋 Process: Setup infrastructure → Run experiments → Keep infrastructure running")
        else:
            print("🔄 Single Experiment Mode")

        print("=" * 70)

        # Start server infrastructure (SuperLink only)
        if not self.start_flower_superlink_only(venv_activate):
            print("❌ Failed to start SuperLink infrastructure. Exiting...")
            return False

        print("⏳ Waiting for SuperLink to be ready...")
        time.sleep(10)

        # Start client infrastructure (SuperNode only)
        success_count = 0
        for i, client_config in enumerate(self.clients_config):
            print(f"Starting SuperNode infrastructure {i+1}/{len(self.clients_config)} on {client_config['host']}...")
            print(f"💡 To manually enter SSH password, connect to {client_config['host']} and run:")
            print(f"   tmux attach -t flower_supernode_{i}")
            print()

            # Start SuperNode infrastructure only
            success = self.start_flower_supernode_only(client_config, venv_activate)

            if success:
                success_count += 1

            time.sleep(5)

        if success_count <= 0:
            print("❌ No SuperNode infrastructure started successfully")
            return False

        print(f"✅ {success_count}/{len(self.clients_config)} SuperNode infrastructure started!")
        print("🔧 Infrastructure: SuperLink + SuperNode (stays running)")
        print("🔧 Per experiment: Fresh ServerApp + ClientApp (restarted each time)")
        print("✅ PyTorch DataLoader multiprocessing is now enabled on both server and client sides")

        print("⏳ Waiting for all components to connect...")
        time.sleep(10)

        # Monitor sessions before starting FL
        self.monitor_tmux_sessions()

        # Run experiments sequentially
        successful_experiments = 0
        for experiment_idx in range(num_experiments):
            print(f"\n{'='*70}")
            print(f"🧪 Starting Experiment {experiment_idx + 1}/{num_experiments}")

            if is_sequential:
                # Extract experiment info from the first train label path
                experiment_info = "Unknown"
                try:
                    server_train_labels = self.server_config.get("train_sequential_labels", [])
                    if server_train_labels and experiment_idx < len(server_train_labels):
                        # Extract seed from path like "seed01", "seed10", etc.
                        label_path = server_train_labels[experiment_idx]
                        if "seed" in label_path:
                            seed_part = label_path.split("seed")[1].split("/")[0][:2]
                            experiment_info = f"Seed {seed_part}"
                except:
                    pass

                print(f"📊 Cross-validation fold: {experiment_info}")
                print(f"🔧 Updating config files for experiment {experiment_idx + 1}...")

                # Update server config file
                if not self.update_server_config_for_experiment(experiment_idx, self.server_config):
                    print(f"❌ Failed to update server config for experiment {experiment_idx + 1}")
                    continue

                # Update client config files
                if not self.update_client_configs_for_experiment(experiment_idx, self.clients_config):
                    print(f"❌ Failed to update client configs for experiment {experiment_idx + 1}")
                    continue

                print(f"✅ Config files updated for experiment {experiment_idx + 1}")

                # Wait a moment for config files to be ready
                time.sleep(3)

            # Update pyproject.toml with correct config files for this experiment
            print(f"🔧 Updating pyproject.toml for experiment {experiment_idx + 1}...")
            if not self.update_pyproject_toml(self.server_config, self.clients_config):
                print("❌ Failed to update pyproject.toml. Continuing anyway...")

            # Wait a moment for pyproject.toml to be ready
            time.sleep(5)

            # Start fresh ServerApp for this experiment
            print(f"🔧 Starting fresh ServerApp for experiment {experiment_idx + 1}...")
            if not self.start_serverapp_for_experiment(venv_activate, experiment_idx):
                print(f"❌ Failed to start ServerApp for experiment {experiment_idx + 1}")
                continue

            # Start fresh ClientApps for this experiment
            print(f"🔧 Starting fresh ClientApps for experiment {experiment_idx + 1}...")
            clientapp_success_count = 0
            for client_idx in range(len(self.clients_config)):
                if self.start_clientapp_for_experiment(client_idx, venv_activate, experiment_idx):
                    clientapp_success_count += 1

            if clientapp_success_count != len(self.clients_config):
                print(f"❌ Only {clientapp_success_count}/{len(self.clients_config)} ClientApps started for experiment {experiment_idx + 1}")
                self.stop_serverapp()
                self.stop_clientapps()
                continue

            print(f"✅ Fresh FL components started: 1 ServerApp + {clientapp_success_count} ClientApps")

            # Wait for components to connect
            print("⏳ Waiting for FL components to connect...")
            time.sleep(10)

            print(f"🌸 Starting Flower App execution for experiment {experiment_idx + 1}...")

            # Run Flower App for this experiment
            if self.run_flower_app(venv_activate, federation_name="multi-machine"):
                print(f"✅ Experiment {experiment_idx + 1} completed successfully!")
                successful_experiments += 1
            else:
                print(f"❌ Experiment {experiment_idx + 1} failed")

                # For sequential experiments, ask if user wants to continue
                if is_sequential and experiment_idx < num_experiments - 1:
                    print("⚠️ Sequential experiment failed. Infrastructure is still running.")
                    print("🔧 You can:")
                    print("   1. Fix the issue and restart the script")
                    print("   2. Continue with next experiment (the current failure will be recorded)")
                    # Stop FL components and continue to next experiment
                    self.stop_serverapp()
                    self.stop_clientapps()
                    continue
                else:
                    # Stop FL components after failed experiment
                    self.stop_serverapp()
                    self.stop_clientapps()
                    break

            # Stop FL components after successful experiment
            print(f"🛑 Stopping FL components after experiment {experiment_idx + 1}...")
            self.stop_serverapp()
            self.stop_clientapps()

            # Add delay between experiments to allow cleanup
            if is_sequential and experiment_idx < num_experiments - 1:
                print(f"⏳ Waiting 10 seconds before next experiment...")
                time.sleep(10)

        print(f"\n{'='*70}")
        print(f"🎯 Sequential Experiment Summary:")
        print(f"   Total experiments: {num_experiments}")
        print(f"   Successful: {successful_experiments}")
        print(f"   Failed: {num_experiments - successful_experiments}")

        if successful_experiments == num_experiments:
            print("🎉 All experiments completed successfully!")
            print("✅ Fresh FL components per experiment resolved multiprocessing and state issues!")
            return True
        elif successful_experiments > 0:
            print(f"⚠️ {successful_experiments}/{num_experiments} experiments completed successfully")
            print("💡 Check logs for failed experiments")
            return True
        else:
            print("❌ All experiments failed")
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
            print("✅ Fresh FL components per experiment resolved multiprocessing and state issues!")
        else:
            print("❌ Sequential experiments failed - federated learning cannot proceed")
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
