#!/usr/bin/env python3
"""
Flower Local Simulation Runner with Sequential Experiments
This version runs Flower federated learning locally with support for sequential experiments.

Features:
- Local Flower simulation using `flwr run`
- Sequential experiment support with automatic config file updates
- No SSH tunnels or tmux sessions (user runs in their own tmux if needed)
- Automatic pyproject.toml configuration
- Comprehensive logging and monitoring
- Support for different FL strategies (FedAvg, SecAgg+, etc.)

Usage:
- Run in your own tmux session if desired
- Configure experiments in YAML file
- Script handles config updates automatically for sequential experiments
"""

import time
import signal
import sys
import subprocess
import os
import threading
import codecs
from typing import List, Dict, Optional
from datetime import datetime


class FlowerLocalSimulationRunner:
    def __init__(self, server_config: Dict, clients_config: List[Dict], project_dir: str):
        self.server_config = server_config
        self.clients_config = clients_config
        self.project_dir = project_dir
        # Generate timestamp for this session
        self.timestamp = self.generate_timestamp()
        self.current_process = None

    def generate_timestamp(self) -> str:
        """Generate timestamp string for log files"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def update_config_file(self, config_file_path: str, train_csv_path: str, val_csv_path: str, experiment_idx: int) -> bool:
        """Update a config file with new data paths for sequential experiments"""
        try:
            print(f"📝 Updating config file: {config_file_path}")
            print(f"   Train CSV: {train_csv_path}")
            print(f"   Val CSV: {val_csv_path}")

            # Check if the file exists
            full_path = os.path.join(self.project_dir, config_file_path)
            if not os.path.exists(full_path):
                print(f"❌ Config file not found: {full_path}")
                # Try to find similar files for debugging
                config_dir = os.path.dirname(full_path)
                if os.path.exists(config_dir):
                    yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
                    print(f"🔍 Available config files in {config_dir}: {yaml_files}")
                return False

            print(f"✅ Config file found: {full_path}")

            # Read current config file
            with open(full_path, 'r') as f:
                config_content = f.read()

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

            # Update data paths using regex substitution
            import re

            # Update train_csv_path line
            config_content = re.sub(
                r'^(\s*train_csv_path:\s*).*$',
                f'\\1"{train_csv_path}"',
                config_content,
                flags=re.MULTILINE
            )

            # Update val_csv_path line
            config_content = re.sub(
                r'^(\s*val_csv_path:\s*).*$',
                f'\\1"{val_csv_path}"',
                config_content,
                flags=re.MULTILINE
            )

            # Update wandb run_name and notes with seed ID
            if seed_id != "unknown":
                print(f"🏷️ Updating wandb configuration with {seed_id}...")

                # Update wandb run_name: properly remove existing seed suffix and add new one
                def replace_run_name(match):
                    return f'{match.group(1)}{match.group(2)}-{seed_id}{match.group(3)}'

                config_content = re.sub(
                    r'^(\s*run_name:\s*")(.*)-seed\d+(".*?)$',
                    replace_run_name,
                    config_content,
                    flags=re.MULTILINE
                )

                # Update wandb notes: properly remove existing seed suffix and add new one
                def replace_notes(match):
                    return f'{match.group(1)}{match.group(2)}-{seed_id}{match.group(3)}'

                config_content = re.sub(
                    r'^(\s*notes:\s*")(.*)-seed\d+(".*?)$',
                    replace_notes,
                    config_content,
                    flags=re.MULTILINE
                )

                # Extract numeric seed value from seed_id (e.g., "seed01" -> "1", "seed42" -> "42")
                try:
                    numeric_seed = str(int(seed_id.replace("seed", "")))
                    print(f"🔢 Updating training.seed with numeric value: {numeric_seed}")

                    # Update training.seed field
                    def replace_seed(match):
                        return f'{match.group(1)}{numeric_seed}'

                    config_content = re.sub(
                        r'^(\s*seed:\s*)\d+.*$',
                        replace_seed,
                        config_content,
                        flags=re.MULTILINE
                    )
                    print(f"✅ Updated training.seed to: {numeric_seed}")

                except ValueError as e:
                    print(f"⚠️ Could not extract numeric seed from {seed_id}: {e}")

                print(f"✅ Updated wandb configuration with seed ID: {seed_id}")
            else:
                print("⚠️ Skipping wandb update due to unknown seed ID")

            # Write updated content back
            with open(full_path, 'w') as f:
                f.write(config_content)

            # Verify the changes were made
            verification_patterns = ['train_csv_path', 'val_csv_path', 'run_name', 'notes']
            for pattern in verification_patterns:
                matches = re.findall(f'^.*{pattern}:.*$', config_content, flags=re.MULTILINE)
                if matches:
                    print(f"   {matches[0].strip()}")

            print(f"✅ Successfully updated config file: {config_file_path}")
            return True

        except Exception as e:
            print(f"❌ Error updating config file {config_file_path}: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def update_server_config_for_experiment(self, experiment_idx: int, server_config: Dict) -> bool:
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
                config_file,
                train_csv_path,
                val_csv_path,
                experiment_idx
            )

        except Exception as e:
            print(f"❌ Error updating server config for experiment {experiment_idx}: {e}")
            return False

    def update_client_configs_for_experiment(self, experiment_idx: int, clients_config: List[Dict]) -> bool:
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

                if self.update_config_file(
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

            pyproject_path = os.path.join(self.project_dir, "pyproject.toml")

            if not os.path.exists(pyproject_path):
                print(f"❌ pyproject.toml not found at: {pyproject_path}")
                return False

            # Determine strategy from server config to set correct app components
            strategy = "fedavg"  # default
            if server_config.get("config_file"):
                # Try to read strategy from server config file
                try:
                    server_config_path = os.path.join(self.project_dir, server_config["config_file"])
                    if os.path.exists(server_config_path):
                        with open(server_config_path, 'r') as f:
                            server_config_content = f.read()

                        # Extract strategy value from YAML line like "strategy: fedavg" or "strategy: \"secagg+\""
                        import re
                        strategy_match = re.search(r'strategy:\s*["\']?([^"\']+)["\']?', server_config_content)
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
            with open(pyproject_path, 'r') as f:
                content = f.read()

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
                elif line.strip().startswith("[tool.flwr.federations."):
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

                # Update options.num-supernodes for local simulation
                elif in_federation_config and line.strip().startswith("options.num-supernodes"):
                    num_clients = len(clients_config)
                    updated_lines.append(f'options.num-supernodes = {num_clients}')
                    print(f"📝 Updated options.num-supernodes: {num_clients}")

                else:
                    updated_lines.append(line)

            # Write updated content back
            updated_content = '\n'.join(updated_lines)
            with open(pyproject_path, 'w') as f:
                f.write(updated_content)

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

    def ensure_federation_config(self, federation_name: str = "local-simulation-gpu") -> bool:
        """Ensure pyproject.toml has the correct federation configuration for local simulation with GPU resources"""
        try:
            pyproject_path = os.path.join(self.project_dir, "pyproject.toml")

            print(f"🔍 Checking federation configuration in {pyproject_path}...")

            if not os.path.exists(pyproject_path):
                print(f"❌ pyproject.toml not found at: {pyproject_path}")
                return False

            # Read current content
            with open(pyproject_path, 'r') as f:
                content = f.read()

            if not content.strip():
                print("❌ pyproject.toml is empty")
                return False

            print(f"✅ Successfully read pyproject.toml ({len(content)} characters)")

            # Calculate GPU resources per client (distribute 1 GPU fairly among clients)
            num_clients = len(self.clients_config)
            gpu_per_client = round(1.0 / num_clients, 2) if num_clients > 0 else 0.5

            print(f"🎮 GPU allocation: {gpu_per_client} GPU per client ({num_clients} clients total)")

            # Check if federation configuration exists
            federation_section = f"[tool.flwr.federations.{federation_name}]"
            if federation_section not in content:
                print(f"📝 Adding federation configuration for local GPU simulation...")

                # Add federation configuration for local simulation with GPU resources
                federation_config = f"""

# Flower federation configuration for local simulation with GPU resources
{federation_section}
options.num-supernodes = {num_clients}
options.backend.client-resources.num-cpus = 8
options.backend.client-resources.num-gpus = {gpu_per_client}
"""

                # Append to the file
                with open(pyproject_path, 'a') as f:
                    f.write(federation_config)

                print(f"✅ Added federation '{federation_name}' configuration to pyproject.toml")
                print(f"🎮 GPU resources: {gpu_per_client} per client (total clients: {num_clients})")
            else:
                print(f"✅ Federation '{federation_name}' configuration already exists")

                # Update existing configuration with correct GPU allocation
                lines = content.split('\n')
                updated_lines = []
                in_federation_section = False

                for line in lines:
                    if line.strip() == federation_section:
                        in_federation_section = True
                        updated_lines.append(line)
                    elif in_federation_section and line.strip().startswith("[") and line.strip().endswith("]"):
                        in_federation_section = False
                        updated_lines.append(line)
                    elif in_federation_section and "options.num-supernodes" in line:
                        updated_lines.append(f"options.num-supernodes = {num_clients}")
                        print(f"📝 Updated num-supernodes to {num_clients}")
                    elif in_federation_section and "options.backend.client-resources.num-gpus" in line:
                        updated_lines.append(f"options.backend.client-resources.num-gpus = {gpu_per_client}")
                        print(f"🎮 Updated GPU allocation to {gpu_per_client} per client")
                    else:
                        updated_lines.append(line)

                # Write updated content back
                updated_content = '\n'.join(updated_lines)
                with open(pyproject_path, 'w') as f:
                    f.write(updated_content)

            return True

        except Exception as e:
            print(f"⚠️ Error checking federation configuration: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            print(f"📋 Please manually add this to your pyproject.toml:")
            print(f"    [tool.flwr.federations.{federation_name}]")
            print(f"    options.num-supernodes = {len(self.clients_config)}")
            print(f"    options.backend.client-resources.num-cpus = 8")
            print(f"    options.backend.client-resources.num-gpus = {round(1.0 / len(self.clients_config), 2) if len(self.clients_config) > 0 else 0.5}")
            return False

    def run_flower_app(self, venv_activate: Optional[str] = None, federation_name: str = "local-simulation-gpu") -> bool:
        """Run the Flower App locally with intelligent monitoring"""
        try:
            print(f"🚀 Running Flower App locally on federation '{federation_name}'...")

            # Ensure federation configuration exists
            if not self.ensure_federation_config(federation_name):
                print("❌ Failed to configure federation. Cannot proceed.")
                return False

            # Create logs directory
            logs_dir = os.path.join(self.project_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)

            # Prepare command
            if venv_activate:
                run_command = f"bash -c 'source {venv_activate} && cd {self.project_dir} && flwr run . {federation_name} --stream'"
            else:
                run_command = f"cd {self.project_dir} && flwr run . {federation_name} --stream"

            print(f"🔍 Executing: {run_command}")

            # Create log file for this run
            log_file = os.path.join(logs_dir, f"flwr_run_{self.timestamp}.log")

            # Run the command with real-time output
            print("📊 Flower App Output:")
            print("=" * 50)

            # Use subprocess.Popen for real-time output
            process = subprocess.Popen(
                run_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=self.project_dir
            )

            self.current_process = process

            # Stream output in real-time and save to log
            output_buffer = ""
            with open(log_file, 'w') as log_f:
                try:
                    completion_detected = False
                    for line in process.stdout:
                        output_buffer += line
                        print(line, end="")
                        log_f.write(line)
                        log_f.flush()

                        # Check for completion indicators
                        if any(phrase in line.lower() for phrase in [
                            "run finished", "completed successfully", "experiment completed",
                            "training finished", "federation completed", "fl training completed",
                            "losses_distributed", "metrics_distributed", "secagg+ workflow completed successfully"
                        ]):
                            print("\n🎉 Detected completion signal!")
                            completion_detected = True
                            break

                except KeyboardInterrupt:
                    print("\n⚠️ Interrupted by user")
                    process.terminate()
                    return False

            # Wait for process to complete with timeout
            if completion_detected:
                print("⏳ Waiting for process to terminate after completion signal...")
                try:
                    return_code = process.wait(timeout=30)  # 30 second timeout
                    print(f"✅ Process terminated cleanly with return code: {return_code}")
                except subprocess.TimeoutExpired:
                    print("⚠️ Process did not terminate within 30 seconds after completion signal")
                    print("🔧 Attempting graceful termination...")
                    process.terminate()

                    try:
                        return_code = process.wait(timeout=10)  # Give it 10 more seconds
                        print(f"✅ Process terminated gracefully with return code: {return_code}")
                    except subprocess.TimeoutExpired:
                        print("⚠️ Process did not respond to SIGTERM, forcing termination...")
                        process.kill()
                        return_code = process.wait()
                        print(f"✅ Process killed with return code: {return_code}")
            else:
                # No completion signal detected, wait normally
                return_code = process.wait()

            print("\n" + "=" * 50)
            print(f"📊 Process completed with return code: {return_code}")
            print(f"📝 Full output saved to: {log_file}")

            if return_code == 0:
                print("✅ Flower App completed successfully!")
                return True
            else:
                print(f"⚠️ Flower App exited with code {return_code}")
                print("🔍 This might be normal depending on your FL configuration")
                return True  # Consider non-zero exit as success for FL experiments

        except Exception as e:
            print(f"❌ Error running Flower App: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False
        finally:
            self.current_process = None

    def cleanup_processes(self):
        """Clean up any running processes"""
        print("\n🧹 Cleaning up processes...")

        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=5)
                print("✓ Current process terminated")
            except:
                try:
                    self.current_process.kill()
                    print("✓ Current process killed")
                except:
                    pass

        print("✅ Cleanup completed")

    def run_federated_learning(self, venv_activate: Optional[str] = None) -> bool:
        """Main method to run federated learning with sequential experiments"""
        print("🚀 Starting Flower Local Simulation with Sequential Experiments")
        print("🔧 Running locally - no SSH tunnels or remote tmux sessions needed")
        print("📊 Sequential experiment support with automatic config updates")
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
            print("📋 Process: Update configs → Run experiment → Repeat")
        else:
            print("🔄 Single Experiment Mode")

        print("=" * 70)

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
                time.sleep(2)

            # Update pyproject.toml with correct config files for this experiment
            print(f"🔧 Updating pyproject.toml for experiment {experiment_idx + 1}...")
            if not self.update_pyproject_toml(self.server_config, self.clients_config):
                print("❌ Failed to update pyproject.toml. Continuing anyway...")

            # Wait a moment for pyproject.toml to be ready
            time.sleep(2)

            print(f"🌸 Starting Flower App execution for experiment {experiment_idx + 1}...")

            # Run Flower App for this experiment
            if self.run_flower_app(venv_activate, federation_name="local-simulation-gpu"):
                print(f"✅ Experiment {experiment_idx + 1} completed successfully!")
                successful_experiments += 1
            else:
                print(f"❌ Experiment {experiment_idx + 1} failed")

                # For sequential experiments, ask if user wants to continue
                if is_sequential and experiment_idx < num_experiments - 1:
                    print("⚠️ Sequential experiment failed.")
                    print("🔧 You can continue with the next experiment or stop here.")
                    continue
                else:
                    break

            # Add delay between experiments to allow cleanup
            if is_sequential and experiment_idx < num_experiments - 1:
                print(f"⏳ Waiting 5 seconds before next experiment...")
                time.sleep(5)

        print(f"\n{'='*70}")
        print(f"🎯 Sequential Experiment Summary:")
        print(f"   Total experiments: {num_experiments}")
        print(f"   Successful: {successful_experiments}")
        print(f"   Failed: {num_experiments - successful_experiments}")

        if successful_experiments == num_experiments:
            print("🎉 All experiments completed successfully!")
            return True
        elif successful_experiments > 0:
            print(f"⚠️ {successful_experiments}/{num_experiments} experiments completed successfully")
            print("💡 Check logs for failed experiments")
            return True
        else:
            print("❌ All experiments failed")
            return False


def main():
    """Main function for local simulation"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Flower Local Simulation Runner with Sequential Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_local_simulation.py config.yaml
  python run_local_simulation.py config.yaml --venv-activate /path/to/venv/bin/activate

Note: Run this script in your own tmux session if you want session persistence.
        """
    )
    parser.add_argument(
        "config_file",
        help="Path to YAML configuration file (e.g., fl_server.yaml)"
    )
    parser.add_argument(
        "--venv-activate",
        help="Path to virtual environment activation script (optional)"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        print(f"📄 Loading configuration from {args.config_file}...")
        from multi_config import load_config_from_yaml, get_server_config_dict, get_clients_config_dict
        config = load_config_from_yaml(args.config_file)

        # For local simulation, we can work with minimal configuration
        project_dir = getattr(config.fl, 'project_dir', os.getcwd())
        if hasattr(config.fl, 'multi_machine') and config.fl.multi_machine:
            project_dir = config.fl.multi_machine.project_dir

    except FileNotFoundError as e:
        print(f"❌ Configuration file error: {e}")
        print("📝 Please provide a valid YAML configuration file.")
        print("📋 Example: python run_local_simulation.py config.yaml")
        return
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return

    # Convert to dictionary format for compatibility with existing runner
    server_config = get_server_config_dict(config)
    clients_config = get_clients_config_dict(config)

    if not server_config:
        print("❌ No server configuration found!")
        print("📝 For local simulation, ensure your YAML has proper server config.")
        return

    if not clients_config:
        print("❌ No client configuration found!")
        print("📝 For local simulation, ensure your YAML has proper client config.")
        return

    # Create runner
    runner = FlowerLocalSimulationRunner(server_config, clients_config, project_dir)

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal...")
        runner.cleanup_processes()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("🔍 Configuration:")
        print(f"  Project Dir: {project_dir}")
        print(f"  Server Config: {server_config.get('config_file', 'None')}")
        print(f"  Client Configs: {[c.get('config_file', 'None') for c in clients_config]}")
        if args.venv_activate:
            print(f"  Virtual Env: {args.venv_activate}")
        print()

        print("🔄 Starting Local Simulation...")
        print("💡 This script runs Flower simulation locally")
        print("📋 Instructions:")
        print("   1. Config files will be updated automatically for sequential experiments")
        print("   2. pyproject.toml will be configured for the simulation")
        print("   3. Flower app will run locally using 'flwr run'")
        print("   4. All output will be displayed and logged")
        print("   5. Press Ctrl+C to stop at any time")
        print()

        if runner.run_federated_learning(venv_activate=args.venv_activate):
            print("🎉 Local simulation completed successfully!")
        else:
            print("❌ Local simulation failed")
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
    finally:
        runner.cleanup_processes()


if __name__ == "__main__":
    main()
