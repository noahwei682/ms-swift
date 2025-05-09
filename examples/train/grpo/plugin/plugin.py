import asyncio
import re
from typing import List, Dict, Any, Optional, Union

import json
import time

from swift.plugin import ORM, orms
from swift.utils import get_logger
from transformers import TrainerCallback as Callback

# import threading
# from typing import List
# import logging
# from concurrent.futures import ThreadPoolExecutor, TimeoutError
# from multiprocessing import Pool

# logger = logging.getLogger(__name__)

logger = get_logger()

# Add wandb import at the top
import wandb

# Initialize reward dictionary, if cannot import from reward module, use these default values
_default_reward_call_counts = {
    "accuracy": 0,
    "bm25": 0,
    "f1": 0,
    "recall": 0,
    "precision": 0,
    "text_similarity": 0,
    "external_math_acc": 0,
    "external_math_format": 0,
    "external_countdown": 0,
    "external_r1v_acc": 0,
    "external_code_reward": 0,
    "external_code_format": 0
}

_default_latest_rewards = {
    "accuracy_reward": 0.0,
    "bm25_reward": 0.0,
    "f1_reward": 0.0,
    "recall_reward": 0.0,
    "precision_reward": 0.0,
    "text_similarity_reward": 0.0,
    "external_math_acc_reward": 0.0,
    "external_math_format_reward": 0.0,
    "external_countdown_reward": 0.0,
    "external_r1v_acc_reward": 0.0,
    "external_code_reward_reward": 0.0,
    "external_code_format_reward": 0.0
}

# Try to import global variables from reward module, if fails use default values
try:
    # Import before creating references to avoid circular imports
    reward_call_counts = None
    latest_rewards = None
    
    from examples.train.grpo.plugin.reward import reward_call_counts, latest_rewards, print_rewards
    print("Successfully imported reward tracking variables from reward module")
except ImportError:
    # Import failed, use default values
    reward_call_counts = _default_reward_call_counts 
    latest_rewards = _default_latest_rewards
    
    # Define print function
    def print_rewards(step=None):
        """Print current all reward values (fallback version)"""
        global latest_rewards
        if step is None:
            step = 0
        
        rewards_str = " | ".join([f"{name}: {round(value, 4)}" for name, value in latest_rewards.items()])
        print(f"[STEP {step}] ALL REWARDS: {rewards_str}")
        
        # If wandb is running, also record a merged metric
        if wandb.run is not None:
            avg_reward = sum(latest_rewards.values()) / len(latest_rewards) if latest_rewards else 0
            wandb.log({
                "all_rewards_mean": avg_reward,
                "all_rewards_step": step
            }, step=step)
    
    print("Warning: Failed to import reward module, using default reward tracking")

# Update reward weights configuration based on SWIFT documentation
REWARD_WEIGHTS = {
    # Core built-in reward functions
    "accuracy_reward": 1.0,      # Accuracy reward for correct answers
    "format_reward": 0.8,        # Format reward for proper response structure
    "cosine_reward": 0.7,        # Cosine reward for length control
    "repetition_reward": 0.6,    # Repetition penalty
    
    # Additional standard rewards
    "bm25_reward": 0.2,          # BM25 similarity reward
    "f1_reward": 0.5,            # F1 score reward
    "recall_reward": 0.5,        # Recall score reward
    "precision_reward": 0.5,     # Precision score reward
    "sbert_cosine_reward": 0.5,  # SBERT similarity reward
    
    # External rewards
    "external_r1v_acc_reward": 0.8,      # External accuracy reward
    "external_math_acc_reward": 0.8,     # Math accuracy reward
    "external_math_format_reward": 0.6,   # Math format reward
    "external_countdown_reward": 0.8,     # Countdown reward
    "external_code_reward_reward": 0.8,   # Code execution reward
    "external_code_format_reward": 0.6    # Code format reward
}

def apply_reward_weights(rewards_dict):
    """Apply weights to rewards and return weighted values"""
    weighted_rewards = {}
    for name, value in rewards_dict.items():
        weight = REWARD_WEIGHTS.get(name, 1.0)  # Default weight is 1.0 if not specified
        weighted_rewards[name] = value * weight
    return weighted_rewards

def print_all_reward_values_to_terminal(step, rewards_dict):
    """Print all reward values with their weights to terminal"""
    print("\n" + "=" * 100)
    print(f"STEP {step} - REWARD VALUES AND WEIGHTS")
    print("=" * 100)
    
    # Print rewards by type with weights
    built_in_rewards = {}
    standard_rewards = {}
    external_rewards = {}
    
    for name, value in rewards_dict.items():
        if name in ["accuracy_reward", "format_reward", "cosine_reward", "repetition_reward"]:
            built_in_rewards[name] = value
        elif name.startswith("external_"):
            external_rewards[name] = value
        else:
            standard_rewards[name] = value
    
    # Print built-in rewards with weights
    if built_in_rewards:
        print("\nBUILT-IN REWARDS:")
        for name, value in sorted(built_in_rewards.items()):
            weight = REWARD_WEIGHTS.get(name, 1.0)
            weighted_value = value * weight
            print(f"  {name:30} Raw: {value:.4f} | Weight: {weight:.2f} | Weighted: {weighted_value:.4f}")
    
    # Print standard rewards with weights
    if standard_rewards:
        print("\nSTANDARD REWARDS:")
        for name, value in sorted(standard_rewards.items()):
            weight = REWARD_WEIGHTS.get(name, 1.0)
            weighted_value = value * weight
            print(f"  {name:30} Raw: {value:.4f} | Weight: {weight:.2f} | Weighted: {weighted_value:.4f}")
    
    # Print external rewards with weights
    if external_rewards:
        print("\nEXTERNAL REWARDS:")
        for name, value in sorted(external_rewards.items()):
            weight = REWARD_WEIGHTS.get(name, 1.0)
            weighted_value = value * weight
            print(f"  {name:30} Raw: {value:.4f} | Weight: {weight:.2f} | Weighted: {weighted_value:.4f}")
    
    # Print total weighted reward
    total_weighted = sum(value * REWARD_WEIGHTS.get(name, 1.0) for name, value in rewards_dict.items())
    print("\n" + "-" * 100)
    print(f"TOTAL WEIGHTED REWARD: {total_weighted:.4f}")
    print("=" * 100 + "\n")

def force_print_all_rewards(step):
    """Force print all reward values with weights"""
    global latest_rewards
    weighted_rewards = apply_reward_weights(latest_rewards)
    print_all_reward_values_to_terminal(step, latest_rewards)

def sync_rewards_to_wandb(step):
    """Sync rewards to wandb with weights applied"""
    global latest_rewards
    if wandb.run is not None:
        # Apply weights to rewards
        weighted_rewards = apply_reward_weights(latest_rewards)
        
        # Log both raw and weighted rewards
        log_dict = {}
        for name, value in latest_rewards.items():
            weight = REWARD_WEIGHTS.get(name, 1.0)
            weighted_value = value * weight
            log_dict[f"{name}_raw"] = value
            log_dict[f"{name}_weighted"] = weighted_value
            log_dict[f"{name}_weight"] = weight
        
        # Add total weighted reward
        total_weighted = sum(value * REWARD_WEIGHTS.get(name, 1.0) for name, value in latest_rewards.items())
        log_dict["total_weighted_reward"] = total_weighted
        
        wandb.log(log_dict, step=step)
        logger.info(f"Synced weighted rewards to wandb at step {step}")

"""
Step 1: Define a Reward Class
    Implement your custom reward calculation logic within the __call__ method.
    The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

Step 2: Register the Reward Class in orms
    For example:
    python orms['external_math_acc'] = MathAccuracy

Step 3: Configure the Arguments
    Use the following arguments when running the script:
    bash --plugin /path/to/plugin.py --reward_funcs external_math_acc
"""


# WandbCallback for logging loss metrics
class WandbLossCallback(Callback):
    """
    A callback that logs training metrics to wandb.
    
    Usage:
    1. Import this class directly in your training script
    2. Create an instance: callback = WandbLossCallback()
    3. Add it to your trainer: trainer.add_callback(callback)
    """
    
    def __init__(self):
        super().__init__()
        self.global_step = 0
        self.loss_history = []
        self.policy_loss_history = []
        self.value_loss_history = []
        self.kl_loss_history = []
        self.entropy_loss_history = []
        self.reward_history = {}
        
        # Add history records for all reward functions
        global latest_rewards
        for reward_name in latest_rewards.keys():
            self.reward_history[reward_name] = []
        
        # Track whether wandb is initialized
        self._wandb_init = False
        
        # Register self to reward module so reward functions can access
        try:
            from examples.train.grpo.plugin.reward import wandb_callback
            import sys
            # Get current module
            current_module = sys.modules.get('examples.train.grpo.plugin.reward')
            if current_module:
                # Set current callback instance as wandb_callback in reward module
                setattr(current_module, 'wandb_callback', self)
                print("Successfully registered WandbLossCallback to reward module")
            else:
                print("Warning: Could not find reward module to register callback")
        except ImportError:
            print("Warning: Could not import reward module to register callback")
        except Exception as e:
            print(f"Warning: Error registering callback to reward module: {e}")
        
        # Last recorded wandb time
        self.last_wandb_log_time = time.time()
        # Minimum log recording interval (seconds), to avoid recording too frequently
        self.min_log_interval = 2.0
    
    def _ensure_wandb_init(self):
        """Ensure wandb is initialized."""
        if not self._wandb_init:
            import wandb
            if not wandb.run:
                try:
                    wandb.init(project="GRPO-Training", resume="allow")
                    
                    # Add specific configuration information to wandb
                    wandb.config.update({
                        "reward_functions": list(latest_rewards.keys()),
                        "plugin_path": __file__,
                        "timestamp": time.strftime("%Y-%m-%d-%H-%M-%S"),
                    })
                    
                    # Create a dedicated reward panel
                    wandb.run.log_code(".", include_fn=lambda path: path.endswith("plugin.py") or path.endswith("reward.py"))
                    
                except Exception as e:
                    print(f"Error initializing wandb: {e}")
            self._wandb_init = True
    
    def print_all_rewards(self):
        """Print current all reward values to terminal"""
        try:
            # Use global print_rewards function
            print_rewards(self.global_step)
        except Exception as e:
            print(f"Error printing rewards: {e}")
            # Backup plan: Print local variables directly
            global latest_rewards
            rewards_str = " | ".join([f"{name}: {round(value, 4)}" for name, value in latest_rewards.items()])
            print(f"[STEP {self.global_step}] ALL REWARDS (FALLBACK): {rewards_str}")
    
    def on_train_begin(self, trainer, **kwargs):
        """Callback for train begin event"""
        self._ensure_wandb_init()
        self.trainer = trainer
        trainer._last_accuracy_rewards = []
        trainer._last_bm25_rewards = []
        trainer._last_f1_rewards = []
        trainer._last_recall_rewards = []
        trainer._last_precision_rewards = []
        trainer._last_sbert_cosine_rewards = []
        
        # Register callback to reward module
        try:
            from examples.train.grpo.plugin.reward import wandb_callback, sync_all_rewards_to_wandb
            import sys
            current_module = sys.modules.get('examples.train.grpo.plugin.reward')
            if current_module:
                setattr(current_module, 'wandb_callback', self)
                print("Successfully re-registered WandbLossCallback to reward module on train begin")
                
                # Force sync all rewards to wandb
                sync_all_rewards_to_wandb(step=0, force=True)
            else:
                print("Warning: Could not find reward module to register callback on train begin")
        except ImportError:
            print("Warning: Could not import reward module to register callback on train begin")
        except Exception as e:
            print(f"Warning: Error registering callback to reward module on train begin: {e}")
        
        print("Training started. Will log metrics to wandb.")
        self.print_all_rewards()  # Print initial reward values
        
        # Record all rewards values to wandb
        try:
            global latest_rewards
            reward_log = {f"train/initial_{k}": v for k, v in latest_rewards.items()}
            reward_log.update({
                "train/initialized": 1,
                "train/total_steps": trainer.num_steps if hasattr(trainer, "num_steps") else 0,
                "train/reward_functions": len(latest_rewards)
            })
            wandb.log(reward_log, step=0)
            
            # Create reward summary table
            reward_table = wandb.Table(columns=["Reward Name", "Initial Value"])
            for reward_name, reward_value in latest_rewards.items():
                reward_table.add_data(reward_name, float(reward_value))
            wandb.log({"reward_summary": reward_table})
            
            print("Successfully logged all reward values to wandb")
        except Exception as e:
            print(f"Error logging rewards to wandb: {e}")
    
    def on_train_end(self, trainer):
        """Called when training ends."""
        # Record final reward values
        try:
            global latest_rewards
            final_reward_log = {f"train/final_{k}": v for k, v in latest_rewards.items()}
            wandb.log(final_reward_log)
            
            # Create final reward summary table
            reward_table = wandb.Table(columns=["Reward Name", "Final Value"])
            for reward_name, reward_value in latest_rewards.items():
                reward_table.add_data(reward_name, float(reward_value))
            wandb.log({"final_reward_summary": reward_table})
            print("Successfully logged final reward values to wandb")
        except Exception as e:
            print(f"Error logging final rewards to wandb: {e}")
            
        if wandb.run is not None:
            wandb.finish()
    
    def on_step_begin(self, trainer, batch):
        """Called at the beginning of each step."""
        self.global_step += 1
    
    def on_step_end(self, trainer, outputs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of each step.
        
        Args:
            trainer: The trainer instance
            outputs: The outputs from the step, including loss metrics
        """
        if outputs is None:
            outputs = {}
        
        self._ensure_wandb_init()
        
        # Extract loss metrics
        loss = outputs.get("loss")
        policy_loss = outputs.get("policy_loss")
        value_loss = outputs.get("value_loss")
        kl_loss = outputs.get("kl")
        entropy = outputs.get("entropy")
        rewards = outputs.get("rewards", {})
        
        # Log to wandb
        log_dict = {
            "train/step": self.global_step,
        }
        
        # Create a dictionary for terminal print
        terminal_log = {
            "step": self.global_step,
        }
        
        # Handle regular training metrics
        if loss is not None:
            log_dict["train/loss"] = loss
            terminal_log["loss"] = round(float(loss), 4)
            self.loss_history.append(loss)
        
        if policy_loss is not None:
            log_dict["train/policy_loss"] = policy_loss
            terminal_log["policy_loss"] = round(float(policy_loss), 4)
            self.policy_loss_history.append(policy_loss)
        
        if value_loss is not None:
            log_dict["train/value_loss"] = value_loss
            terminal_log["value_loss"] = round(float(value_loss), 4)
            self.value_loss_history.append(value_loss)
        
        if kl_loss is not None:
            log_dict["train/kl_loss"] = kl_loss
            terminal_log["kl_loss"] = round(float(kl_loss), 4)
            self.kl_loss_history.append(kl_loss)
        
        if entropy is not None:
            log_dict["train/entropy"] = entropy
            terminal_log["entropy"] = round(float(entropy), 4)
            self.entropy_loss_history.append(entropy)
        
        # Extract rewards from outputs
        if isinstance(rewards, dict):
            for reward_name, reward_value in rewards.items():
                if reward_name not in self.reward_history:
                    self.reward_history[reward_name] = []
                
                if isinstance(reward_value, (int, float)):
                    self.reward_history[reward_name].append(reward_value)
                    log_dict[f"train/reward_{reward_name}"] = reward_value
                    terminal_log[f"reward_{reward_name}"] = round(float(reward_value), 4)
        elif isinstance(rewards, (int, float, list)):
            # Handle the case where rewards is a single value or list
            reward_value = sum(rewards) / len(rewards) if isinstance(rewards, list) else rewards
            if "default" not in self.reward_history:
                self.reward_history["default"] = []
            
            self.reward_history["default"].append(reward_value)
            log_dict["train/reward"] = reward_value
            terminal_log["reward"] = round(float(reward_value), 4)
        
        # Actively get all reward values from global latest_rewards
        # Ensure latest values are recorded even if reward functions not called in current step
        global latest_rewards
        
        # Sync update latest_rewards and trainer._last_*_rewards
        for reward_name in ["accuracy", "bm25", "f1", "recall", "precision", "text_similarity"]:
            rewards_attr_name = f"_last_{reward_name}_rewards"
            if hasattr(trainer, rewards_attr_name) and getattr(trainer, rewards_attr_name):
                rewards_list = getattr(trainer, rewards_attr_name)
                if rewards_list:
                    avg_reward = sum(rewards_list) / len(rewards_list)
                    latest_rewards[f"{reward_name}_reward"] = avg_reward
        
        # Only add essential reward metrics to wandb log
        # Focus on the specific requested rewards
        target_rewards = [
            "accuracy_reward", 
            "bm25_reward", 
            "f1_reward", 
            "recall_reward", 
            "precision_reward", 
            "text_similarity_reward", 
            "external_r1v_acc_reward"
        ]
        
        # Just log the current values of the target rewards
        for reward_name in target_rewards:
            if reward_name in latest_rewards:
                reward_value = latest_rewards[reward_name]
                # Ensure each reward has an entry in history
                if reward_name not in self.reward_history:
                    self.reward_history[reward_name] = []
                
                # Add reward value to history
                self.reward_history[reward_name].append(reward_value)
                
                # Only record current value, not min/max/std
                log_dict[f"train/{reward_name}"] = reward_value
                terminal_log[reward_name] = round(float(reward_value), 4)
        
        # Log to wandb with unified step - simplified to just the essential metrics
        wandb.log(log_dict, step=self.global_step)
        
        # Enhanced terminal printing of rewards
        print_all_reward_values_to_terminal(self.global_step, latest_rewards)
        
        # Every step prints reward values and loss in terminal (keep this for compatibility)
        logger.info(f"REWARDS & LOSS at step {self.global_step}: {terminal_log}")
        
        # Log additional info to console
        if self.global_step % 10 == 0:
            logger.info(f"Step {self.global_step}: loss={loss}, "
                       f"policy_loss={policy_loss}, "
                       f"value_loss={value_loss}")

# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        global reward_call_counts, latest_rewards
        reward_call_counts["external_math_acc"] += 1
        step_count = reward_call_counts["external_math_acc"]
        
        logger.info(f"External Math Accuracy reward call #{step_count}")
        
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
            
        # Calculate average reward and update latest_rewards
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        latest_rewards["external_math_acc_reward"] = avg_reward
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "external_math_acc_reward": avg_reward,
                "external_math_acc_reward_call": step_count,
                "external_math_acc_rewards_raw": rewards,
                "external_math_acc_reward_min": min(rewards) if rewards else 0,
                "external_math_acc_reward_max": max(rewards) if rewards else 0,
            })
            logger.info(f"LOG to wandb: external_math_acc_reward={avg_reward} at step {step_count}")
        
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        global reward_call_counts, latest_rewards
        reward_call_counts["external_math_format"] += 1
        step_count = reward_call_counts["external_math_format"]
        
        logger.info(f"External Math Format reward call #{step_count}")
        
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        rewards = [1.0 if match else 0.0 for match in matches]
        
        # Calculate average reward and update latest_rewards
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        latest_rewards["external_math_format_reward"] = avg_reward
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "external_math_format_reward": avg_reward,
                "external_math_format_reward_call": step_count,
                "external_math_format_rewards_raw": rewards,
                "external_math_format_reward_min": min(rewards) if rewards else 0,
                "external_math_format_reward_max": max(rewards) if rewards else 0,
            })
            logger.info(f"LOG to wandb: external_math_format_reward={avg_reward} at step {step_count}")
        
        return rewards


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        global reward_call_counts, latest_rewards
        reward_call_counts["external_countdown"] += 1
        step_count = reward_call_counts["external_countdown"]
        
        logger.info(f"External Countdown reward call #{step_count}")
        
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
                
        # Calculate average reward and update latest_rewards
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        latest_rewards["external_countdown_reward"] = avg_reward
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "external_countdown_reward": avg_reward,
                "external_countdown_reward_call": step_count,
                "external_countdown_rewards_raw": rewards,
                "external_countdown_reward_min": min(rewards) if rewards else 0,
                "external_countdown_reward_max": max(rewards) if rewards else 0,
            })
            logger.info(f"LOG to wandb: external_countdown_reward={avg_reward} at step {step_count}")
        
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        global reward_call_counts, latest_rewards
        reward_call_counts["external_r1v_acc"] += 1
        step_count = reward_call_counts["external_r1v_acc"]
        
        logger.info(f"External R1V Accuracy reward call #{step_count}")
        
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
            
        # Calculate average reward and update latest_rewards
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        latest_rewards["external_r1v_acc_reward"] = avg_reward
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "external_r1v_acc_reward": avg_reward,
                "external_r1v_acc_reward_call": step_count,
                "external_r1v_acc_rewards_raw": rewards,
                "external_r1v_acc_reward_min": min(rewards) if rewards else 0,
                "external_r1v_acc_reward_max": max(rewards) if rewards else 0,
            })
            logger.info(f"LOG to wandb: external_r1v_acc_reward={avg_reward} at step {step_count}")
        
        return rewards


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        global reward_call_counts, latest_rewards
        reward_call_counts["external_code_reward"] += 1
        step_count = reward_call_counts["external_code_reward"]
        
        logger.info(f"External Code Reward call #{step_count}")
        
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)
            
        # Calculate average reward and update latest_rewards
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        latest_rewards["external_code_reward_reward"] = avg_reward
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "external_code_reward_reward": avg_reward,
                "external_code_reward_reward_call": step_count,
                "external_code_reward_rewards_raw": rewards,
                "external_code_reward_reward_min": min(rewards) if rewards else 0,
                "external_code_reward_reward_max": max(rewards) if rewards else 0,
            })
            logger.info(f"LOG to wandb: external_code_reward_reward={avg_reward} at step {step_count}")
  
        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        global reward_call_counts, latest_rewards
        reward_call_counts["external_code_format"] += 1
        step_count = reward_call_counts["external_code_format"]
        
        logger.info(f"External Code Format reward call #{step_count}")
        
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
            
        # Calculate average reward and update latest_rewards
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        latest_rewards["external_code_format_reward"] = avg_reward
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "external_code_format_reward": avg_reward,
                "external_code_format_reward_call": step_count,
                "external_code_format_rewards_raw": rewards,
                "external_code_format_reward_min": min(rewards) if rewards else 0,
                "external_code_format_reward_max": max(rewards) if rewards else 0,
            })
            logger.info(f"LOG to wandb: external_code_format_reward={avg_reward} at step {step_count}")
        
        return rewards


class AccuracyReward(ORM):
    
    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion is the same as the ground truth."""
        global reward_call_counts, latest_rewards
        reward_call_counts["accuracy"] += 1
        step_count = reward_call_counts["accuracy"]
        
        logger.info(f"Accuracy reward call #{step_count}")
        
        from math_verify import LatexExtractionConfig, parse, verify
        from sympy import sympify, Symbol, solve, Eq
        import re
        
        # Get solutions from solution key directly
        if "solution" not in kwargs:
            logger.warning(f"No solutions found in kwargs: {list(kwargs.keys())}")
            return [0.0] * len(completions)
        solutions = kwargs["solution"]
        
        # Robust content extraction
        completion_contents = []
        for completion in completions:
            try:
                if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                    content = completion[0].get("content", "")
                elif isinstance(completion, dict):
                    content = completion.get("content", "")
                else:
                    content = str(completion)
                completion_contents.append(content)
            except Exception as e:
                logger.warning(f"Error extracting content from completion: {e}")
                completion_contents.append("")
        
        # Ensure solutions and completions have matching lengths
        if len(solutions) != len(completion_contents):
            logger.warning(f"Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
            min_len = min(len(solutions), len(completion_contents))
            solutions = solutions[:min_len]
            completion_contents = completion_contents[:min_len]
        
        def extract_math_expression(text):
            """Extract mathematical expression from text."""
            # Try to find expressions in LaTeX format
            latex_pattern = r'\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]'
            matches = re.findall(latex_pattern, text)
            if matches:
                return matches[0].strip('$()[]\\')
            
            # Try to find boxed answers
            boxed_pattern = r'\\boxed\{.*?\}'
            matches = re.findall(boxed_pattern, text)
            if matches:
                return matches[0].replace('\\boxed{', '').replace('}', '')
            
            return text.strip()
        
        def safe_verify(expr1, expr2):
            """Safely verify two mathematical expressions."""
            try:
                # First try direct verification
                return float(verify(expr1, expr2))
            except Exception as e:
                try:
                    # If direct verification fails, try symbolic comparison
                    sympy_expr1 = sympify(expr1)
                    sympy_expr2 = sympify(expr2)
                    
                    # If both expressions are equations
                    if isinstance(sympy_expr1, Eq) and isinstance(sympy_expr2, Eq):
                        # Solve both equations and compare solutions
                        var = Symbol('x')
                        sol1 = solve(sympy_expr1, var)
                        sol2 = solve(sympy_expr2, var)
                        return float(len(set(sol1) & set(sol2)) > 0)
                    
                    # If expressions are not equations, compare them directly
                    return float(sympy_expr1.equals(sympy_expr2))
                except Exception as e2:
                    logger.warning(f"Error in symbolic comparison: {str(e2)[:200]}")
                    return 0.0
        
        # Calculate rewards
        rewards = []
        for content, solution in zip(completion_contents, solutions):
            try:
                # Extract mathematical expressions
                content_expr = extract_math_expression(content)
                solution_expr = extract_math_expression(solution)
                
                # Parse expressions
                try:
                    gold_parsed = parse(solution_expr, extraction_mode="first_match", 
                                     extraction_config=[LatexExtractionConfig()])
                    answer_parsed = parse(content_expr, extraction_mode="first_match", 
                                       extraction_config=[LatexExtractionConfig()])
                    
                    if len(gold_parsed) != 0:
                        reward = safe_verify(answer_parsed, gold_parsed)
                    else:
                        # If parsing fails, try direct string comparison
                        reward = float(content_expr.strip() == solution_expr.strip())
                except Exception as e:
                    logger.warning(f"Error in expression parsing: {str(e)[:200]}")
                    # Fallback to direct string comparison
                    reward = float(content_expr.strip() == solution_expr.strip())
                
                rewards.append(reward)
            except Exception as e:
                logger.warning(f"Error in accuracy_reward: {str(e)[:200]}")
                rewards.append(0.0)
        
        # Calculate average reward and update latest_rewards
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        latest_rewards["accuracy_reward"] = avg_reward
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "accuracy_reward": avg_reward,
                "accuracy_reward_call": step_count,
                "accuracy_rewards_raw": rewards,
                "accuracy_reward_min": min(rewards) if rewards else 0,
                "accuracy_reward_max": max(rewards) if rewards else 0,
            })
            logger.info(f"LOG to wandb: accuracy_reward={avg_reward} at step {step_count}")
        
        return rewards


class BM25Reward(ORM):
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that computes BM25 scores between model completion and solution"""
        global reward_call_counts, latest_rewards
        reward_call_counts["bm25"] += 1
        step_count = reward_call_counts["bm25"]
        
        logger.info(f"BM25 reward call #{step_count}")
        
        import math
        
        # Get solutions from solution key directly
        if "solution" not in kwargs:
            logger.warning(f"No solutions found in kwargs: {list(kwargs.keys())}")
            return [0.0] * len(completions)
        solutions = kwargs["solution"]
        
        # Robust content extraction
        completion_contents = []
        for completion in completions:
            try:
                if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                    content = completion[0].get("content", "")
                elif isinstance(completion, dict):
                    content = completion.get("content", "")
                else:
                    content = str(completion)
                completion_contents.append(content)
            except Exception as e:
                logger.warning(f"Error extracting content from completion: {e}")
                completion_contents.append("")
        
        # Ensure solutions and completions have matching lengths
        if len(solutions) != len(completion_contents):
            logger.warning(f"Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
            # Truncate the longer list to match the shorter list
            min_len = min(len(solutions), len(completion_contents))
            solutions = solutions[:min_len]
            completion_contents = completion_contents[:min_len]
        
        # Create corpus for BM25 calculation
        corpus = solutions + completion_contents
        
        # Calculate BM25 scores
        rewards = []
        k1 = 1.2  # Parameter for BM25
        b = 0.75  # Parameter for BM25
        
        # Calculate average document length for BM25
        N = len(corpus)
        avgdl = sum(len(doc.split()) for doc in corpus) / N if N > 0 else 0
        
        # Helper function to calculate IDF
        def idf(term):
            n_t = sum(1 for doc in corpus if term in doc.split())
            return math.log((N - n_t + 0.5) / (n_t + 0.5) + 1) if n_t > 0 else 0
        
        # Calculate BM25 for each completion-solution pair
        for content, solution in zip(completion_contents, solutions):
            try:
                # Use solution as query and content as document
                query_terms = solution.split()
                doc_terms = content.split()
                doc_len = len(doc_terms)
                
                # Calculate BM25 score
                bm25_score = 0
                for term in query_terms:
                    f_td = doc_terms.count(term)  # Term frequency
                    idf_t = idf(term)
                    if f_td > 0 and idf_t > 0:
                        numerator = f_td * (k1 + 1)
                        denominator = f_td + k1 * (1 - b + b * (doc_len / avgdl)) if avgdl > 0 else 1
                        bm25_score += idf_t * (numerator / denominator)
                
                # Normalize BM25 score to a 0-1 range
                # Assuming a maximum possible score of 10 for normalization
                normalized_score = min(bm25_score / 10.0, 1.0)
                rewards.append(normalized_score)
                
            except Exception as e:
                logger.warning(f"Error in BM25 calculation: {e}")
                rewards.append(0.0)
        
        # Calculate average reward and update latest_rewards
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        latest_rewards["bm25_reward"] = avg_reward
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "bm25_reward": avg_reward,
                "bm25_reward_call": step_count,
                "bm25_rewards_raw": rewards,
                "bm25_reward_min": min(rewards) if rewards else 0,
                "bm25_reward_max": max(rewards) if rewards else 0,
            })
            logger.info(f"LOG to wandb: bm25_reward={avg_reward} at step {step_count}")
        
        return rewards


class F1Reward(ORM):
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that computes F1 scores between model completion and solution"""
        global reward_call_counts, latest_rewards
        reward_call_counts["f1"] += 1
        step_count = reward_call_counts["f1"]
        
        logger.info(f"F1 reward call #{step_count}")
        
        # Get solutions from solution key directly
        if "solution" not in kwargs:
            logger.warning(f"No solutions found in kwargs: {list(kwargs.keys())}")
            return [0.0] * len(completions)
        solutions = kwargs["solution"]
        
        # Robust content extraction
        completion_contents = []
        for completion in completions:
            try:
                if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                    content = completion[0].get("content", "")
                elif isinstance(completion, dict):
                    content = completion.get("content", "")
                else:
                    content = str(completion)
                completion_contents.append(content)
            except Exception as e:
                logger.warning(f"Error extracting content from completion: {e}")
                completion_contents.append("")
        
        # Ensure solutions and completions have matching lengths
        if len(solutions) != len(completion_contents):
            logger.warning(f"Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
            # Truncate the longer list to match the shorter list
            min_len = min(len(solutions), len(completion_contents))
            solutions = solutions[:min_len]
            completion_contents = completion_contents[:min_len]
        
        # Calculate F1 scores
        rewards = []
        for content, solution in zip(completion_contents, solutions):
            try:
                # Tokenize content and solution (simple splitting by whitespace)
                content_tokens = set(content.lower().split())
                solution_tokens = set(solution.lower().split())
                
                # Calculate precision, recall, and F1 score
                if not solution_tokens or not content_tokens:
                    # If either set is empty, we can't compute a meaningful F1 score
                    rewards.append(0.0)
                    continue
                    
                # Find common tokens (intersection)
                common_tokens = content_tokens.intersection(solution_tokens)
                
                # Calculate precision: common / generated
                precision = len(common_tokens) / len(content_tokens) if content_tokens else 0
                
                # Calculate recall: common / reference
                recall = len(common_tokens) / len(solution_tokens) if solution_tokens else 0
                
                # Calculate F1 score: harmonic mean of precision and recall
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
                    
                rewards.append(f1)
                
            except Exception as e:
                logger.warning(f"Error in F1 calculation: {e}")
                rewards.append(0.0)
        
        # Calculate average reward and update latest_rewards
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        latest_rewards["f1_reward"] = avg_reward
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "f1_reward": avg_reward,
                "f1_reward_call": step_count,
                "f1_rewards_raw": rewards,
                "f1_reward_min": min(rewards) if rewards else 0,
                "f1_reward_max": max(rewards) if rewards else 0,
            })
            logger.info(f"LOG to wandb: f1_reward={avg_reward} at step {step_count}")
        
        return rewards


class RecallReward(ORM):
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that computes recall score between model completion and solution"""
        global reward_call_counts, latest_rewards
        reward_call_counts["recall"] += 1
        step_count = reward_call_counts["recall"]
        
        logger.info(f"Recall reward call #{step_count}")
        
        # Get solutions from solution key directly
        if "solution" not in kwargs:
            logger.warning(f"No solutions found in kwargs: {list(kwargs.keys())}")
            return [0.0] * len(completions)
        solutions = kwargs["solution"]
        
        # Robust content extraction
        completion_contents = []
        for completion in completions:
            try:
                if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                    content = completion[0].get("content", "")
                elif isinstance(completion, dict):
                    content = completion.get("content", "")
                else:
                    content = str(completion)
                completion_contents.append(content)
            except Exception as e:
                logger.warning(f"Error extracting content from completion: {e}")
                completion_contents.append("")
        
        # Ensure solutions and completions have matching lengths
        if len(solutions) != len(completion_contents):
            logger.warning(f"Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
            # Truncate the longer list to match the shorter list
            min_len = min(len(solutions), len(completion_contents))
            solutions = solutions[:min_len]
            completion_contents = completion_contents[:min_len]
        
        # Calculate Recall scores
        rewards = []
        for content, solution in zip(completion_contents, solutions):
            try:
                # Tokenize content and solution (simple splitting by whitespace)
                content_tokens = set(content.lower().split())
                solution_tokens = set(solution.lower().split())
                
                # Calculate recall score
                if not solution_tokens:
                    # If solution is empty, recall is meaningless
                    rewards.append(0.0)
                    continue
                    
                # Find common tokens (intersection)
                common_tokens = content_tokens.intersection(solution_tokens)
                
                # Calculate recall: common / reference
                recall = len(common_tokens) / len(solution_tokens)
                
                rewards.append(recall)
                
            except Exception as e:
                logger.warning(f"Error in Recall calculation: {e}")
                rewards.append(0.0)
        
        # Calculate average reward and update latest_rewards
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        latest_rewards["recall_reward"] = avg_reward
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "recall_reward": avg_reward,
                "recall_reward_call": step_count,
                "recall_rewards_raw": rewards,
                "recall_reward_min": min(rewards) if rewards else 0,
                "recall_reward_max": max(rewards) if rewards else 0,
            })
            logger.info(f"LOG to wandb: recall_reward={avg_reward} at step {step_count}")
        
        return rewards


class PrecisionReward(ORM):
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that computes precision score between model completion and solution"""
        global reward_call_counts, latest_rewards
        reward_call_counts["precision"] += 1
        step_count = reward_call_counts["precision"]
        
        logger.info(f"Precision reward call #{step_count}")
        
        # Get solutions from solution key directly
        if "solution" not in kwargs:
            logger.warning(f"No solutions found in kwargs: {list(kwargs.keys())}")
            return [0.0] * len(completions)
        solutions = kwargs["solution"]
        
        # Robust content extraction
        completion_contents = []
        for completion in completions:
            try:
                if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                    content = completion[0].get("content", "")
                elif isinstance(completion, dict):
                    content = completion.get("content", "")
                else:
                    content = str(completion)
                completion_contents.append(content)
            except Exception as e:
                logger.warning(f"Error extracting content from completion: {e}")
                completion_contents.append("")
        
        # Ensure solutions and completions have matching lengths
        if len(solutions) != len(completion_contents):
            logger.warning(f"Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
            # Truncate the longer list to match the shorter list
            min_len = min(len(solutions), len(completion_contents))
            solutions = solutions[:min_len]
            completion_contents = completion_contents[:min_len]
        
        # Calculate Precision scores
        rewards = []
        for content, solution in zip(completion_contents, solutions):
            try:
                # Tokenize content and solution (simple splitting by whitespace)
                content_tokens = set(content.lower().split())
                solution_tokens = set(solution.lower().split())
                
                # Calculate precision score
                if not content_tokens:
                    # If completion is empty, precision is meaningless
                    rewards.append(0.0)
                    continue
                    
                # Find common tokens (intersection)
                common_tokens = content_tokens.intersection(solution_tokens)
                
                # Calculate precision: common / generated
                precision = len(common_tokens) / len(content_tokens)
                
                rewards.append(precision)
                
            except Exception as e:
                logger.warning(f"Error in Precision calculation: {e}")
                rewards.append(0.0)
        
        # Calculate average reward and update latest_rewards
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        latest_rewards["precision_reward"] = avg_reward
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "precision_reward": avg_reward,
                "precision_reward_call": step_count,
                "precision_rewards_raw": rewards,
                "precision_reward_min": min(rewards) if rewards else 0,
                "precision_reward_max": max(rewards) if rewards else 0,
            })
            logger.info(f"LOG to wandb: precision_reward={avg_reward} at step {step_count}")
        
        return rewards
    
import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from typing import List

class TextSimilarityReward(ORM):
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self._wandb_init = False
        self._wandb_retry_count = 0
        self._max_retries = 3
        self._wandb_metrics_buffer = []
        
    def _ensure_wandb_init(self):
        """确保wandb已初始化，包含重试机制"""
        if self._wandb_init:
            return True
            
        if self._wandb_retry_count >= self._max_retries:
            logger.warning("Max wandb initialization retries reached. Continuing without wandb logging.")
            return False
            
        try:
            if wandb.run is None:
                # 设置更长的超时时间
                wandb.init(
                    project="GRPO-Training",
                    resume="allow",
                    settings=wandb.Settings(
                        start_method="thread",
                        timeout=60,
                        _disable_stats=True
                    )
                )
            self._wandb_init = True
            logger.info("Wandb initialized for TextSimilarityReward")
            return True
        except Exception as e:
            self._wandb_retry_count += 1
            logger.warning(f"Failed to initialize wandb (attempt {self._wandb_retry_count}/{self._max_retries}): {str(e)[:200]}")
            return False
            
    def _safe_wandb_log(self, metrics, step):
        """安全地记录wandb指标，包含缓冲机制"""
        if not self._ensure_wandb_init():
            # 如果初始化失败，将指标添加到缓冲区
            self._wandb_metrics_buffer.append((metrics, step))
            return
            
        try:
            wandb.log(metrics, step=step, commit=True)
        except Exception as e:
            logger.warning(f"Error logging to wandb: {str(e)[:200]}")
            # 将失败的指标添加到缓冲区
            self._wandb_metrics_buffer.append((metrics, step))
            
    def _flush_wandb_buffer(self):
        """尝试刷新wandb缓冲区"""
        if not self._wandb_metrics_buffer:
            return
            
        if not self._ensure_wandb_init():
            return
            
        remaining_metrics = []
        for metrics, step in self._wandb_metrics_buffer:
            try:
                wandb.log(metrics, step=step, commit=True)
            except Exception as e:
                logger.warning(f"Error flushing buffered metrics to wandb: {str(e)[:200]}")
                remaining_metrics.append((metrics, step))
                
        self._wandb_metrics_buffer = remaining_metrics
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that computes text similarity between completions and solutions"""
        global reward_call_counts, latest_rewards
        reward_call_counts["text_similarity"] += 1
        step_count = reward_call_counts["text_similarity"]
        
        logger.info(f"Text Similarity reward call #{step_count}")
        
        # 默认返回全零数组
        default_return = [0.0] * len(completions)
        
        if not completions or "solution" not in kwargs:
            return default_return
        
        solutions = kwargs["solution"]
        if not solutions:
            return default_return
        
        # 文本提取函数
        def extract_text(text):
            try:
                if isinstance(text, list) and text and isinstance(text[0], dict):
                    return text[0].get("content", "").strip() or " "
                elif isinstance(text, dict):
                    return text.get("content", "").strip() or " "
                return str(text).strip() or " "
            except:
                return " "
        
        try:
            # 处理文本
            completion_texts = [extract_text(c) for c in completions]
            solution_texts = [extract_text(s) for s in solutions]
            min_len = min(len(completion_texts), len(solution_texts))
            
            if min_len == 0:
                return default_return
            
            # 确保输入是列表且不为空
            if not completion_texts[:min_len] or not solution_texts[:min_len]:
                return default_return
            
            try:
                # 使用TF-IDF向量化文本
                all_texts = completion_texts[:min_len] + solution_texts[:min_len]
                tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                
                # 分离completion和solution的向量
                completion_vectors = tfidf_matrix[:min_len]
                solution_vectors = tfidf_matrix[min_len:]
                
                # 计算余弦相似度
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(completion_vectors, solution_vectors)
                
                # 提取对角线上的相似度值
                rewards = [float(similarity_matrix[i, i]) for i in range(min_len)]
                
                # 如果长度不足，补充0
                if len(rewards) < len(completions):
                    rewards.extend([0.0] * (len(completions) - len(rewards)))
                
                # 计算统计信息
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                min_reward = min(rewards) if rewards else 0
                max_reward = max(rewards) if rewards else 0
                std_reward = (sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 0
                
                # 更新全局奖励值
                latest_rewards["text_similarity_reward"] = avg_reward
                
                # 准备wandb指标
                base_metrics = {
                    "text_similarity_reward": avg_reward,
                    "text_similarity_reward_step": step_count,
                    "text_similarity_rewards_raw": rewards,
                    "text_similarity_reward_min": min_reward,
                    "text_similarity_reward_max": max_reward,
                    "text_similarity_reward_std": std_reward,
                }
                
                stats_metrics = {
                    "text_similarity_stats/mean": avg_reward,
                    "text_similarity_stats/std": std_reward,
                    "text_similarity_stats/min": min_reward,
                    "text_similarity_stats/max": max_reward,
                    "text_similarity_stats/median": sorted(rewards)[len(rewards)//2] if rewards else 0,
                    "text_similarity_stats/num_samples": len(rewards),
                }
                
                # 安全地记录到wandb
                self._safe_wandb_log(base_metrics, step_count)
                self._safe_wandb_log(stats_metrics, step_count)
                
                # 记录示例
                if rewards:
                    for i, (comp, sol, reward) in enumerate(zip(completion_texts[:3], solution_texts[:3], rewards[:3])):
                        example_metrics = {
                            f"text_similarity_examples/example_{i}/completion": comp,
                            f"text_similarity_examples/example_{i}/solution": sol,
                            f"text_similarity_examples/example_{i}/similarity": reward,
                        }
                        self._safe_wandb_log(example_metrics, step_count)
                
                # 尝试刷新缓冲区
                self._flush_wandb_buffer()
                
                logger.info(f"LOG to wandb: text_similarity_reward={avg_reward:.4f} (min={min_reward:.4f}, max={max_reward:.4f}, std={std_reward:.4f}) at step {step_count}")
                
                return rewards
                
            except Exception as e:
                logger.warning(f"Error in text similarity calculation: {str(e)[:200]}")
                return default_return
                
        except Exception as e:
            logger.warning(f"Text similarity reward error: {str(e)[:200]}")
            return default_return

orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_code_format'] = CodeFormat
orms['accuracy_reward'] = AccuracyReward
orms['bm25_reward'] = BM25Reward
orms['f1_reward'] = F1Reward
orms['recall_reward'] = RecallReward
orms['precision_reward'] = PrecisionReward
orms['text_similarity_reward'] = TextSimilarityReward

# Add a new function at the top of the file
def force_print_all_rewards(step):
    """Force print all reward values, regardless of whether they are calculated in the current step"""
    global latest_rewards
    # Print more obvious separator line
    print("\n" + "=" * 80)
    print(f"[Force Print] Step {step} All REWARD Values:")
    
    # Print rewards by type
    external_rewards = {}
    standard_rewards = {}
    
    for name, value in latest_rewards.items():
        if name.startswith("external_"):
            external_rewards[name] = value
        else:
            standard_rewards[name] = value
    
    # Print standard rewards
    if standard_rewards:
        print("Standard REWARD:")
        for name, value in standard_rewards.items():
            print(f"  - {name}: {round(float(value), 6)}")
    
    # Print external rewards
    if external_rewards:
        print("External REWARD:")
        for name, value in external_rewards.items():
            print(f"  - {name}: {round(float(value), 6)}")
    
    print("=" * 80 + "\n")
    
    # Try sync to wandb - simplified to only upload essential metrics
    try:
        if wandb.run is not None:
            # Only sync essential reward metrics
            reward_log = {}
            target_rewards = [
                "accuracy_reward", 
                "bm25_reward", 
                "f1_reward", 
                "recall_reward", 
                "precision_reward", 
                "text_similarity_reward", 
                "external_r1v_acc_reward"
            ]
            
            # Add only target rewards to log
            for reward_name in target_rewards:
                if reward_name in latest_rewards:
                    reward_log[reward_name] = latest_rewards[reward_name]
            
            # Upload to wandb
            wandb.log(reward_log, step=step)
            print(f"Synced essential rewards to wandb at step {step}")
    except Exception as e:
        print(f"Could not sync rewards to wandb: {e}")

# Add a function that can be called from command line to check rewards
def check_rewards():
    """
    Function to check current reward status in command line
    
    Usage:
    1. Import this function in training script: 
       from examples.train.grpo.plugin.plugin import check_rewards
    
    2. Call anywhere needed:
       check_rewards()
    
    3. Or in training running, in Python interactive shell:
       import sys
       sys.path.append('/path/to/ms-swift')
       from examples.train.grpo.plugin.plugin import check_rewards
       check_rewards()
    """
    print("\n" + "#" * 100)
    print("## Current REWARD Status Check")
    print("#" * 100)
    
    # Print reward call counts
    print("\n## Reward Call Counts:")
    for reward_name, count in reward_call_counts.items():
        print(f"  - {reward_name}: {count} calls")
    
    # Print latest reward values
    print("\n## Latest Reward Values:")
    standard_rewards = {}
    external_rewards = {}
    
    for name, value in latest_rewards.items():
        if name.startswith("external_"):
            external_rewards[name] = value
        else:
            standard_rewards[name] = value
    
    if standard_rewards:
        print("\n### Standard REWARD:")
        for name, value in standard_rewards.items():
            print(f"  - {name}: {round(float(value), 6)}")
    
    if external_rewards:
        print("\n### External REWARD:")
        for name, value in external_rewards.items():
            print(f"  - {name}: {round(float(value), 6)}")
    
    # If wandb is running, print wandb information
    try:
        if wandb.run:
            print(f"\n## Wandb Running Information:")
            print(f"  - Run ID: {wandb.run.id}")
            print(f"  - Project: {wandb.run.project}")
            print(f"  - Entity: {wandb.run.entity}")
            print(f"  - Path: {wandb.run.path}")
            print(f"  - URL: {wandb.run.get_url()}")
    except:
        print("\n## Wandb Not Running")
    
    print("\n" + "#" * 100)
    print("## Check Completed")
    print("#" * 100 + "\n")
    
    # Sync current reward values to wandb
    try:
        from examples.train.grpo.plugin.reward import sync_all_rewards_to_wandb
        sync_all_rewards_to_wandb(force=True)
        print("All rewards synced to wandb")
    except Exception as e:
        print(f"Sync rewards to wandb failed: {e}")
    
    return {
        "call_counts": reward_call_counts,
        "latest_values": latest_rewards
    }

# Manual sync all rewards to wandb function
def sync_rewards_to_wandb(step=None):
    """
    Manual trigger to sync all rewards to wandb
    
    Args:
        step: Specify step number, if not specified use current highest reward call count
        
    Usage:
        1. Import and call in training script:
           from examples.train.grpo.plugin.plugin import sync_rewards_to_wandb
           sync_rewards_to_wandb()
           
        2. Call from command line:
           import sys
           sys.path.append('/path/to/ms-swift')
           from examples.train.grpo.plugin.plugin import sync_rewards_to_wandb
           sync_rewards_to_wandb()
    """
    try:
        # If step not provided, use the max of all call counts
        if step is None:
            step = max(reward_call_counts.values()) if reward_call_counts else 0
        
        # Only sync essential reward metrics
        reward_log = {}
        target_rewards = [
            "accuracy_reward", 
            "bm25_reward", 
            "f1_reward", 
            "recall_reward", 
            "precision_reward", 
            "text_similarity_reward", 
            "external_r1v_acc_reward"
        ]
        
        # Add only target rewards to log
        for reward_name in target_rewards:
            if reward_name in latest_rewards:
                reward_log[reward_name] = latest_rewards[reward_name]
        
        # Upload to wandb
        if wandb.run is not None:
            wandb.log(reward_log, step=step)
            print(f"Manual trigger synced essential rewards to wandb, step={step}")
            return True
        else:
            print("No active wandb run found")
            return False
    except Exception as e:
        print(f"Sync rewards to wandb failed: {e}")
        return False

"""
Usage:

1. Reward functions:
   - All reward functions are automatically called and recorded during training
   - You can specify reward functions to use with --reward_funcs parameter

2. How to check all reward values:
   - Check in training logs, printed after each step
   - Check in Wandb dashboard
   - Run the following code to check current reward status:
     ```python
     from examples.train.grpo.plugin.plugin import check_rewards
     check_rewards()
     ```
   - Force sync latest rewards to wandb:
     ```python
     from examples.train.grpo.plugin.plugin import sync_rewards_to_wandb
     sync_rewards_to_wandb()
     ```

3. Why some reward values show as 0:
   - If a reward function is not specified in command line parameter, it won't be called
   - Ensure to add --reward_funcs parameter at training start and specify all needed reward functions
   - For example: --reward_funcs accuracy_reward bm25_reward f1_reward
"""

# Add example launch script at the end of the file
"""
Example: How to launch training using all reward functions

```bash
# Basic command
python -m examples.train.grpo.train \
  --plugin /Users/weiwei/Downloads/ms-swift-main\ 3/examples/train/grpo/plugin/plugin.py \
  --reward_funcs accuracy_reward bm25_reward f1_reward recall_reward precision_reward text_similarity_reward external_r1v_acc \
  # Other necessary parameters
  --model_type bloom \
  --model_id bigscience/bloom-1b7 \
  # ... Other training parameters

# Full command example - Use space to separate multiple reward functions
python -m examples.train.grpo.train \
  --plugin /Users/weiwei/Downloads/ms-swift-main\ 3/examples/train/grpo/plugin/plugin.py \
  --reward_funcs accuracy_reward bm25_reward f1_reward recall_reward precision_reward text_similarity_reward external_r1v_acc external_math_acc external_math_format external_countdown external_code_reward external_code_format \
  --model_type bloom \
  --model_id bigscience/bloom-1b7 \
  --adapter_name lora \
  --lora_target_modules all \
  --train_data alpaca.json \
  --dataset_concatenation \
  --per_device_train_batch_size 8 \
  --per_device_mini_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --fp16 \
  --do_train \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --adam_epsilon 1e-6 \
  --max_grad_norm 1.0 \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_strategy epoch \
  --output_dir ./output/grpo-bloom-1b7 \
  --use_wandb
```

Notes:
1. Important: reward_funcs parameter uses space to separate multiple functions, not comma or multiple times
2. Wrong example: `--reward_funcs accuracy_reward,bm25_reward` (This will cause error, don't use comma)
3. Wrong example: `--reward_funcs accuracy_reward --reward_funcs bm25_reward` (This will cause error, only last one will work)
4. Correct example: `--reward_funcs accuracy_reward bm25_reward f1_reward` (Use space to separate)
5. If adding custom reward function, ensure to register it in orms dictionary
6. To print current reward status, use check_rewards() function
7. For reward functions with special dependencies, ensure related dependencies are installed (like sentence-transformers, math_verify, etc.)
"""
