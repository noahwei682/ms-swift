import asyncio
import re
from typing import List

import json

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()
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


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
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
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


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

        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


class AccuracyReward(ORM):
    
    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion is the same as the ground truth."""
        from math_verify import LatexExtractionConfig, parse, verify
        
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
        
        # Calculate rewards
        rewards = []
        for content, solution in zip(completion_contents, solutions):
            try:
                gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
                answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
                
                if len(gold_parsed) != 0:
                    try:
                        reward = float(verify(answer_parsed, gold_parsed))
                        rewards.append(reward)
                    except Exception as e:
                        logger.warning(f"Verification error: {e}")
                        rewards.append(0.0)
                else:
                    rewards.append(1.0)
            except Exception as e:
                logger.warning(f"Error in accuracy_reward: {e}")
                rewards.append(0.0)
        
        return rewards


class BM25Reward(ORM):
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that computes BM25 scores between model completion and solution"""
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
        
        return rewards


class F1Reward(ORM):
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that computes F1 scores between model completion and solution"""
        
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
        
        return rewards


class RecallReward(ORM):
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that computes recall score between model completion and solution"""
        
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
        
        return rewards


class PrecisionReward(ORM):
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that computes precision score between model completion and solution"""
        
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
        
        return rewards


class SBERTCosineReward(ORM):
    
    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('sentence_transformers') is not None, (
            "The sentence_transformers package is required but not installed. "
            "Please install it using 'pip install sentence-transformers'.")
        assert importlib.util.find_spec('sklearn') is not None, (
            "The scikit-learn package is required but not installed. "
            "Please install it using 'pip install scikit-learn'.")
        self.model = None
    
    def load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that computes cosine similarity between sentence embeddings of completions and solutions"""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Check if SBERT model is available
        try:
            model = self.load_model()
        except Exception as e:
            logger.warning(f"Failed to load Sentence-BERT model: {e}")
            return [0.0] * len(completions)
        
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
        
        # Calculate SBERT Cosine Similarity scores
        rewards = []
        
        try:
            # Generate embeddings for solutions and completions
            solution_embeddings = model.encode(solutions)
            completion_embeddings = model.encode(completion_contents)
            
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(solution_embeddings, completion_embeddings)
            
            # Extract diagonal (pairwise similarities between corresponding items)
            for i in range(min(len(solutions), len(completion_contents))):
                sim_score = float(similarity_matrix[i, i])
                rewards.append(sim_score)
        
        except Exception as e:
            logger.warning(f"Error in SBERT Cosine calculation: {e}")
            # If there's an error, return zeros
            rewards = [0.0] * len(completion_contents)
        
        return rewards


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
orms['sbert_cosine_reward'] = SBERTCosineReward
