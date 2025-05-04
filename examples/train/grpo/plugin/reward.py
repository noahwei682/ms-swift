
def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    global reward_call_counts
    reward_call_counts["accuracy"] += 1
    step_count = reward_call_counts["accuracy"]
    
    print(f"Accuracy reward call #{step_count}")
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
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
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
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
                    print(f"Verification error: {e}")
                    rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception as e:
            print(f"Error in accuracy_reward: {e}")
            rewards.append(0.0)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "accuracy_reward_direct": avg_reward,
            "accuracy_reward_call": step_count,
            "accuracy_rewards_raw": rewards,
            "accuracy_reward_min": min(rewards) if rewards else 0,
            "accuracy_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: accuracy_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_accuracy_rewards = rewards
    
    return rewards

# Define BM25 reward function based on BM25 implementation
def bm25_reward(completions, **kwargs):
    """Reward function that computes BM25 scores between model completion and solution"""
    global reward_call_counts
    reward_call_counts["bm25"] += 1
    step_count = reward_call_counts["bm25"]
    
    print(f"BM25 reward call #{step_count}")
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
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
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
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
            
            print(f"BM25 score for pair: {normalized_score:.4f}")
        except Exception as e:
            print(f"Error in BM25 calculation: {e}")
            rewards.append(0.0)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "bm25_reward_direct": avg_reward,
            "bm25_reward_call": step_count,
            "bm25_rewards_raw": rewards,
            "bm25_reward_min": min(rewards) if rewards else 0,
            "bm25_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: bm25_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_bm25_rewards = rewards
    
    return rewards

# Define F1 score reward function
def f1_reward(completions, **kwargs):
    """Reward function that computes F1 scores between model completion and solution"""
    global reward_call_counts
    reward_call_counts["f1"] += 1
    step_count = reward_call_counts["f1"]
    
    print(f"F1 reward call #{step_count}")
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
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
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
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
            print(f"F1 score for pair: {f1:.4f} (P: {precision:.4f}, R: {recall:.4f}, common: {len(common_tokens)})")
            
        except Exception as e:
            print(f"Error in F1 calculation: {e}")
            rewards.append(0.0)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "f1_reward_direct": avg_reward,
            "f1_reward_call": step_count,
            "f1_rewards_raw": rewards,
            "f1_reward_min": min(rewards) if rewards else 0,
            "f1_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: f1_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_f1_rewards = rewards
    
    return rewards

# Define Recall reward function
def recall_reward(completions, **kwargs):
    """Reward function that computes recall score between model completion and solution"""
    global reward_call_counts
    reward_call_counts["recall"] += 1
    step_count = reward_call_counts["recall"]
    
    print(f"Recall reward call #{step_count}")
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
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
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
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
            print(f"Recall score for pair: {recall:.4f} (common: {len(common_tokens)}, solution: {len(solution_tokens)})")
            
        except Exception as e:
            print(f"Error in Recall calculation: {e}")
            rewards.append(0.0)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "recall_reward_direct": avg_reward,
            "recall_reward_call": step_count,
            "recall_rewards_raw": rewards,
            "recall_reward_min": min(rewards) if rewards else 0,
            "recall_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: recall_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_recall_rewards = rewards
    
    return rewards

# Define Precision reward function
def precision_reward(completions, **kwargs):
    """Reward function that computes precision score between model completion and solution"""
    global reward_call_counts
    reward_call_counts["precision"] += 1
    step_count = reward_call_counts["precision"]
    
    print(f"Precision reward call #{step_count}")
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
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
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
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
            print(f"Precision score for pair: {precision:.4f} (common: {len(common_tokens)}, generated: {len(content_tokens)})")
            
        except Exception as e:
            print(f"Error in Precision calculation: {e}")
            rewards.append(0.0)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "precision_reward_direct": avg_reward,
            "precision_reward_call": step_count,
            "precision_rewards_raw": rewards,
            "precision_reward_min": min(rewards) if rewards else 0,
            "precision_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: precision_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_precision_rewards = rewards
    
    return rewards

# Define Sentence-BERT Cosine Similarity reward function
def sbert_cosine_reward(completions, **kwargs):
    """Reward function that computes cosine similarity between sentence embeddings of completions and solutions"""
    global reward_call_counts, sbert_model
    
    reward_call_counts["sbert_cosine"] += 1
    step_count = reward_call_counts["sbert_cosine"]
    
    print(f"SBERT Cosine reward call #{step_count}")
    
    # Check if SBERT model is available
    if sbert_model is None:
        print("Sentence-BERT model not available. Returning zero rewards.")
        return [0.0] * len(completions)
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
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
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
        # Truncate the longer list to match the shorter list
        min_len = min(len(solutions), len(completion_contents))
        solutions = solutions[:min_len]
        completion_contents = completion_contents[:min_len]
    
    # Calculate SBERT Cosine Similarity scores
    rewards = []
    
    try:
        # Generate embeddings for solutions and completions
        solution_embeddings = sbert_model.encode(solutions, convert_to_tensor=True)
        completion_embeddings = sbert_model.encode(completion_contents, convert_to_tensor=True)
        
        # Convert to numpy arrays for sklearn cosine_similarity
        if torch.is_tensor(solution_embeddings):
            solution_embeddings_np = solution_embeddings.cpu().numpy()
        else:
            solution_embeddings_np = solution_embeddings
            
        if torch.is_tensor(completion_embeddings):
            completion_embeddings_np = completion_embeddings.cpu().numpy()
        else:
            completion_embeddings_np = completion_embeddings
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(solution_embeddings_np, completion_embeddings_np)
        
        # Extract diagonal (pairwise similarities between corresponding items)
        for i in range(min(len(solutions), len(completion_contents))):
            sim_score = float(similarity_matrix[i, i])
            rewards.append(sim_score)
            print(f"SBERT Cosine similarity for pair {i}: {sim_score:.4f}")
    
    except Exception as e:
        print(f"Error in SBERT Cosine calculation: {e}")
        # If there's an error, return zeros
        rewards = [0.0] * len(completion_contents)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "sbert_cosine_reward_direct": avg_reward,
            "sbert_cosine_reward_call": step_count,
            "sbert_cosine_rewards_raw": rewards,
            "sbert_cosine_reward_min": min(rewards) if rewards else 0,
            "sbert_cosine_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: sbert_cosine_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_sbert_cosine_rewards = rewards
    
    return rewards
