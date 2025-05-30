
import os
import re
import copy
import math

from datetime import datetime
from math_verify import parse, verify
from collections import deque
import ast

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return float(inter) / union

def direct_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        student_answer = content.strip()
        # Compare the extracted answers
        if student_answer == ground_truth:
            reward = 1.0
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
            except:
                pass
    return rewards

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
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
        
        # If float verification failed, try symbolic verification
        if reward == 0.0 and content_match is None:
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"extract answer: {student_answer}\n")
                    f.write(f"Solution: {sol}\n")
            except:
                pass
    return rewards


def accuracy_reward_weight(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()

            # Compare the extracted answers
            if student_answer == ground_truth:
                import numpy as np
                sample = np.random.uniform(0, 1.5)

                if '<think>' in content or '</think>' in content:
                    if sample >= 1:
                        reward = 1.5
                    else:
                        reward = 1
                else:
                    if sample >= 1:
                        reward = 1.
                    else:
                        reward = 1.5
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail

        # If float verification failed, try symbolic verification
        if reward == 0.0 and content_match is None:
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    if '<think>' in content or '</think>' in content:
                        if sample >= 1:
                            reward = 1.5
                        else:
                            reward = 1
                    else:
                        if sample >= 1:
                            reward = 1.
                        else:
                            reward = 1.5
            except Exception:
                pass  # Continue to next verification method if this fails

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"extract answer: {student_answer}\n")
                    f.write(f"Solution: {sol}\n")
            except:
                pass
    return rewards


def accuracy_reward_weight(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()

            # Compare the extracted answers
            if student_answer == ground_truth:
                import numpy as np
                sample = np.random.uniform(0, 1.5)

                if '<think>' in content or '</think>' in content:
                    if sample >= 1:
                        reward = 1.5
                    else:
                        reward = 1
                else:
                    if sample >= 1:
                        reward = 1.
                    else:
                        reward = 1.5
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail

        # If float verification failed, try symbolic verification
        if reward == 0.0 and content_match is None:
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    if '<think>' in content or '</think>' in content:
                        if sample >= 1:
                            reward = 1.5
                        else:
                            reward = 1
                    else:
                        if sample >= 1:
                            reward = 1.
                        else:
                            reward = 1.5
            except Exception:
                pass  # Continue to next verification method if this fails

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"extract answer: {student_answer}\n")
                    f.write(f"Solution: {sol}\n")
            except:
                pass
    return rewards


def math_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    epsilon1=0.05
    epsilon2=0.20
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            
            # Try to convert both answers to float for numerical comparison
            try:
                a_pred = float(student_answer)
                a_gt = float(ground_truth)
                
                # Calculate absolute difference
                diff = abs(a_pred - a_gt)
                abs_gt = abs(a_gt)
                
                # Handle exact match case
                if diff < epsilon1 * abs_gt:
                    reward = 1.0
                # Handle completely incorrect case
                elif diff > epsilon2 * abs_gt:
                    reward = 0.0
                # Handle partial match case with smooth transition
                else:
                    normalized_diff = (diff - epsilon1 * abs_gt) / ((epsilon2 - epsilon1) * abs_gt)
                    reward = 0.5 * (math.cos(math.pi * normalized_diff) + 1)
                    
            except ValueError:
                # If conversion to float fails, do exact string matching
                if student_answer == ground_truth:
                    reward = 1.0
                    
        except Exception:
            pass  # Keep reward as 0.0 if string matching fails
        
        # If float verification failed, try symbolic verification
        if reward == 0.0 and content_match is None:
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails           

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
            except:
                pass
    return rewards


def func_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    def extract_items(text):
        pattern = re.compile(r"(\w+)\((\w+),\s*'?(\w+)'?\)")
        matches = pattern.findall(text)
        filtered_matches = list(set(matches))
        return filtered_matches, len(filtered_matches) / len(matches)
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        try:
            # Extract (func, object_id, value) pairs
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            content_match = content_match.group(1).strip() if content_match else content.strip()
            pred_list, repeat_panelty = extract_items(content_match)
            sol_list, _ = extract_items(sol)
            
            item_score = repeat_panelty / max(len(pred_list), len(sol_list))
            
            pred_queue = deque(pred_list)
            sol_queue = deque(sol_list)
            
            # full mapping
            full_mapping_num = 0
            exact_matches = [(p, s) for p in pred_queue for s in sol_queue if p == s]
            for p, s in exact_matches:
                if p in pred_queue and s in sol_queue:
                    full_mapping_num += 1
                    pred_queue.remove(p)
                    sol_queue.remove(s)
            reward += full_mapping_num * item_score
            
            # (func, object_id) mapping
            partial_matches_1_num = 0
            partial_matches_1 = [(p, s) for p in pred_queue for s in sol_queue if p[:2] == s[:2]]
            for p, s in partial_matches_1:
                if p in pred_queue and s in sol_queue:
                    partial_matches_1_num += 1
                    pred_queue.remove(p)
                    sol_queue.remove(s)
            reward += partial_matches_1_num * item_score * 0.5
            
            # (func, value) mapping
            partial_matches_2_num = 0
            partial_matches_2 = [(p, s) for p in pred_queue for s in sol_queue if (p[0], p[2]) == (s[0], s[2])]
            for p, s in partial_matches_2:
                if p in pred_queue and s in sol_queue:
                    partial_matches_2_num += 1
                    pred_queue.remove(p)
                    sol_queue.remove(s)
            reward += partial_matches_2_num * item_score * 0.5
            
            # only-func mapping
            func_matches_num = 0
            func_matches = [(p, s) for p in pred_queue for s in sol_queue if p[0] == s[0]]
            for p, s in func_matches:
                if p in pred_queue and s in sol_queue:
                    func_matches_num += 1
                    pred_queue.remove(p)
                    sol_queue.remove(s)
            reward += func_matches_num * item_score * 0.25

        except Exception:
            pass

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                    f.write(f"Full Mapping: {exact_matches}\n")
                    f.write(f"Func-Object Mapping: {partial_matches_1}\n")
                    f.write(f"Func-Value Mapping: {partial_matches_2}\n")
                    f.write(f"Func-Only: {func_matches}\n")
            except:
                pass
    return rewards


def only_full_func_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    def extract_items(text):
        pattern = re.compile(r"(\w+)\((\w+),\s*'?(\w+)'?\)")
        matches = pattern.findall(text)
        filtered_matches = list(set(matches))
        return filtered_matches, len(filtered_matches) / len(matches)
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        try:
            # Extract (func, object_id, value) pairs
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            content_match = content_match.group(1).strip() if content_match else content.strip()
            pred_list, repeat_panelty = extract_items(content_match)
            sol_list, _ = extract_items(sol)
            
            item_score = repeat_panelty / max(len(pred_list), len(sol_list))
            
            pred_queue = deque(pred_list)
            sol_queue = deque(sol_list)
            
            # full mapping
            full_mapping_num = 0
            exact_matches = [(p, s) for p in pred_queue for s in sol_queue if p == s]
            for p, s in exact_matches:
                if p in pred_queue and s in sol_queue:
                    full_mapping_num += 1
                    pred_queue.remove(p)
                    sol_queue.remove(s)
            reward += full_mapping_num * item_score

        except Exception:
            pass

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                    f.write(f"Full Mapping: {exact_matches}\n")
            except:
                pass
    return rewards


def penalty_func_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    def extract_items(text):
        pattern = re.compile(r"(\w+)\((\w+),\s*'?(\w+)'?\)")
        matches = pattern.findall(text)
        filtered_matches = list(set(matches))
        return filtered_matches, len(filtered_matches) / len(matches)
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try string matching
        try:
            # Extract (func, object_id, value) pairs
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            content_match = content_match.group(1).strip() if content_match else content.strip()
            pred_list, repeat_panelty = extract_items(content_match)
            sol_list, _ = extract_items(sol)
            
            item_score = repeat_panelty / max(len(pred_list), len(sol_list))
            
            pred_queue = deque(pred_list)
            sol_queue = deque(sol_list)
            
            # full mapping
            full_mapping_num = 0
            exact_matches = [(p, s) for p in pred_queue for s in sol_queue if p == s]
            for p, s in exact_matches:
                if p in pred_queue and s in sol_queue:
                    full_mapping_num += 1
                    pred_queue.remove(p)
                    sol_queue.remove(s)
            reward += full_mapping_num * item_score
            
            # (func, object_id) mapping
            partial_matches_1_num = 0
            partial_matches_1 = [(p, s) for p in pred_queue for s in sol_queue if p[:2] == s[:2]]
            for p, s in partial_matches_1:
                if p in pred_queue and s in sol_queue:
                    partial_matches_1_num += 1
                    pred_queue.remove(p)
                    sol_queue.remove(s)
            reward += partial_matches_1_num * item_score * -0.25
            
            # (func, value) mapping
            partial_matches_2_num = 0
            partial_matches_2 = [(p, s) for p in pred_queue for s in sol_queue if (p[0], p[2]) == (s[0], s[2])]
            for p, s in partial_matches_2:
                if p in pred_queue and s in sol_queue:
                    partial_matches_2_num += 1
                    pred_queue.remove(p)
                    sol_queue.remove(s)
            reward += partial_matches_2_num * item_score * -0.25
            
            # only-func mapping
            func_matches_num = 0
            func_matches = [(p, s) for p in pred_queue for s in sol_queue if p[0] == s[0]]
            for p, s in func_matches:
                if p in pred_queue and s in sol_queue:
                    func_matches_num += 1
                    pred_queue.remove(p)
                    sol_queue.remove(s)
            reward += func_matches_num * item_score * -0.5

        except Exception:
            pass

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                    f.write(f"Full Mapping: {exact_matches}\n")
                    f.write(f"Func-Object Mapping: {partial_matches_1}\n")
                    f.write(f"Func-Value Mapping: {partial_matches_2}\n")
                    f.write(f"Func-Only: {func_matches}\n")
            except:
                pass
    return rewards



def accuracy_reward_mix(completions, solution, reward_type, **kwargs):
    if reward_type[0]=='normal':
        #print('normal')
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, sol in zip(contents, solution):
            reward = 0.0
            # Try string matching
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)

                #print(content_match.group(1).strip())
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                #print(student_answer)
                #print(student_answer, sol)
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

            # If float verification failed, try symbolic verification
            if reward == 0.0:
                try:
                    answer = parse(content)
                    if float(verify(answer, parse(sol))) > 0:
                        reward = 1.0
                except Exception:
                    pass  # Continue to next verification method if this fails

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "True":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                try:
                    with open(log_path, "a") as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"extract answer: {student_answer}\n")
                        f.write(f"reward type: {reward_type[0]}\n")
                        f.write(f"Solution: {sol}\n")
                except:
                    pass
        return rewards
    elif reward_type[0] == 'grounding':
        #print('grounding')
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        #current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            #print(content,sol)
            reward = 0.0
            # Try symbolic verification first
            content_answer = content
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                #print('***************')
                #print(content_answer_match)
                #print('***************')
                if content_answer_match:

                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = ast.literal_eval(sol_match.group(1).strip() if sol_match else sol.strip())

                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    #print('???????????')
                    #print(bbox_match)
                    #print('???????????')
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)),
                                int(bbox_match.group(4))]
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, ground_truth)
                else:
                    content_answer = content
            except Exception:
                pass  # Continue to next verification method if this fails

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "True":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a") as f1:
                    #print('>>>>>>>>>>>>>>>>>>>>>')
                    #print(content_answer)
                    #print('>>>>>>>>>>>>>>>>>>>>>')
                    f1.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f1.write(f"Content: {content}\n")
                    f1.write(f"extract answer: {content_answer}\n")
                    f1.write(f"reward type: {reward_type[0]}\n")
                    f1.write(f"Solution: {sol}\n")

        return rewards
    else:
        sys.exit('error reward')

def format_reward_mix(completions, reward_type,**kwargs):
    """Reward function that checks if the completion has a specific format."""
    # since batch size is 1, the completions within a batch are with same reward type.
    # need to be revised if bs>1
    if reward_type[0]=='normal':
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    elif reward_type[0]=='grounding':
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
    else:
        sys.exit('error reward')

    completion_contents = [completion[0]["content"] for completion in completions]


    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    for i in range(len(matches)):
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH1")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- Format reward: {1.0 if matches[i] else 0.0} -------------\n")
                    f.write(f"Content: {completion_contents[i]}\n")
                    f.write(f"reward type: {reward_type[i]}\n")
            except:
                pass
    #print(reward_type)
    return [1.0 if match else 0.0 for match in matches]



def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def format_reward_adaptive_three(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""

    pattern1 = r"<check>.*?</check>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    pattern2 = r"<check>.*?</check>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches1 = [re.fullmatch(pattern1, content, re.DOTALL) for content in completion_contents]
    matches2 = [re.fullmatch(pattern2, content, re.DOTALL) for content in completion_contents]
    matches3 = [re.search(r'<check>(.*?)</check>', content) for content in completion_contents]
    rewards = []
    for i in range(len(matches1)):
        if matches1[i] or matches2[i]:
            if matches3[i]:
                if matches3[i].group(1).strip() == 'need thinking'  and '<think>' in completion_contents[i]:
                    rewards.append(1.)
                elif matches3[i].group(1).strip() == 'not need thinking' and '<think>' not in completion_contents[i]:
                    rewards.append(1.)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards



def format_reward_adaptive_three_task(completions,task, **kwargs):
    """Reward function that checks if the completion has a specific format."""

    pattern1 = r"<task>.*?</task>\s*<check>.*?</check>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    pattern2 = r"<task>.*?</task>\s*<check>.*?</check>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches1 = [re.fullmatch(pattern1, content, re.DOTALL) for content in completion_contents]
    matches2 = [re.fullmatch(pattern2, content, re.DOTALL) for content in completion_contents]
    matches3 = [re.search(r'<check>(.*?)</check>', content) for content in completion_contents]
    matches4 = [re.search(r'<task>(.*?)</task>', content) for content in completion_contents]

    rewards = []
    #print(task)
    for i in range(len(matches1)):
        #if matches4[i]:
        #    print(matches4[i].group(1).strip(), task[i])
        if matches1[i] or matches2[i]:
            if matches3[i]:
                if matches3[i].group(1).strip() == 'need thinking'  and '<think>' in completion_contents[i]:
                    if matches4[i] and matches4[i].group(1).strip()=='math task' and matches4[i].group(1).strip()==task[i]:
                        rewards.append(1.)
                    else:
                        rewards.append(0.0)
                elif matches3[i].group(1).strip() == 'not need thinking' and '<think>' not in completion_contents[i]:
                    c1 = (matches4[i] and matches4[i].group(1).strip()=='perception task')
                    if c1 and matches4[i].group(1).strip()==task[i]:
                        rewards.append(1.)
                    else:
                        rewards.append(0.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards



def format_reward_adaptive_three_check(completions, need_think, **kwargs):
    """Reward function that checks if the completion has a specific format."""

    pattern1 = r"<check>.*?</check>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    pattern2 = r"<check>.*?</check>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches1 = [re.fullmatch(pattern1, content, re.DOTALL) for content in completion_contents]
    matches2 = [re.fullmatch(pattern2, content, re.DOTALL) for content in completion_contents]
    matches3 = [re.search(r'<check>(.*?)</check>', content) for content in completion_contents]
    rewards = []
    #print(need_think)
    for i in range(len(matches1)):
        if matches1[i] or matches2[i]:
            if matches3[i]:
                if matches3[i].group(1).strip() == 'need thinking'  and '<think>' in completion_contents[i]:
                    if need_think[i]:
                        rewards.append(1)
                    else:
                        rewards.append(-0.5)
                elif matches3[i].group(1).strip() == 'not need thinking' and '<think>' not in completion_contents[i]:
                    if need_think[i]:
                        rewards.append(-0.5)
                    else:
                        rewards.append(1.)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
        if os.getenv("DEBUG_MODE") == "True":
            log_path = os.getenv("LOG_PATH1")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- Format reward: {rewards[i]} -------------\n")
                    f.write(f"Content: {completion_contents[i]}\n")
                    f.write(f"Need Think?: {need_think[i]}\n")
            except:
                pass
    return rewards

def format_reward_adaptive(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""

    pattern1 = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern2 = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches1 = [re.fullmatch(pattern1, content, re.DOTALL) for content in completion_contents]
    matches2 = [re.fullmatch(pattern2, content, re.DOTALL) for content in completion_contents]
    rewards = []
    for i in range(len(matches1)):
        if matches1[i] or matches2[i]:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward_reason(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    #pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<answer>.*?</answer>\s*<reason>.*?</reason>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def caption_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<summary>.*?</summary>\s*<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reasoning_steps_reward(completions, **kwargs):
    """Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]

LENGTH_REWARD_START_POINT = 800

def len_reward(completions, solution, current_step, **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solution

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        correct = False
        # Try string matching
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            
            # Compare the extracted answers
            if student_answer == ground_truth:
                correct = True
            elif float(student_answer) == float(ground_truth):
                correct = True
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail
        
        # If symbolic verification failed, try symbolic verification
        if correct is False and content_match is None:
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    correct = True
            except Exception:
                pass  # Continue to next verification method if this fails
        correctness.append(copy.deepcopy(correct))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        if current_step < LENGTH_REWARD_START_POINT:
            reward = 0.0
        else:
            reward = 0.05 * reward

        rewards.append(float(reward))

    return rewards