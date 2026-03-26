def get_gpt4_score_ICE():
    example_1 = """
[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer]: Extracted Answer: \\((-2, 1)\\)
Judgement: 0
""" # noqa

    example_2 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer]: B:2\u221a{{3}}
Judgement: 0
""" # noqa

    example_3 = """
[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer]: Range: \\((-4, 1]\\)
Judgement: 0
""" # noqa

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer]: null
Judgement: 0
""" # noqa

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\n(A):2\n(B):2\u221a{{3}}\n(C):\u221a{{3}}\n(D):2\u221a{{2}}
[Standard Answer]: (C)
[Model_answer]: C
Judgement: 1
""" # noqa
    return [example_1, example_2, example_3, example_4]


def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""
    return [example_1, example_2, example_3, example_4, example_5]

def build_extract_prompt(prediction, question):
    task_description = """
Please read the following example.
Then output the answer extracted from the model response directly. No "Extracted answer:" in your answer.
Extract the complete and final answer to accurately assess correctness regardless of presentation format, especially for responses involving simplification, approximation, or format conversions.
\n
"""
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += 'Question: ' + question + '\n'
    prompt += 'Model response: ' + prediction
    prompt += 'Extracted answer:'
    return prompt   


def build_score_prompt(question, extract, answer):
    task_description = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
""" # noqa
    demo_prompt = task_description
    examples = get_gpt4_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    Please output the judgement score directly with no explanation.
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer]: {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt

def build_score_prompt_with_validation(question, extract, answer):
    if answer.strip().startswith("VALIDATION RULES"):
        task_description = """
Below are validation rules for a math question. Please check if the extracted answer satisfies ALL the following rules. 
If the extracted answer satisfies all rules, output 1; otherwise, output 0. Output only the judgement score, no explanation.

[Question]: {question}
[Validation Rules]: {rules}
[Extracted Answer]: {extract}
Judgement:"""
        rules = answer.strip().replace("VALIDATION RULES", "").strip()
        return task_description.format(question=question, rules=rules, extract=extract)
    else:
        return build_score_prompt(question, extract, answer)


# ==================== Tree Node Extraction Prompts ====================

ToT_EXTRACTION_SYSTEM_PROMPT = """You are an expert in decomposing mathematical reasoning into tree structures.

YOUR TASK: Extract key reasoning steps from solutions into a hierarchical tree where:
- DEPTH = Sequential reasoning steps that depend on previous conclusions (parent depth + 1 = child depth)
- BREADTH = Parallel exploration of different possibilities at the same depth level

CORE PRINCIPLES:
1. **EXTRACT verbatim**: Each node content MUST be directly extracted from the original solution text, preserving the original wording.
2. **CRITICAL steps only**: Focus on major logical leaps, calculations, key deductions, conclusions rather than simple listings, obvious observations, repeated information. AVOID extracting multiple nodes for simple enumeration (e.g., listing A, B, C, D, E separately). Keep complete calculation steps as ONE node (e.g., "ans = 4 + 7 = 11" should not be split). Keep algebraic transformations together (e.g., "S = ∑(1+k)(2+k) = ∑(k²+3k+2)" is ONE node).
3. **Tree structure**: Node ID: {depth}.{sequence}, where child depth = parent depth + 1. Remember (1) Depth 1 nodes: PARENT must be "None" (these are root nodes). (2) Nodes at same depth are SIBLINGS (different branches), not parent-child!

SPECIAL CASE - SIMPLE FINAL ANSWERS:
If the solution is just a simple final answer (like "D", "A", "42", "True", etc.), create a single root node:
- [NODE:1.1] with [PARENT:None]
- [ORIGINAL: The exact final answer text]
- [TYPE:conclusion]
This ensures even simple answers are properly structured as trees.
"""

def build_tot_extraction_user_prompt(question: str, answer: str, prediction: str):
    prompt = f"""Problem:
{question}

Reference answer: {answer}

Solution to decompose (EXTRACT FROM, DO NOT REWRITE):
{prediction}

TASK: Extract key reasoning steps into a tree structure using the format below. REMEMBER that **parent depth = child depth - 1**.

EXTRACTION GUIDELINES:
- Extract original text verbatim, avoid paraphrasing
- Focus on scoring points: calculations, key deductions, major conclusions
- Skip: trivial observations, simple listings (don't create separate nodes for A, B, C, D)
- Keep complete calculation steps as ONE node (e.g., "ans = 4 + 7 = 11" should not be split)
- **NEVER split algebraic transformations**: "S = ∑(1+k)(2+k) = ∑(k²+3k+2)" is ONE node, not multiple
- Number of nodes depends on problem complexity (quality over quantity)

**IMPORTANT - SIMPLE FINAL ANSWERS:**
If the solution is just a simple final answer (like "D", "A", "42", "True", etc.), create a single root node:
[NODE:1.1]
[PARENT:None]
[ORIGINAL:The exact final answer text]
[TYPE:conclusion]

STEP TYPES: calculation, deduction, observation, conclusion, exploration, verification, other

EXAMPLE FOR COMPLEX SOLUTION:
[NODE:1.1]
[PARENT:None]
[ORIGINAL:The squirrel is in the top-right compartment, next to the acorns (E).]
[TYPE:observation]

[NODE:2.1]
[PARENT:1.1]
[ORIGINAL:Analyzing the maze, the compartment containing apples (A) is separated from the rest of the maze by a wall. No continuous path connects this compartment to the squirrel's area.]
[TYPE:deduction]

[NODE:2.2]
[PARENT:1.1]
[ORIGINAL:Paths connect the compartments with berries to the squirrel's area.]
[TYPE:verification]

[NODE:3.1]
[PARENT:2.1]
[ORIGINAL:Only the compartment with A (Apples) has no connecting path to the squirrel's starting position due to a maze wall.]
[TYPE:conclusion]

EXAMPLE FOR SIMPLE FINAL ANSWER:
[NODE:1.1]
[PARENT:None]
[ORIGINAL:D]
[TYPE:conclusion]

OUTPUT FORMAT:
[NODE:{{depth}}.{{sequence}}]  ← First number = depth (1, 2, 3, 4, ...)
[PARENT:None OR {{parent_id}}] ← Must be (depth-1).X or None
[ORIGINAL:Exact text from solution]
[TYPE:calculation/deduction/observation/conclusion/exploration/verification/other]

Now extract nodes from the solution above. 
VERIFY: (1) parent depth = node depth - 1, (2) algebraic transformations NOT split across nodes.
"""
    return prompt


# ==================== Tree Node Judgement Prompts ====================

ToT_JUDGEMENT_SYSTEM_PROMPT = """You are an expert in evaluating mathematical and logical reasoning steps.

YOUR TASK: Judge the correctness of a single reasoning step within a larger reasoning tree.

You will be provided with:
- The original problem and reference answer
- The specific reasoning step (node) to evaluate
- Parent nodes (previous reasoning steps) for context
- An image if relevant to the problem

EVALUATION CRITERIA:
1. **Correctness**: Is the reasoning in this step logically sound and factually correct?
2. **Validity**: Does it follow properly from the parent nodes?
3. **Accuracy**: For calculations, are the results correct?
4. **Relevance**: Does it contribute meaningfully to solving the problem?
5. **Final Answer Check**: If this is the final answer node, does it match the reference answer?

OUTPUT: Simply respond with "True" if the reasoning step is correct, or "False" if it contains errors, logical flaws, or incorrect conclusions.
**CRITICAL OUTPUT FORMAT REQUIREMENT:**
- You MUST respond with EXACTLY one word: either "True" or "False"
- DO NOT include any explanation, analysis, reasoning, or additional text
- DO NOT use phrases like "The answer is...", "The final answer is...", "Based on..."
- Your entire response must be a single word: True or False
"""

def build_tot_judgement_user_prompt(
    question: str, 
    reference_answer: str, 
    current_node: dict, 
    parent_nodes: list = None,
    is_final_node: bool = False
):
    """
    Build prompt for judging a single tree node.
    
    Args:
        question: The original problem
        reference_answer: The correct answer for reference
        current_node: The node to judge (dict with 'id', 'original_text', 'step_type', etc.)
        parent_nodes: List of parent nodes for context (in order from root to immediate parent)
        is_final_node: Whether this is the final answer node
    """
    
    # Build context from parent nodes
    context_str = ""
    if parent_nodes and len(parent_nodes) > 0:
        context_str = "\n**Previous Reasoning Steps (for context):**\n"
        for i, parent in enumerate(parent_nodes, 1):
            parent_id = parent.get('id', f'node_{i}')
            parent_text = parent.get('original_text', '')
            context_str += f"{parent_id}: {parent_text}\n"
    else:
        context_str = "\n**Note:** This is a root node with no previous steps.\n"
    
    # Current node information
    node_id = current_node.get('id', 'unknown')
    node_text = current_node.get('original_text', '')
    node_type = current_node.get('step_type', 'unknown')
    
    # Special instruction for final node
    final_node_instruction = ""
    if is_final_node:
        final_node_instruction = f"""
**CRITICAL - FINAL ANSWER NODE**: 
This node contains the final answer. You MUST compare it with the reference answer: "{reference_answer}"
- If the final answer matches the reference answer, judge as True
- If the final answer differs from the reference answer, judge as False
This node contains the final answer. Compare it with reference answer: "{reference_answer}"
- Match → output True
- Mismatch → output False
(Remember: output ONLY True or False, no explanation)
"""
    
    prompt = f"""**Problem:**
{question}

**Reference Answer (for guidance):**
{reference_answer}
{context_str}
**Current Reasoning Step to Evaluate:**
Node ID: {node_id}
Step Type: {node_type}
Content: {node_text}
{final_node_instruction}
**Your Task:**
Evaluate whether this reasoning step is correct based on:
1. The problem context and reference answer
2. Previous reasoning steps (if any)
3. The image (if provided and relevant)
4. Logical soundness and mathematical accuracy

**Important Considerations:**
- For calculation nodes: Verify the arithmetic/algebraic operations are correct
- For deduction nodes: Check if the logical inference is valid
- For observation nodes: Verify if the observation is accurate (especially with image)
- For conclusion nodes: Check if the conclusion follows from previous steps
- Consider that intermediate steps can be correct even if the final answer is wrong
- Consider that a step can be wrong even if it leads to the correct final answer

Respond with ONLY one word: "True" (if correct) or "False" (if incorrect/flawed).
**STRICT OUTPUT FORMAT - FOLLOW EXACTLY:**
Output ONLY a single word with no other text:
- "True" if the reasoning step is correct
- "False" if it contains any errors or flaws

DO NOT explain your reasoning. DO NOT add any prefix or suffix. Just output: True or False
"""
    
    return prompt
