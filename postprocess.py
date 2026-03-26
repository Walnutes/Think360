import argparse
import json
import os
from tqdm import tqdm
from openai import OpenAI
from mathruler.grader import grade_answer
from loguru import logger
from prompt import build_extract_prompt, build_score_prompt_with_validation

client = OpenAI(
    api_key="API_KEY", 
    base_url="API_URL",
)

def get_response(client, messages):
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": messages}
    ]
    response = client.chat.completions.create(model='gpt-4o-mini', messages=messages, temperature=0.0)
    response_text = response.choices[0].message.content.strip()
    return response_text

def llm_eval_score(question, prediction, answer, client):
    """
    Evaluate the score of a prediction using the model.
    """
    extract_prompt = build_extract_prompt(prediction, question)
    extracted_answer = get_response(client, extract_prompt)

    if grade_answer(extracted_answer, answer):
        return extracted_answer, 1.0
    else:
        score_prompt = build_score_prompt_with_validation(question, extracted_answer, answer)
        response_text = get_response(client, score_prompt)
        if response_text in ['0', '1']:
            return extracted_answer, int(response_text)
    return extracted_answer, 0.0

def evaluate_prediction(item: dict, client: OpenAI):
    """
    Evaluate a prediction against the ground truth
    """
    prediction = item.get('prediction', '')
    question = item.get('question', '')
    answer = item.get('answer', '')

    if grade_answer(prediction, answer):
        return prediction, 1.0
    else:
        try:
            extracted_answer, score = llm_eval_score(question, prediction, answer, client)
            return extracted_answer, score
        except Exception as e:
            logger.error(f"Error in the stage of llm eval score: {e}")
            return prediction, 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract answer from prediction file.")
    parser.add_argument('--prediction_file', type=str, required=True, help="Path to the prediction file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output file.")
    args = parser.parse_args()

    with open(args.prediction_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for item in tqdm(data, desc="Evaluating"):
        try:
            extract_ans, accuracy = evaluate_prediction(item, client)
            item['extraction'] = extract_ans
            item['accuracy'] = accuracy
            results.append(item)
        except Exception as e:
            logger.error(f"Error evaluating prediction {item.get('id')}: {str(e)}")
            item['extraction'] = ""
            item['accuracy'] = 0.0
            results.append(item)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)