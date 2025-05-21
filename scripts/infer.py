import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_dir: str):
    """
    Load the fine-tuned Seq2Seq model and tokenizer from the given directory.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return tokenizer, model


def infer_question(tokenizer, model, question: str, max_input_length: int = 512, max_output_length: int = 128):
    """
    Generate an answer from the fine-tuned model for a given question.

    Args:
        tokenizer: Hugging Face tokenizer
        model: Hugging Face Seq2Seq model
        question: The question string (should include any prefix like "question: ...")
        max_input_length: Maximum token length for the input
        max_output_length: Maximum token length for the output

    Returns:
        answer: The model-generated answer string
    """
    # Tokenize the input question
    inputs = tokenizer(
        question,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_input_length,
    )

    # Generate with the model
    outputs = model.generate(
        **inputs,
        max_length=max_output_length,
        num_beams=4,
        early_stopping=True,
    )

    # Decode the output tokens
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def main():
    parser = argparse.ArgumentParser(
        description="Infer answers from a fine-tuned SUSIE IOM model"
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='models/iom_model',
        help='Path to the fine-tuned model directory'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Question to ask the model (prefix as needed, e.g., "question: ...")'
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)
    model.eval()

    if args.question:
        answer = infer_question(tokenizer, model, args.question)
        print("Answer:", answer)
    else:
        print("Entering interactive mode. Type 'exit' to quit.")
        while True:
            question = input("question: ").strip()
            if question.lower() in ('exit', 'quit'):
                break
            q_text = question if question.startswith('question:') else f"question: {question}"
            answer = infer_question(tokenizer, model, q_text)
            print("answer:", answer)

if __name__ == '__main__':
    main()