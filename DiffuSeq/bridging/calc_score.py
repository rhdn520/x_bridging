from dotenv import load_dotenv
load_dotenv()
from bert_score import score

def get_bert_score(output_lst: list[str], ref_lst: list[str]):
    """
    Calculates the BERT score between two sentences.

    Args:
        output_lst: list[str]: Model Output.
        ref_lst: list[str]: Reference Sentences.

    Returns:
        tuple: A tuple containing the precision, recall, and F1 BERT scores
               as floating-point numbers. Returns (None, None, None) if an
               error occurs.
    """
    try:
        # The bert_score.score function expects lists of sentences
        candidates = output_lst
        references = ref_lst

        # Calculate BERT scores.
        # lang="en" uses the roberta-large model by default.
        # verbose=True will show a progress bar.
        P, R, F1 = score(
            candidates,
            references,
            lang="en",
            verbose=True,
            # You can specify a different model if needed, for example:
            # model_type='bert-base-uncased'
        )

        # The results are tensors, so we extract the float value using .item()
        precision = P
        recall = R
        f1_score = F1

        return precision, recall, f1_score

    except Exception as e:
        print(f"An error occurred while calculating BERT score: {e}")
        return None, None, None

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Sentences are very similar
    sent1_a = "The cat is on the mat."
    sent1_b = "There is a cat on the mat."

    print(f"Calculating BERT score for:\n  - '{sent1_a}'\n  - '{sent1_b}'")
    precision1, recall1, f1_1 = get_bert_score(sent1_a, sent1_b)

    if f1_1 is not None:
        print(f"\nScores:\n  - Precision: {precision1:.4f}\n  - Recall:    {recall1:.4f}\n  - F1 Score:  {f1_1:.4f}")
        print("-" * 30)


    # Example 2: Sentences have some overlapping meaning
    sent2_a = "He loves to play football."
    sent2_b = "He is a fan of soccer."

    print(f"\nCalculating BERT score for:\n  - '{sent2_a}'\n  - '{sent2_b}'")
    precision2, recall2, f1_2 = get_bert_score(sent2_a, sent2_b)

    if f1_2 is not None:
        print(f"\nScores:\n  - Precision: {precision2:.4f}\n  - Recall:    {recall2:.4f}\n  - F1 Score:  {f1_2:.4f}")
        print("-" * 30)


    # Example 3: Sentences are completely different
    sent3_a = "The sky is blue."
    sent3_b = "Python is a programming language."

    print(f"\nCalculating BERT score for:\n  - '{sent3_a}'\n  - '{sent3_b}'")
    precision3, recall3, f1_3 = get_bert_score(sent3_a, sent3_b)

    if f1_3 is not None:
        print(f"\nScores:\n  - Precision: {precision3:.4f}\n  - Recall:    {recall3:.4f}\n  - F1 Score:  {f1_3:.4f}")
        print("-" * 30)