from bridging.calc_score import get_bert_score
import json
import numpy as np

if __name__ == "__main__":
    result_file = "/home/seungwoochoi/data/x_bridging/DiffuSeq/bridging/outputs/test_result9.jsonl"
    with open(result_file, "r") as f:
        results = f.readlines()
    
    # print(results)

    bridg_recovers = [json.loads(line)["bridg_recover"] for line in results]
    vanilla_recovers = [json.loads(line)["vanilla_recover"] for line in results]
    references = [json.loads(line)["reference"] for line in results]

    print(len(references))
    print(len(bridg_recovers))
    print(len(vanilla_recovers))

    bridg_bert_scores_p, bridg_bert_scores_r, bridg_bert_scores_f1 = get_bert_score(bridg_recovers, references)
    vanilla_bert_scores_p, vanilla_bert_scores_r, vanilla_bert_scores_f1 = get_bert_score(vanilla_recovers, references)

    print(bridg_bert_scores_f1)

    print("BERT Scores: Precision / Recall / F1")

    print(f"Bridging BERT Score: {np.mean(bridg_bert_scores_p.tolist()):.4f} / {np.mean(bridg_bert_scores_r.tolist()):.4f} / {np.mean(bridg_bert_scores_f1.tolist()):.4f}")
    print(f"Vanilla BERT Score: {np.mean(vanilla_bert_scores_p.tolist()):.4f} / {np.mean(vanilla_bert_scores_r.tolist()):.4f} / {np.mean(vanilla_bert_scores_f1.tolist()):.4f}")

    #visualize with matplotlib (f1 scores for each example)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(bridg_bert_scores_f1.tolist(), label="Bridging F1 Score", marker='o')
    plt.plot(vanilla_bert_scores_f1.tolist(), label="Vanilla F1 Score", marker='o')
    plt.xlabel("Sample Index")
    plt.ylabel("F1 Score")
    plt.title("BERT F1 Scores Comparison")
    plt.legend()
    #save fig
    plt.savefig("/home/seungwoochoi/data/x_bridging/DiffuSeq/bert_f1_scores_comparison.png")

    #visualize with matplotlib (linear relationship between the delta(bridg - f1) and vanilla f1 scores)
    plt.figure(figsize=(10, 5))
    plt.scatter((bridg_bert_scores_f1 - vanilla_bert_scores_f1).tolist(), vanilla_bert_scores_f1.tolist(), label="Delta vs Vanilla", marker='o')
    plt.xlabel("Delta (Bridging - Vanilla) F1 Score")
    plt.ylabel("Vanilla F1 Score")
    plt.title("Linear Relationship between Delta and Vanilla F1 Scores")
    plt.legend()
    plt.savefig("/home/seungwoochoi/data/x_bridging/DiffuSeq/bert_f1_scores_delta_vs_vanilla.png")