#!/usr/bin/env python
# -*- coding: utf-8 -*-

# best performance:
# BLEU: 0.060
# METEOR: 0.120
# ROUGE-1-F1: 0.150
# ROUGE-2-F1: 0.045
# ROUGE-L-F1: 0.140

import csv
import nltk

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def main():
    # 如果你的 CSV 文件是 "my_llava_eeg_results.csv"
    csv_path = "my_llava_eeg_results.csv"

    # 若还没下载 nltk 数据，需要先执行:
    # import nltk
    # nltk.download("punkt")
    # nltk.download("wordnet")

    references = []
    predictions = []

    # 读取 CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = row["expected_output"].replace("</s>", "").strip()
            hyp = row["model_output"].replace("</s>", "").strip()

            references.append(ref)
            predictions.append(hyp)

    # 定义相关评分器
    smooth = SmoothingFunction()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    bleu_scores = []
    meteor_scores = []
    rouge1_f1 = []
    rouge2_f1 = []
    rougel_f1 = []

    # 对每一对 (ref, hyp) 进行评分
    for ref, hyp in zip(references, predictions):
        # 1) 先用 nltk.word_tokenize 做分词
        ref_tokens = nltk.word_tokenize(ref)
        hyp_tokens = nltk.word_tokenize(hyp)

        # ========== BLEU ==========
        b = sentence_bleu(
            [ref_tokens],       # BLEU 需要 [reference_tokens]
            hyp_tokens,         # hypothesis_tokens
            smoothing_function=smooth.method1
        )
        bleu_scores.append(b)

        # ========== METEOR ==========
        # meteor_score 的签名通常是 meteor_score(list_of_reference_tokens, hypothesis_tokens)
        # 这里要注意它的第一个参数是一个二维结构: [ [ref_token1, ref_token2, ...] ]
        # 因为 meteor_score 可以同时传多个参考答案
        m = meteor_score([ref_tokens], hyp_tokens)
        meteor_scores.append(m)

        # ========== ROUGE ==========
        # rouge_scorer.RougeScorer(...) 默认直接对字符串进行匹配
        # 因此这里可以不分词，或把它们 join 回去。两种用法都可以。
        # 如果想用原始字符串:
        r = scorer.score(ref, hyp)
        rouge1_f1.append(r["rouge1"].fmeasure)
        rouge2_f1.append(r["rouge2"].fmeasure)
        rougel_f1.append(r["rougeL"].fmeasure)

    # 计算平均值
    def average(lst):
        return sum(lst) / len(lst) if lst else 0.0

    avg_bleu   = average(bleu_scores)
    avg_meteor = average(meteor_scores)
    avg_r1     = average(rouge1_f1)
    avg_r2     = average(rouge2_f1)
    avg_rl     = average(rougel_f1)

    # 输出
    print("=== Evaluation Results ===")
    print(f"BLEU:         {avg_bleu:.4f}")
    print(f"METEOR:       {avg_meteor:.4f}")
    print(f"ROUGE-1-F1:   {avg_r1:.4f}")
    print(f"ROUGE-2-F1:   {avg_r2:.4f}")
    print(f"ROUGE-L-F1:   {avg_rl:.4f}")

if __name__ == "__main__":
    main()
