from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def bleu(reference, candidate, weights=(0, 1, 0, 0)):
    smooth = SmoothingFunction()
    score = sentence_bleu(reference, candidate, weights=weights, smoothing_function=smooth.method4)
    return score


def rouge_1(reference, candidate):
    temp = 0
    ngram_all = len(reference)
    for x in reference:
        if x in candidate:
            temp = temp + 1
    rouge_1 = temp / ngram_all
    return rouge_1


def rouge_2(reference, candidate):
    gram_2_reference = []
    gram_2_candidate = []
    temp = 0
    ngram_all = len(reference) - 1
    for x in range(len(candidate) - 1):
        gram_2_candidate.append(candidate[x] + candidate[x + 1])
    for x in range(len(reference) - 1):
        gram_2_reference.append(reference[x] + reference[x + 1])
    for x in gram_2_candidate:
        if x in gram_2_reference:
            temp = temp + 1
    rouge_2 = temp / ngram_all
    return rouge_2


if __name__ == '__main__':
    reference = [['This', 'is', 'a', 'test']]
    candidate = ['This', 'is', 'test']
    print("bleu: " + str(bleu(reference, candidate)))
    print("rouge1: " + str(rouge_1(reference[0], candidate)))
    print("rouge2: " + str(rouge_2(reference[0], candidate)))
