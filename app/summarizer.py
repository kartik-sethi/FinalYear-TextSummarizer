import spacy
from collections import Counter
from PyPDF2 import PdfReader
import io

nlp = spacy.load('en_core_web_sm')

class Summarizer:
    def __init__(self, nlp):
        self.nlp = nlp

    def _clean_text(self, text):
        cleaned_text = text.replace('\n', ' ')
        doc = self.nlp(cleaned_text)
        sents = [sent.text for sent in doc.sents]
        return doc, sents

    def _get_word_weights(self, doc):
        words = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        word_counts = Counter(words)
        max_count = max(word_counts.values())
        word_weights = { key : value / max_count for key, value in word_counts.items() }
        return word_weights

    def _get_sentence_weights(self, doc, word_weights):
        sentence_weights = {}
        for idx, sentence in enumerate(doc.sents):
            cleaned_sentence = [token.lemma_.lower() for token in sentence if not token.is_stop and not token.is_punct]
            cleaned_length = len(cleaned_sentence)
            sentence_scores = [word_weights.get(word, 0) for word in cleaned_sentence]
            sentence_weights[idx] = sum(sentence_scores) / cleaned_length
        
        return sentence_weights



    def _get_top_n_sentences(self, sents, sentence_weights, n):
        sentence_weights = sorted(sentence_weights.items(), key=lambda x: -x[1])
        top_sentences_idx = [sentence[0] for sentence in sentence_weights[:n]]
        summary = []
        for idx in top_sentences_idx:
            summary.append(sents[idx])
        summary_text = " ".join(summary)
        
        return summary_text

    def summarize_text(self, text, n):
        doc, sents = self._clean_text(text)
        word_weights = self._get_word_weights(doc)
        sentence_weights = self._get_sentence_weights(doc, word_weights)
        summary = self._get_top_n_sentences(sents, sentence_weights, n)
        return word_weights, sentence_weights, sents, summary