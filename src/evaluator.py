from typing import List, Dict, Any, Union
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from classifier.detector import PhishingClassifier

class Evaluator:
    def __init__(self, classifier: PhishingClassifier):
        self.classifier = classifier

    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        y_true = []
        y_pred = []

        for item in test_data:
            text_or_email = item["text"]
            expected_label = item["expected_label"]
            
            result = self.classifier.classify(text_or_email)
            y_true.append(expected_label)
            y_pred.append(result["classification"])

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=["phishing", "legitimate", "uncertain"])

        return {
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "labels": ["phishing", "legitimate", "uncertain"]
        }