from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import shap
import matplotlib.pyplot as plt

def evaluate_business_cost(y_true, y_pred, y_prob):
    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    
    # Calculate Cost
    # Assuming: False Negative (Missing Fraud) costs $100. False Positive (Blocking good user) costs $5.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    financial_loss = (fn * 100) + (fp * 5)
    
    print(f"Model AUPRC: {auprc:.4f}")
    print(f"Total Business Cost of Errors: ${financial_loss}")
    print(f"False Positives (User Friction): {fp}")
    print(f"False Negatives (Fraud Loss): {fn}")
    
    return auprc

def generate_shap_explanations(model, X_sample):
    """Provides actionable insights for business teams"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample)