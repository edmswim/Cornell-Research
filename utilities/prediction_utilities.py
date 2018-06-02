import numpy as np

#convert the prediction prob vec into ema scores. Used for classification
def convert_preds_into_ema(preds):
    pred_final = []
    # transform the predicted probability vector into an ema score by taking the index with the highest probability
    for i in range(len(preds)):
        ema_score = np.argmax(preds[i])
        pred_final = np.append(pred_final, ema_score)
    return pred_final