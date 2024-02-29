import json
import os
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from evaluate import load
from transformers import set_seed

def compute_f1_scores(df, thresholds=np.arange(0, 1.01, 0.01), model_name="llava"):

    f1_scores = []

    for threshold in thresholds:
        # Adjust predictions: If expl_score <= threshold, flip label_pred to ensure it's counted as incorrect
        # For binary classification, assuming labels are 0 and 1
        df['adjusted_label_pred'] = np.where(df['bertscore'] <= threshold, 1 - df['label_true'], df['label_pred'])
        
        # Compute F1 score with adjusted predictions
        f1 = f1_score(df['label_true'], df['adjusted_label_pred'], average="macro")
        
        f1_scores.append((model_name, threshold, f1))
    
    scores_df = pd.DataFrame(f1_scores, columns=['model_name', 'threshold', 'f1'])

    return scores_df

def get_f1_and_bertscore(pred, true, model_name="llava"):

    set_seed(42)
    eval_df = pred.merge(true, on='id', how='left', suffixes=('_pred', '_true')) 
    # fill NA preds with incorrect label
    eval_df['label_pred'] = eval_df['label_pred'].fillna(1 - eval_df['label_true']).astype(int)

    bertscore = load("bertscore")
    results = bertscore.compute(predictions=eval_df['explanation_pred'], 
                                references=eval_df['explanation_true'],
                                model_type="microsoft/deberta-xlarge-mnli", 
                                lang="en")
    eval_df['bertscore'] = results['f1']
    bscore = eval_df['bertscore'].mean()
    print('BERTScore:', bscore)

    f1_df = compute_f1_scores(eval_df, model_name=model_name)

    print('f1@0:', f1_df[f1_df['threshold'] == 0]['f1'].values[0])
    print('f1@50:', f1_df[f1_df['threshold'] == 0.5]['f1'].values[0])
    print('f1@60:', f1_df[f1_df['threshold'] == 0.6]['f1'].values[0])
    print('f1@100:', f1_df[f1_df['threshold'] == 1]['f1'].values[0])

    f10 = f1_df[f1_df['threshold'] == 0]['f1'].values[0]
    f150 = f1_df[f1_df['threshold'] == 0.5]['f1'].values[0]
    f160 = f1_df[f1_df['threshold'] == 0.6]['f1'].values[0]

    scores = {
        'ac': f10,
        'acc': f150,
        'accc': f160
    }
    print(scores)

    score_dir = '/app/output'
    with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
        score_file.write(json.dumps(scores))

    return f1_df

reference_dir = os.path.join('/app/input', 'ref')
prediction_dir = os.path.join('/app/input', 'res')

print('Reading prediction')
true = pd.read_csv(os.path.join(reference_dir, 'v-flute_test_noimage_v2.csv'))
pred = pd.read_csv(os.path.join(prediction_dir, 'pred.csv'))

# scores = {
#     'ac': 1,
#     'acc': 2,
#     'accc': 3,
#     'accuracy': 4
# }
# print(scores)
# score_dir = '/app/output'
# with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
#     score_file.write(json.dumps(scores))

print('Checking Accuracy')
get_f1_and_bertscore(pred, true)