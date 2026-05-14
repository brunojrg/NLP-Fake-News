# Fake News Detection with Classical Machine Learning and Transformers

This project applies natural language processing (NLP) to classify news headlines/articles as fake or real, comparing traditional machine learning models with transformer-based approaches on the same fake news detection task.

## Index

- [Project overview](#project-overview)
- [Dataset description](#dataset-description)
- [Research goal](#research-goal)
- [Workflow](#workflow)
- [Main findings](#main-findings)
- [Selected results](#selected-results)
- [How to reproduce](#how-to-reproduce)
- [Collaborators](#collaborators)
- [Repository structure](#repository-structure)
- [Next steps](#next-steps)
- [Notes](#notes)

## Project overview

The repository documents an end-to-end text classification workflow: dataset inspection, preprocessing, baseline model development, model optimization, transformer experiments, and final prediction generation for an unseen test set.

The project combines classical sparse-text methods such as Multinomial Naive Bayes, Random Forest, and XGBoost with transformer models including DistilBERT, BERT Tiny, and RoBERTa, making it useful both as a machine learning benchmark and as an NLP portfolio project.

## Dataset description

The training dataset used in the notebooks contains 34,152 rows and 2 columns: `label` and `text`. The label is binary (`0` for Fake News or `1` for Real News), and the raw text field contains the news content or headline used for classification.

During data quality checks, no missing values were found, but 1,946 duplicated entries were identified and removed. After duplicate removal and filtering out empty rows produced during preprocessing, the cleaned training set used in later notebooks contained 32,203 rows and 5 columns, including intermediate processed text fields such as `text_clean`, `text_no_stopwords`, and `text_lemmatized`.

The testing file contains 9,984 rows and 2 columns, with the label column initially filled with the placeholder value `2` for prediction tasks. The training and testing dataset can be found in the `dataset` folder.

## Research goal

The main goal of this project was to determine how well different NLP models can detect fake news and to compare the performance of classical machine learning models against transformer-based models on the same task.

More specifically, the project explored three questions:
- How much performance can be achieved with classical text vectorization and standard ML classifiers?
- Which optimized classical model performs best after tuning?
- Do transformer-based models outperform the best classical baseline strongly enough to justify their additional complexity?

## Workflow

### 1. Data cleaning and preprocessing

The main pipeline removes duplicates, cleans the raw text with regular expressions, keeps numbers where relevant, removes stopwords, and applies lemmatization using NLTK. The resulting processed columns are reused in downstream notebooks so that model comparisons are performed on a consistent text representation.

### 2. Feature engineering

For the classical models, text is transformed into numerical features using Bag-of-Words and TF-IDF vectorization.The Random Forest notebook tests seven vectorizer configurations, varying n-grams, vocabulary size, minimum and maximum document frequency, custom tokenization, and sublinear TF scaling.

### 3. Classical model experiments

The project evaluates Multinomial Naive Bayes, Random Forest, and XGBoost as classical NLP baselines and tuned models. The Random Forest optimization notebook uses `RandomizedSearchCV` with 20 iterations and 5-fold stratified cross-validation to tune the best-performing vectorizer setup. The `XGBoost` optimization also uses `RandomizedSearchCV`, but with 250 iterations and 5-fold cross-validation.

### 4. Transformer experiments

Three transformer notebooks evaluate `distilbert/distilbert-base-uncased`, `mrm8488/bert-tiny-finetuned-fake-news-detection`, and a `hamzab/roberta-fake-news-classification` news classification model on the same classification task. These experiments use Hugging Face tokenizers, sequence-classification models, dataset splits, and trainer-based evaluation pipelines.

### 5. Final prediction generation

The notebooks also apply the selected best model to the external testing dataset, replacing placeholder labels with predicted class labels and exporting the results as CSV files.

## Main findings

- The preprocessing pipeline substantially improved data quality by removing 1,946 duplicates and producing a final cleaned training dataset of 32,203 usable rows.

- In the Random Forest experiments, the best vectorizer among seven tested setups was a Bag-of-Words unigram configuration, which achieved a baseline weighted F1-score of 0.9152 before tuning.

- After hyperparameter tuning, the Random Forest model reached 0.9359 validation accuracy and 0.9359 weighted F1-score, making it a strong classical benchmark.

- MultinomialNB was one of the best-performing classical models, achieving 93.57% accuracy after tuning.

- For MultinomialNB, using CountVectorizer with unigrams and bigrams performed better than unigrams alone, showing that short phrases were informative for fake news detection.

- The final MultinomialNB model was well balanced across both classes, with similar precision, recall, and F1-scores for fake and real news.

- XGBoost improved substantially after hyperparameter tuning, increasing from a baseline accuracy of about 89.6% to 93.1%.

- For XGBoost, the best results came from CountVectorizer with unigram features; adding more aggressive feature restrictions or relying on bigrams reduced performance.

- Both optimized models achieved strong and balanced classification results, but MultinomialNB slightly outperformed XGBoost on this dataset.

- Random Forest and MultinomialNB were effectively tied at the top. MultinomialNB remained the most practical classical reference because it delivered almost the same accuracy with simpler, faster and more explainable modelling. **Feature representation and careful tuning mattered more than model complexity alone.**

- A plain `distilbert-base-uncased` pipeline used without proper task-specific fine-tuning performed poorly, reaching only 0.4727 accuracy, which highlights the importance of using task-adapted transformer training rather than off-the-shelf checkpoints alone.

- The fine-tuned BERT Tiny model outperformed the tuned Random Forest model, achieving 0.9668 accuracy on the validation/test split with balanced precision and recall near 0.97 for both classes.

## Selected results

| Model / Setup | Main result |
|---|---|
| Random Forest baseline + BoW unigram | Accuracy 0.9152, weighted F1 0.9152. |
| Tuned Random Forest | Accuracy 0.9359, weighted F1 0.9359.
| Tuned MultinomialNB | Accuracy 0.9357, weighted F1 0.9357. |
 |
| RoBERTa fake news classification | Accuracy 0.9858, precision/recall/F1 about 0.99 for both classes. |
| DistilBERT base pipeline (non-task-specific) | Accuracy 0.4727, much weaker than tuned models. |

The available notebooks therefore suggest that the best-performing approach in this repository is the fine-tuned RoBERTa transformer model, while the tuned Random Forest and MultinomialNB provide a strong and computationally simpler baseline.

## How to reproduce

### Requirements

The notebooks rely mainly on the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `matplotlib`
- `seaborn`
- `xgboost`
- `transformers`
- `datasets`
- `torch`

### Suggested steps

1. Create a Python environment and install the dependencies listed above.
2. Download the training and testing datasets and place them in the expected `dataset/` directory or update the file paths in the notebooks accordingly.
3. Run the notebooks in a logical order, starting with the main preprocessing/baseline notebook and then the model-specific optimization notebooks.
4. For transformer experiments, ensure access to enough RAM/VRAM or use Google Colab, which is the environment referenced in some notebooks.

Example setup:

```bash
git clone https://github.com/brunojrg/NLP-Fake-News.git
cd NLP-Fake-News
pip install pandas numpy scikit-learn nltk matplotlib seaborn xgboost transformers datasets torch jupyter
jupyter notebook
```

## Collaborators

- Bruno Gregório - https://github.com/brunojrg
- Satish Rajagopal - https://github.com/satishbabu06
- Vítor Ferraz - https://github.com/vitorferraz19

## Repository structure

```text
.
├── main.ipynb                                              # Data inspection, preprocessing, baseline workflow, and test prediction generation
├── MultinomialNB.ipynb                                     # Multinomial Naive Bayes optimization and final prediction pipeline
├── main_RF.ipynb                                           # Random Forest vectorizer experiments, tuning, comparison, and predictions
├── XGBoost_NLP.ipynb                                       # XGBoost optimization for fake news classification
├── transformers_nlp_distilbert-base-uncased.ipynb          # DistilBERT experiment
├── transformers_nlp_bert-tiny-finetuned-fake-news-detection.ipynb  # Fine-tuned BERT Tiny experiment
├── transformers_nlp_roberta-fake-news-classification.ipynb # RoBERTa experiment
├── Fake News Detection.pdf                                 # Project presentation
└── README.md                                               # Project documentation
```

## Next steps

Possible improvements for future work include:
- Add a `requirements.txt` or `environment.yml` file to make reproduction easier.
- Consolidate all model results into a single comparison table with the same evaluation metrics and split definitions.
- Add ROC-AUC, precision-recall curves, and error analysis for the strongest models.
- Further fine tune the transformer models for better predictions.
- Turn the best model into a lightweight Streamlit or Flask app for interactive fake news classification.

## Notes

This repository is a good example of a comparative NLP project because it shows both classical feature-based modeling and modern transformer-based modeling within the same problem setting. The strongest message is that the project does not stop at a baseline model: it includes preprocessing, tuning, model comparison, and out-of-sample prediction generation.
