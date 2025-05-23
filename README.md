# disaster-tweet-classification
The primary goal of this project is to apply advanced Natural Language Processing (NLP) techniques to analyse and classify tweets as either related to real disaster events or not. By leveraging a labelled dataset provided by Kaggle's "Real or Not? NLP with Disaster Tweets" competition, the project aims to simulate a real-world scenario where social media data is used for crisis response and early detection. The workflow involves cleaning and preprocessing raw tweet text by removing noise such as URLs, emojis, and stopwords, followed by transforming the cleaned text into numerical form using vectorisation methods like TF-IDF or word embeddings. Multiple machine learning models—including Logistic Regression, Naive Bayes, Support Vector Machines (SVM), and optionally LSTM networks—are trained and evaluated. The models are assessed using a variety of performance metrics, including accuracy, precision, recall, and F1-score, to determine their effectiveness in distinguishing between disaster-related and non-disaster tweets. Through this process, the project not only highlights key technical skills in NLP and classification but also demonstrates how data-driven insights can be derived from unstructured social media content to support real-time decision-making in emergency situations.

## Dataset Description

- Source: [Kaggle Dataset](https://www.kaggle.com/competitions/nlp-getting-started/data)
- Files:
  - `train.csv`: 7,613 tweets with labels (1 = disaster, 0 = not disaster)
  - `test.csv`: 3,263 tweets without labels
  - Each tweet contains:
- `id`
- `keyword`
- `location`
- `text`
- `target` (label)

## Folder Structure

```
disaster-tweet-classification/
├── data/                   # Contains train.csv and test.csv (from Kaggle, not pushed to GitHub)
├── notebooks/
│   └── NLPProject.ipynb    # Jupyter notebook with all code and results
├── src/
│   ├── preprocessing.py    # Custom text cleaning and preprocessing functions
│   ├── model.py            # Model training logic for multiple classifiers
│   └── utils.py            # Evaluation and helper functions
├── main.py   
├── requirements.txt        # Python libraries needed
├── README.md               # Project overview and instructions
└── .gitignore              # Prevents uploading unnecessary files
```

  
## How It Works:
The workflow of this project follows a structured machine learning pipeline, starting with thorough data preprocessing. This involves cleaning the tweet text by removing URLs, emojis, special characters, and converting all text to lowercase for uniformity. Tokenisation, stemming, and the removal of common stopwords are then applied using NLTK to normalise the text and reduce dimensionality. Once cleaned, the text is transformed into numerical representations using feature extraction techniques such as Bag of Words and TF-IDF vectorisation, with optional use of word embeddings for capturing contextual meaning. These features are fed into various classification models, including Logistic Regression, Naive Bayes, Support Vector Machine (SVM), and a Long Short-Term Memory (LSTM) network implemented using TensorFlow and Keras. Each model is trained and tested to assess its ability to classify tweets correctly. The performance of these models is evaluated using key metrics such as accuracy, F1 score, precision, recall, and confusion matrices, providing a comprehensive understanding of their strengths and trade-offs in identifying disaster-related tweets.

## High-Level Architecture Diagram
                +-----------------+
                |  Raw Tweets     |       ← Kaggle CSV (train.csv)
                +--------+--------+
                         |
                         v
          +-----------------------------+
          |  Data Preprocessing         |
          | - Clean text (remove noise) |
          | - Tokenize & stem           |
          | - Handle null values        |
          +-------------+---------------+
                        |
                        v
           +----------------------------+
           | Feature Engineering        |
           | - TF-IDF or Count Vector   |
           | - Word embeddings (optional)|
           +-------------+--------------+
                         |
                         v
         +-------------------------------+
         | Model Training                |
         | - Logistic Regression         |
         | - Naive Bayes / SVM / LSTM    |
         +---------------+---------------+
                         |
                         v
              +---------------------+
              | Model Evaluation    |
              | - F1, Accuracy      |
              | - Confusion Matrix  |
              +---------------------+
                         |
                         v
              +----------------------+
              | Prediction on new    |
              | tweets (optional)    |
              +----------------------+

## Architecture: 

![e57a7cfa-9c85-4087-9362-824fe97b1dd3](https://github.com/user-attachments/assets/e34e4522-0aac-46c9-a21a-b642b891e6d6)


Tools & Libraries
- Python
- Pandas, NumPy
- NLTK, Scikit-learn, TensorFlow
- Matplotlib, Seaborn

## How to Run

1. Clone this repo:
    ```
    git clone https://github.com/Varshakumar28/disaster-tweet-classification.git
    cd disaster-tweet-classification
    ```

2. Install required libraries:
    ```
    pip install -r requirements.txt
    ```

3. Download the dataset from Kaggle and place CSVs into the `data/` folder.

4. Run the notebook:
    ```
    Jupyter notebook notebooks/NLPProject.ipynb

## Result & Insight: 
The experiments revealed that Logistic Regression served as a reliable baseline model, delivering consistent and interpretable performance with minimal computational requirements. However, the LSTM model, while more resource-intensive, demonstrated improved accuracy by capturing deeper contextual relationships within the tweet text. Additionally, the analysis highlighted that specific keywords and patterns within tweets played a significant role in distinguishing disaster-related content from unrelated messages, reinforcing the importance of effective feature extraction and preprocessing in NLP-driven classification tasks.

## License
This project is licensed under the MIT License.

## Author
Varsha Medukonduru — Graduate student specialising in Business Analytics & Data Engineering.

