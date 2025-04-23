import pandas as pd
import jieba
from collections import Counter
import re
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np # Needed for TF-IDF calculation

# --- Configuration ---
# TODO: Replace with the actual path to your data file
DATA_FILE_PATH = 'projects/default_cause_analysis/sample_credit_defaults.csv' # Using sample data
# TODO: Replace with the actual name of the column containing default descriptions
TEXT_COLUMN_NAME = 'default_description_chinese' # Column name from sample data
# TODO: Adjust the number of top words to display
TOP_N_WORDS = 50 # Reduced for sample data clarity, adjust as needed
# TODO: Add or modify Chinese stop words as needed
# You can find comprehensive stop word lists online, e.g., search for "chinese stop words list"
CHINESE_STOP_WORDS = set([
    '的', '了', '是', '我', '你', '他', '她', '它', '们', '这', '那', '之', '与', '和', '或',
    '在', '于', '从', '向', '到', '以', '因', '为', '就', '而', '但', '不', '都', '也',
    '很', '最', '多', '少', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    '个', '只', '被', '把', '对', '给', '让', '请', '使', '得', '着', '过', '啊', '哦',
    '嗯', '吧', '吗', '呢', '啥', '什么', '怎么', '哪里', '谁', '时候', '没有', '如果',
    '所以', '但是', '并且', '或者', '还是', '因为', '我们', '你们', '他们', '这个',
    '那个', '这些', '那些', '一些', '这种', '那样', '然后', '现在', '后来', '可能',
    '可以', '需要', '必须', '应该', '能够', '觉得', '知道', '表示', '进行', '问题',
    '情况', '公司', '客户', '电话', '联系', '款', '项', '元', '钱', '金额', '日', '月',
    '年', '号', '人', '说', '导致', '由于', '原因', '无法', '已经', '目前', '暂时',
    # Added from sample context, review for your actual data
    '还款', '影响', '资金' , '工作', '中', '后'
])

# --- Functions ---

def segment_and_filter(text, stop_words):
    """Segments Chinese text, removes punctuation/numbers, and filters stop words/single chars."""
    if pd.isna(text):
        return []
    # Keep only Chinese characters
    text_chinese_only = re.sub(r'[^一-龥]', '', str(text))
    segmented_words = jieba.lcut(text_chinese_only)
    # Filter stop words and single characters
    filtered_words = [
        word for word in segmented_words
        if word not in stop_words and len(word) > 1
    ]
    return filtered_words

def calculate_word_scores(series, stop_words):
    """
    Calculates word frequency and summed TF-IDF scores for words in a pandas Series.
    Returns a dictionary: {word: {'frequency': count, 'tfidf_sum': score}}
    """
    # Process each document (text entry)
    processed_docs_list = [] # List of lists of words for frequency counting
    processed_docs_str = []  # List of space-separated strings for TF-IDF

    for text in series.dropna():
        filtered_words = segment_and_filter(text, stop_words)
        if filtered_words:
            processed_docs_list.append(filtered_words)
            processed_docs_str.append(" ".join(filtered_words)) # Join for TfidfVectorizer

    if not processed_docs_str:
        return {}, {} # Return empty if no processable documents found

    # 1. Calculate Frequency
    all_words_flat = [word for sublist in processed_docs_list for word in sublist]
    word_frequencies = Counter(all_words_flat)

    # 2. Calculate TF-IDF
    vectorizer = TfidfVectorizer(analyzer='word',
                                 tokenizer=lambda x: x.split(), # Already tokenized
                                 preprocessor=lambda x: x,      # Already preprocessed
                                 token_pattern=None)            # Use tokenizer
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_docs_str)
        feature_names = vectorizer.get_feature_names_out()
        # Sum TF-IDF scores for each term across all documents
        tfidf_scores_sum = np.asarray(tfidf_matrix.sum(axis=0)).ravel() # Sum columns

        # Map feature names (words) to their summed TF-IDF scores
        word_tfidf_sums = {word: score for word, score in zip(feature_names, tfidf_scores_sum)}

    except ValueError as e:
        # Handle case where vocabulary might be empty after filtering
        print(f"TF-IDF Calculation Warning: {e}. TF-IDF scores will be empty.")
        word_tfidf_sums = {}


    # Combine Frequency and TF-IDF scores
    combined_scores = {}
    all_unique_words = set(word_frequencies.keys()) | set(word_tfidf_sums.keys())

    for word in all_unique_words:
        combined_scores[word] = {
            'frequency': word_frequencies.get(word, 0),
            'tfidf_sum': word_tfidf_sums.get(word, 0.0)
        }

    return combined_scores


# --- Main Execution ---

if __name__ == "__main__":
    try:
        # Load the data
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"Successfully loaded data from {DATA_FILE_PATH}")
        print(f"DataFrame shape: {df.shape}")

        # Check if the specified column exists
        if TEXT_COLUMN_NAME not in df.columns:
            raise ValueError(
                f"Column '{TEXT_COLUMN_NAME}' not found in the CSV file. "
                f"Available columns are: {', '.join(df.columns)}"
            )

        # Get the text data series
        text_series = df[TEXT_COLUMN_NAME]
        print(f"Processing column: '{TEXT_COLUMN_NAME}'")

        # Initialize jieba
        print("Initializing jieba...")
        # Suppress verbose logging from jieba initialization if desired
        # import logging
        # jieba.setLogLevel(logging.INFO)
        jieba.initialize()

        # Calculate word scores (Frequency and TF-IDF)
        print(f"Calculating word scores (Frequency and TF-IDF Sum)...")
        word_scores = calculate_word_scores(text_series, CHINESE_STOP_WORDS)

        if not word_scores:
             print("No words found after filtering. Exiting.")
        else:
            # Sort by Frequency
            sorted_by_freq = sorted(word_scores.items(), key=lambda item: item[1]['frequency'], reverse=True)

            # Sort by TF-IDF Sum
            sorted_by_tfidf = sorted(word_scores.items(), key=lambda item: item[1]['tfidf_sum'], reverse=True)

            # Print the results
            print(f"--- Top {TOP_N_WORDS} Words by Frequency ---")
            print("(Word: Frequency | TF-IDF Sum)")
            for word, scores in sorted_by_freq[:TOP_N_WORDS]:
                print(f"{word}: {scores['frequency']} | {scores['tfidf_sum']:.4f}")

            print(f"--- Top {TOP_N_WORDS} Words by TF-IDF Sum ---")
            print("(Word: TF-IDF Sum | Frequency)")
            for word, scores in sorted_by_tfidf[:TOP_N_WORDS]:
                 print(f"{word}: {scores['tfidf_sum']:.4f} | {scores['frequency']}")


            # Suggestion for regex
            print("--- Regex Suggestion ---")
            print("Use the frequent words (or high TF-IDF words) above to build regex patterns.")
            print("High frequency words are common, high TF-IDF words might be more specific.")
            print("Example: If '失业' (unemployed) is high frequency and '疫情' (epidemic) has high TF-IDF,")
            print("you might create patterns like r'失业' or r'疫情.*(停工|收入)'")


    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_FILE_PATH}'.")
        print("Please update the DATA_FILE_PATH variable in the script.")
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except ImportError:
        print("Error: scikit-learn library not found.")
        print("Please install it using: pip install scikit-learn pandas jieba")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

