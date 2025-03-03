from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
import numpy as np
def get_bernoulli(folder_path, label):
    total_lines = []
    file_names = []
    
    # Read all files in the specified folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, mode='r', encoding='utf-8', errors='replace') as f:
            text = f.read()
            # Uncomment the following line if you wish to print each file's content for debugging
            # print(text)
            total_lines.append(text)
        file_names.append(filename)
    
    # Use CountVectorizer with binary=True to create a Bernoulli model representation
    vectorizer = CountVectorizer(binary=True)
    word_counts = vectorizer.fit_transform(total_lines)
    
    # Filter out words that contain any digit
    valid_words = []
    valid_indices = []
    for i, word in enumerate(vectorizer.get_feature_names_out()):
        if not any(char.isdigit() for char in word):
            valid_words.append(word)
            valid_indices.append(i)
    
    filtered_word_counts = word_counts[:, valid_indices]
    
    # Create DataFrame where rows are files and columns are words
    df = pd.DataFrame(filtered_word_counts.toarray(), index=file_names, columns=valid_words)
    df['label'] = label
    #print(df)
    return df, len(file_names)

def get_bow(folder_path, label):
    total_lines = []
    file_names = []
    
    # Read all files in the specified folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
            total_lines.append(f.read())
        file_names.append(filename)
    
    # Use CountVectorizer to convert the text to word count matrix
    vectorizer = CountVectorizer()
    word_counts = vectorizer.fit_transform(total_lines)
    words = vectorizer.get_feature_names_out()

    valid_words = []
    valid_indices = []
    
    # Filter out words that contain digits
    for i, word in enumerate(words):
        if not any(char.isdigit() for char in word):
            valid_words.append(word)
            valid_indices.append(i)
    
    # Filter word counts to keep only valid words
    filtered_word_counts = word_counts[:, valid_indices]

    # Convert word counts to DataFrame
    df = pd.DataFrame(filtered_word_counts.toarray(), index=file_names, columns=valid_words)
    
    # Add label column for ham (0) or spam (1)
    df['label'] = label
    
    return df, len(file_names)

def train_naive_bayes(combined_df):
    # Split data by label
    ham_data = combined_df[combined_df['label'] == 0]
    spam_data = combined_df[combined_df['label'] == 1]
    
    # Calculate prior probabilities
    total_docs = len(combined_df)
    spam_count = len(spam_data)
    ham_count = len(ham_data)
    
    prior_spam = spam_count / total_docs
    prior_ham = ham_count / total_docs
    
    print(f"Prior probability of spam: {prior_spam:.4f}")
    print(f"Prior probability of ham: {prior_ham:.4f}")
    
    # Get vocabulary (all columns except 'label')
    vocabulary = [col for col in combined_df.columns if col != 'label']
    
    # Calculate word frequencies in spam and ham
    spam_word_counts = spam_data[vocabulary].sum()
    ham_word_counts = ham_data[vocabulary].sum()
    
    # Total word counts
    total_spam_words = spam_word_counts.sum()
    total_ham_words = ham_word_counts.sum()
    
    # Calculate conditional probabilities with Laplace smoothing
    alpha = 1  # Smoothing parameter
    vocab_size = len(vocabulary)
    
    # Log probabilities for numerical stability
    log_prob_word_given_spam = {}
    log_prob_word_given_ham = {}
    
    for word in vocabulary:
        # Apply Laplace smoothing and convert to log space
        prob_word_given_spam = np.log((spam_word_counts[word] + alpha) / (total_spam_words + alpha * vocab_size))
        prob_word_given_ham = np.log((ham_word_counts[word] + alpha) / (total_ham_words + alpha * vocab_size))
        
        log_prob_word_given_spam[word] = prob_word_given_spam
        log_prob_word_given_ham[word] = prob_word_given_ham
    
    # Convert prior probabilities to log space
    log_prior_spam = np.log(prior_spam)
    log_prior_ham = np.log(prior_ham)
    
    return {
        'log_prob_word_given_spam': log_prob_word_given_spam,
        'log_prob_word_given_ham': log_prob_word_given_ham,
        'log_prior_spam': log_prior_spam,
        'log_prior_ham': log_prior_ham,
        'vocabulary': vocabulary,
        'total_spam':total_spam_words,
        'total_ham':total_ham_words
    }

def classify(document, model, vectorizer):
    # Extract features from the document
    features = vectorizer.transform([document]).toarray()[0]
    
    # Calculate score for spam and ham
    log_score_spam = model['log_prior_spam']
    log_score_ham = model['log_prior_ham']
    
    for i, word in enumerate(model['vocabulary']):
        if features[i] > 0:  # If the word appears in the document
            log_score_spam += model['log_prob_word_given_spam'][word]
            log_score_ham += model['log_prob_word_given_ham'][word]
    
    # Classify based on higher score
    return 1 if log_score_spam > log_score_ham else 0
def testing(test_folder, model):
    
     # Test data
    
    # Create a single vectorizer for consistent feature extraction
    all_docs = []
    for filename in os.listdir(test_folder + 'ham'):
        filepath = os.path.join(test_folder + 'ham', filename)
        with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
            all_docs.append(f.read())
    
    for filename in os.listdir(test_folder + 'spam'):
        filepath = os.path.join(test_folder + 'spam', filename)
        with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
            all_docs.append(f.read())
    
    
    vectorizer = CountVectorizer(vocabulary=model['vocabulary'])
    vectorizer.fit(all_docs)
    
    # Test the model
    correct = 0
    total = 0
    spam_predicted=0
    spam_correct =0
    # Test on ham emails
    for filename in os.listdir(test_folder + 'ham'):
        filepath = os.path.join(test_folder + 'ham', filename)
        with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
            document = f.read()
        prediction = classify(document, model, vectorizer)
        if prediction == 0:  # Correctly classified as ham
            correct += 1
        else:
            spam_predicted+=1
        total += 1
    
    # Test on spam emails
    for filename in os.listdir(test_folder + 'spam'):
        filepath = os.path.join(test_folder + 'spam', filename)
        with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
            document = f.read()
        prediction = classify(document, model, vectorizer)
        if prediction == 1:  # Correctly classified as spam
            correct += 1
            spam_correct+=1
            spam_predicted+=1
        total += 1
    spam_total = len(os.listdir(test_folder + 'spam'))
    return {'correct': correct, 'total': total, 'spam_predicted':spam_predicted,'spam_correct':spam_correct, 'spam_total':spam_total}
def test_bow():
     # Load data from different datasets
    #enron1

    folder_path = 'D:/Projects/cs4375/enron1_train/enron1/train/'
    ham_df, ham_count = get_bow(folder_path + 'ham', label=0)
    spam_df, spam_count = get_bow(folder_path + 'spam', label=1)
    enron1_df = pd.concat([ham_df,spam_df])
    enron1_df.to_csv('enron1_bow_train.csv', index=False)
    
    #enron 2
    folder_path = 'D:/Projects/cs4375/enron2_train/train/'
    ham2_df, ham2_count = get_bow(folder_path + 'ham', label=0)
    spam2_df, spam2_count = get_bow(folder_path + 'spam', label=1)
    enron2_df = pd.concat([ham2_df,spam2_df])
    enron2_df.to_csv('enron2_bow_train.csv', index=False)

    #enron 4
    folder_path = 'D:/Projects/cs4375/enron4_train/enron4/train/'
    ham4_df, ham4_count = get_bow(folder_path + 'ham', label=0)
    spam4_df, spam4_count = get_bow(folder_path + 'spam', label=1)
    enron4_df = pd.concat([ham4_df,spam4_df])
    enron4_df.to_csv('enron4_bow_train.csv', index=False)

    

    # Print dataset statistics
    print(f"Dataset statistics:")
    print(f"Enron1 - Ham: {ham_count}, Spam: {spam_count}")
    print(f"Enron2 - Ham: {ham2_count}, Spam: {spam2_count}")
    print(f"Enron4 - Ham: {ham4_count}, Spam: {spam4_count}")
    
    # Combine all datasets
    combined_df = pd.concat([ham_df, spam_df, ham2_df, spam2_df, ham4_df, spam4_df])
    combined_df = combined_df.fillna(0)
    
    model = train_naive_bayes(combined_df)
    result = testing('D:/Projects/cs4375/enron1_test/enron1/test/',model)
    result2 = testing('D:/Projects/cs4375/enron2_test/test/',model)
    result3 = testing('D:/Projects/cs4375/enron4_test/enron4/test/',model)
    total_correct = result['correct'] + result2['correct'] + result3['correct']
    total_count = result['total'] + result2['total'] + result3['total']
    total_spam_predicted = result['spam_predicted'] + result2['spam_predicted'] + result3['spam_predicted']
    total_spam_correct = result['spam_correct'] + result2['spam_correct'] + result3['spam_correct']
    total_spam_actual = result['spam_total'] + result2['spam_total'] + result3['spam_total']
    accuracy = total_correct / total_count  
    precision = total_spam_correct/total_spam_predicted
    recall = total_spam_correct/ total_spam_actual 
    f1 =  2 * ((precision*recall)/(precision+recall))
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    bow_csv()
def bow_csv():
    folder_path = 'D:/Projects/cs4375/enron1_test/enron1/test/'
    ham_df, ham_count = get_bow(folder_path + 'ham', label=0)
    spam_df, spam_count = get_bow(folder_path + 'spam', label=1)
    enron1_df = pd.concat([ham_df, spam_df])
    enron1_df.to_csv('enron1_bow_test.csv', index=False)
    
    # Enron2 - test
    folder_path = 'D:/Projects/cs4375/enron2_test/test/'
    ham_df, ham_count = get_bow(folder_path + 'ham', label=0)
    spam_df, spam_count = get_bow(folder_path + 'spam', label=1)
    enron2_df = pd.concat([ham_df, spam_df])
    enron2_df.to_csv('enron2_bow_test.csv', index=False)
    
    # Enron4 - test
    folder_path = 'D:/Projects/cs4375/enron4_test/enron4/test/'
    ham_df, ham_count = get_bow(folder_path + 'ham', label=0)
    spam_df, spam_count = get_bow(folder_path + 'spam', label=1)
    enron4_df = pd.concat([ham_df, spam_df])
    enron4_df.to_csv('enron4_bow_test.csv', index=False) 
def test_bernoulli():
    folder_path = 'D:/Projects/cs4375/enron1_train/enron1/train/'
    ham_df, ham_count = get_bernoulli(folder_path + 'ham', label=0)
    spam_df, spam_count = get_bernoulli(folder_path + 'spam', label=1)
    enron1_df = pd.concat([ham_df, spam_df])
    enron1_df.to_csv('enron1_bernoulli_train.csv', index=False)
    
    # Enron2 - training
    folder_path = 'D:/Projects/cs4375/enron2_train/train/'
    ham2_df, ham2_count = get_bernoulli(folder_path + 'ham', label=0)
    spam2_df, spam2_count = get_bernoulli(folder_path + 'spam', label=1)
    enron2_df = pd.concat([ham_df, spam_df])
    enron2_df.to_csv('enron2_bernoulli_train.csv', index=False)
    
    # Enron4 - training
    folder_path = 'D:/Projects/cs4375/enron4_train/enron4/train/'
    ham4_df, ham4_count = get_bernoulli(folder_path + 'ham', label=0)
    spam4_df, spam4_count = get_bernoulli(folder_path + 'spam', label=1)
    enron4_df = pd.concat([ham_df, spam_df])
    enron4_df.to_csv('enron4_bernoulli_train.csv', index=False)
    print(f"Dataset statistics:")
    print(f"Enron1 - Ham: {ham_count}, Spam: {spam_count}")
    print(f"Enron2 - Ham: {ham2_count}, Spam: {spam2_count}")
    print(f"Enron4 - Ham: {ham4_count}, Spam: {spam4_count}")
    
    # Combine all datasets
    combined_df = pd.concat([ham_df, spam_df, ham2_df, spam2_df, ham4_df, spam4_df])
    combined_df = combined_df.fillna(0)
    
    # Train the Naive Bayes model
    model = train_naive_bayes(combined_df)
    result = testing('D:/Projects/cs4375/enron1_test/enron1/test/',model)
    result2 = testing('D:/Projects/cs4375/enron2_test/test/',model)
    result3 = testing('D:/Projects/cs4375/enron4_test/enron4/test/',model)
    total_correct = result['correct'] + result2['correct'] + result3['correct']
    total_count = result['total'] + result2['total'] + result3['total']

    accuracy = total_correct / total_count  
    print(f"Test accuracy: {accuracy:.4f}")
    folder_path = 'D:/Projects/cs4375/enron1_test/enron1/test/'
    ham_df, ham_count = get_bernoulli(folder_path + 'ham', label=0)
    spam_df, spam_count = get_bernoulli(folder_path + 'spam', label=1)
    enron1_df = pd.concat([ham_df, spam_df])
    enron1_df.to_csv('enron1_bernoulli_test.csv', index=False)
    
    # Enron2 - test
    folder_path = 'D:/Projects/cs4375/enron2_test/test/'
    ham_df, ham_count = get_bernoulli(folder_path + 'ham', label=0)
    spam_df, spam_count = get_bernoulli(folder_path + 'spam', label=1)
    enron2_df = pd.concat([ham_df, spam_df])
    enron2_df.to_csv('enron2_bernoulli_test.csv', index=False)
    
    # Enron4 - test
    folder_path = 'D:/Projects/cs4375/enron4_test/enron4/test/'
    ham_df, ham_count = get_bernoulli(folder_path + 'ham', label=0)
    spam_df, spam_count = get_bernoulli(folder_path + 'spam', label=1)
    enron4_df = pd.concat([ham_df, spam_df])
    enron4_df.to_csv('enron4_bernoulli_test.csv', index=False)
def main():
    test_bow()
    test_bernoulli()
    
   

if __name__ == "__main__":
    main()


