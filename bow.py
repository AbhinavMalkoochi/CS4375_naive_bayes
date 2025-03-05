from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Bow:

    def get_bow(self, folder_path, label):
        total_lines = []
        file_names = []
        
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
                total_lines.append(f.read())
            file_names.append(filename)
        
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
        
        filtered_word_counts = word_counts[:, valid_indices]

        df = pd.DataFrame(filtered_word_counts.toarray(), index=file_names, columns=valid_words)
        df['label'] = label
        
        return df, len(file_names)

    def train_naive_bayes(self, combined_df):
        prior_spam = len(combined_df[combined_df['label'] == 1]) / len(combined_df)
        prior_ham = len(combined_df[combined_df['label'] == 0]) / len(combined_df)
        
        vocabulary = [col for col in combined_df.columns if col != 'label']
        
        spam_word_counts = combined_df[combined_df['label'] == 1][vocabulary].sum()
        ham_word_counts = combined_df[combined_df['label'] == 0][vocabulary].sum()
        
        total_spam_words = spam_word_counts.sum()
        total_ham_words = ham_word_counts.sum()
        
        alpha = 1  
        vocab_size = len(vocabulary)
        
        log_prob_word_given_spam = {}
        log_prob_word_given_ham = {}
        
        for word in vocabulary:
            prob_word_given_spam = np.log((spam_word_counts[word] + alpha) / (total_spam_words + alpha * vocab_size))
            prob_word_given_ham = np.log((ham_word_counts[word] + alpha) / (total_ham_words + alpha * vocab_size))
            
            log_prob_word_given_spam[word] = prob_word_given_spam
            log_prob_word_given_ham[word] = prob_word_given_ham
        
        log_prior_spam = np.log(prior_spam)
        log_prior_ham = np.log(prior_ham)
        
        return {
            'log_prob_word_given_spam': log_prob_word_given_spam,
            'log_prob_word_given_ham': log_prob_word_given_ham,
            'log_prior_spam': log_prior_spam,
            'log_prior_ham': log_prior_ham,
            'vocabulary': vocabulary,
            'total_spam': total_spam_words,
            'total_ham': total_ham_words
        }

    def classify(self, document, model, vectorizer):
        features = vectorizer.transform([document]).toarray()[0]
        
        log_score_spam = model['log_prior_spam']
        log_score_ham = model['log_prior_ham']
        
        for i, word in enumerate(model['vocabulary']):
            if features[i] > 0:  
                log_score_spam += model['log_prob_word_given_spam'][word]
                log_score_ham += model['log_prob_word_given_ham'][word]
        
        return 1 if log_score_spam > log_score_ham else 0

    def testing(self, test_folder, model):
        # Build paths using os.path.join for clarity
        ham_path = os.path.join(test_folder, 'ham')
        spam_path = os.path.join(test_folder, 'spam')
        
        all_docs = []
        for filename in os.listdir(ham_path):
            filepath = os.path.join(ham_path, filename)
            with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
                all_docs.append(f.read())
        
        for filename in os.listdir(spam_path):
            filepath = os.path.join(spam_path, filename)
            with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
                all_docs.append(f.read())
        
        vectorizer = CountVectorizer(vocabulary=model['vocabulary'])
        vectorizer.fit(all_docs)
        
        correct = 0
        total = 0
        spam_predicted = 0
        spam_correct = 0
        
        # Test on ham emails
        for filename in os.listdir(ham_path):
            filepath = os.path.join(ham_path, filename)
            with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
                document = f.read()
            prediction = self.classify(document, model, vectorizer)
            if prediction == 0:  # Correctly classified as ham
                correct += 1
            else:
                spam_predicted += 1
            total += 1
        
        # Test on spam emails
        for filename in os.listdir(spam_path):
            filepath = os.path.join(spam_path, filename)
            with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
                document = f.read()
            prediction = self.classify(document, model, vectorizer)
            if prediction == 1:  # Correctly classified as spam
                correct += 1
                spam_correct += 1
                spam_predicted += 1
            total += 1
        
        spam_total = len(os.listdir(spam_path))
        return {'correct': correct, 'total': total, 'spam_predicted': spam_predicted, 'spam_correct': spam_correct, 'spam_total': spam_total}

    def calculate_metrics(self, result):
        total = result['total']
        correct = result['correct']
        spam_predicted = result['spam_predicted']
        spam_correct = result['spam_correct']
        spam_total = result['spam_total']
        
        accuracy = correct / total if total else 0
        precision = spam_correct / spam_predicted if spam_predicted else 0
        recall = spam_correct / spam_total if spam_total else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) else 0
        return accuracy, precision, recall, f1

    def test_bow(self):
        # Load data from different datasets
        # Enron1 training
        folder_path = os.path.join('.', 'enron1_train', 'enron1', 'train')
        ham_df, ham_count = self.get_bow(os.path.join(folder_path, 'ham'), label=0)
        spam_df, spam_count = self.get_bow(os.path.join(folder_path, 'spam'), label=1)
        enron1_df = pd.concat([ham_df, spam_df])
        enron1_df.to_csv('enron1_bow_train.csv', index=False)
        
        # Enron2 training
        folder_path = os.path.join('.', 'enron2_train', 'train')
        ham2_df, ham2_count = self.get_bow(os.path.join(folder_path, 'ham'), label=0)
        spam2_df, spam2_count = self.get_bow(os.path.join(folder_path, 'spam'), label=1)
        enron2_df = pd.concat([ham2_df, spam2_df])
        enron2_df.to_csv('enron2_bow_train.csv', index=False)

        # Enron4 training
        folder_path = os.path.join('.', 'enron4_train', 'enron4', 'train')
        ham4_df, ham4_count = self.get_bow(os.path.join(folder_path, 'ham'), label=0)
        spam4_df, spam4_count = self.get_bow(os.path.join(folder_path, 'spam'), label=1)
        enron4_df = pd.concat([ham4_df, spam4_df])
        enron4_df.to_csv('enron4_bow_train.csv', index=False)
        
        # Print dataset statistics for training sets
        print("Training Dataset statistics:")
        print(f"Enron1 - Ham: {ham_count}, Spam: {spam_count}")
        print(f"Enron2 - Ham: {ham2_count}, Spam: {spam2_count}")
        print(f"Enron4 - Ham: {ham4_count}, Spam: {spam4_count}")
        
        # Combine all datasets for training the model
        combined_df = pd.concat([ham_df, spam_df, ham2_df, spam2_df, ham4_df, spam4_df]).fillna(0)
        
        model = self.train_naive_bayes(combined_df)
        
        result1 = self.testing(os.path.join('.', 'enron1_test', 'enron1', 'test'), model)
        result2 = self.testing(os.path.join('.', 'enron2_test', 'test'), model)
        result3 = self.testing(os.path.join('.', 'enron4_test', 'enron4', 'test'), model)
        
        # Calculate metrics for each dataset
        metrics1 = self.calculate_metrics(result1)
        metrics2 = self.calculate_metrics(result2)
        metrics3 = self.calculate_metrics(result3)
        
        # Print individual statistics
        print("\nEnron1 Test Results:")
        print(f"Accuracy: {metrics1[0]:.4f}")
        print(f"Precision: {metrics1[1]:.4f}")
        print(f"Recall: {metrics1[2]:.4f}")
        print(f"F1: {metrics1[3]:.4f}")
        
        print("\nEnron2 Test Results:")
        print(f"Accuracy: {metrics2[0]:.4f}")
        print(f"Precision: {metrics2[1]:.4f}")
        print(f"Recall: {metrics2[2]:.4f}")
        print(f"F1: {metrics2[3]:.4f}")
        
        print("\nEnron4 Test Results:")
        print(f"Accuracy: {metrics3[0]:.4f}")
        print(f"Precision: {metrics3[1]:.4f}")
        print(f"Recall: {metrics3[2]:.4f}")
        print(f"F1: {metrics3[3]:.4f}")
        
        # Optionally, if you want to print combined metrics:
        total_correct = result1['correct'] + result2['correct'] + result3['correct']
        total_count = result1['total'] + result2['total'] + result3['total']
        total_spam_predicted = result1['spam_predicted'] + result2['spam_predicted'] + result3['spam_predicted']
        total_spam_correct = result1['spam_correct'] + result2['spam_correct'] + result3['spam_correct']
        total_spam_actual = result1['spam_total'] + result2['spam_total'] + result3['spam_total']
        overall_accuracy = total_correct / total_count if total_count else 0
        overall_precision = total_spam_correct / total_spam_predicted if total_spam_predicted else 0
        overall_recall = total_spam_correct / total_spam_actual if total_spam_actual else 0
        overall_f1 = 2 * ((overall_precision * overall_recall) / (overall_precision + overall_recall)) if (overall_precision + overall_recall) else 0

        print("\nCombined Test Results:")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print(f"Overall F1: {overall_f1:.4f}")
        
        return self.bow_csv()
        
    def bow_csv(self):
        datasets = {}
        
        # Enron1 - test
        folder_path = os.path.join('.', 'enron1_test', 'enron1', 'test')
        ham_df, _ = self.get_bow(os.path.join(folder_path, 'ham'), label=0)
        spam_df, _ = self.get_bow(os.path.join(folder_path, 'spam'), label=1)
        enron1_df = pd.concat([ham_df, spam_df])
        enron1_df.to_csv('enron1_bow_test.csv', index=False)
        datasets['enron1'] = enron1_df
        
        # Enron2 - test
        folder_path = os.path.join('.', 'enron2_test', 'test')
        ham_df, _ = self.get_bow(os.path.join(folder_path, 'ham'), label=0)
        spam_df, _ = self.get_bow(os.path.join(folder_path, 'spam'), label=1)
        enron2_df = pd.concat([ham_df, spam_df])
        enron2_df.to_csv('enron2_bow_test.csv', index=False)
        datasets['enron2'] = enron2_df
        
        # Enron4 - test
        folder_path = os.path.join('.', 'enron4_test', 'enron4', 'test')
        ham_df, _ = self.get_bow(os.path.join(folder_path, 'ham'), label=0)
        spam_df, _ = self.get_bow(os.path.join(folder_path, 'spam'), label=1)
        enron4_df = pd.concat([ham_df, spam_df])
        enron4_df.to_csv('enron4_bow_test.csv', index=False)
        datasets['enron4'] = enron4_df

    def train_mcap_logistic_regression(self, df, lambda_reg=0.1, learning_rate=0.01, max_iterations=1000):
        X = df.drop('label', axis=1).values
        y = df['label'].values
        
        m, n = X.shape
        
        weights = np.zeros(n)
        bias = 0
        
        for _ in range(max_iterations):
            predictions = 1 / (1 + np.exp(-(np.dot(X, weights) + bias)))
            
            dw = (1/m) * np.dot(X.T, (predictions - y)) + (lambda_reg/m) * weights
            db = (1/m) * np.sum(predictions - y)
            weights -= learning_rate * dw
            bias -= learning_rate * db
        
        return {
            'weights': weights,
            'bias': bias,
            'vocabulary': list(df.columns[df.columns != 'label'])
        }
    
    def mcap_classify(self, document, model, vectorizer):
        doc_vector = vectorizer.transform([document]).toarray()[0]
        vocabulary = model['vocabulary']
        features = doc_vector[:len(vocabulary)]
        z = np.dot(features, model['weights']) + model['bias']
        probability = 1 / (1 + np.exp(-z))
        return 1 if probability >= 0.5 else 0
    

    
    def test_mcap_logistic_regression(self):
        # Enron1 training
        folder_path = os.path.join('.', 'enron1_train', 'enron1', 'train')
        ham_df, ham_count = self.get_bow(os.path.join(folder_path, 'ham'), label=0)
        spam_df, spam_count = self.get_bow(os.path.join(folder_path, 'spam'), label=1)
        enron1_df = pd.concat([ham_df, spam_df])
        
        # Enron2 training
        folder_path = os.path.join('.', 'enron2_train', 'train')
        ham2_df, ham2_count = self.get_bow(os.path.join(folder_path, 'ham'), label=0)
        spam2_df, spam2_count = self.get_bow(os.path.join(folder_path, 'spam'), label=1)
        enron2_df = pd.concat([ham2_df, spam2_df])
        
        # Enron4 training
        folder_path = os.path.join('.', 'enron4_train', 'enron4', 'train')
        ham4_df, ham4_count = self.get_bow(os.path.join(folder_path, 'ham'), label=0)
        spam4_df, spam4_count = self.get_bow(os.path.join(folder_path, 'spam'), label=1)
        enron4_df = pd.concat([ham4_df, spam4_df])
        
        print("Training Dataset statistics:")
        print(f"Enron1 - Ham: {ham_count}, Spam: {spam_count}")
        print(f"Enron2 - Ham: {ham2_count}, Spam: {spam2_count}")
        print(f"Enron4 - Ham: {ham4_count}, Spam: {spam4_count}")
        
        combined_df = pd.concat([ham_df, spam_df, ham2_df, spam2_df, ham4_df, spam4_df]).fillna(0)
        
        best_lambda = 0
        lambda_values = [0.01, 0.1, 1, 10]
        
        # Prepare data
        X = combined_df.drop('label', axis=1)
        y = combined_df['label']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        
        all_docs = list(train_df.drop('label', axis=1).apply(lambda row: ' '.join(row.index[row == 1]), axis=1)) + \
                   list(val_df.drop('label', axis=1).apply(lambda row: ' '.join(row.index[row == 1]), axis=1))
        vectorizer = CountVectorizer(vocabulary=list(X.columns), binary=True)
        vectorizer.fit(all_docs)
        
        best_f1 = 0
        best_lambda = 0.01
        
        for lambda_reg in lambda_values:
            model = self.train_mcap_logistic_regression(train_df, lambda_reg=lambda_reg)
            
            correct = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            for _, row in val_df.iterrows():
                document = ' '.join(row.index[row == 1])
                true_label = row['label']
                prediction = self.mcap_classify(document, model, vectorizer)
                
                if prediction == true_label:
                    correct += 1
                
                if prediction == 1 and true_label == 1:
                    true_positives += 1
                elif prediction == 1 and true_label == 0:
                    false_positives += 1
                elif prediction == 0 and true_label == 1:
                    false_negatives += 1
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_lambda = lambda_reg
        
        model = self.train_mcap_logistic_regression(combined_df, lambda_reg=best_lambda)
        
        all_test_docs = []
        test_paths = [
            os.path.join('.', 'enron1_test', 'enron1', 'test'),
            os.path.join('.', 'enron2_test', 'test'),
            os.path.join('.', 'enron4_test', 'enron4', 'test')
        ]
        
        for test_path in test_paths:
            for label_folder in ['ham', 'spam']:
                folder = os.path.join(test_path, label_folder)
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
                        all_test_docs.append(f.read())
        
        vectorizer = CountVectorizer(vocabulary=model['vocabulary'], binary=True)
        vectorizer.fit(all_test_docs)
        
        results = []
        for test_path in test_paths:
            correct = 0
            total = 0
            spam_predicted = 0
            spam_correct = 0
            
            ham_path = os.path.join(test_path, 'ham')
            for filename in os.listdir(ham_path):
                filepath = os.path.join(ham_path, filename)
                with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
                    document = f.read()
                prediction = self.mcap_classify(document, model, vectorizer)
                if prediction == 0:  # correctly classified as ham
                    correct += 1
                else:
                    spam_predicted += 1
                total += 1
            
            spam_path = os.path.join(test_path, 'spam')
            for filename in os.listdir(spam_path):
                filepath = os.path.join(spam_path, filename)
                with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
                    document = f.read()
                prediction = self.mcap_classify(document, model, vectorizer)
                if prediction == 1:  # correctly classified as spam
                    correct += 1
                    spam_correct += 1
                    spam_predicted += 1
                total += 1
            
            spam_total = len(os.listdir(spam_path))
            
            accuracy = correct / total if total else 0
            precision = spam_correct / spam_predicted if spam_predicted else 0
            recall = spam_correct / spam_total if spam_total else 0
            f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) else 0
            
            results.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        dataset_names = ['Enron1', 'Enron2', 'Enron4']
        for name, result in zip(dataset_names, results):
            print(f"\n{name} Logistic Regression results:")
            print(f"Accuracy:   {result['accuracy']:.4f}")
            print(f"Precision: {result['precision']:.4f}")
            print(f"Recall:    {result['recall']:.4f}")
            print(f"F1 Score:  {result['f1']:.4f}")
