import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class Bernoulli:

    def train_naive_bayes(self, df, alpha=1):
        ham_data = df[df['label'] == 0]
        spam_data = df[df['label'] == 1]
        
        total_docs = len(df)
        ham_count = len(ham_data)
        spam_count = len(spam_data)
        
        log_p_ham = np.log(ham_count / total_docs)
        log_p_spam = np.log(spam_count / total_docs)
        
        vocabulary = [col for col in df.columns if col != 'label']
        
        log_likelihood_ham = {}
        log_likelihood_spam = {}
        for word in vocabulary:
            count_word_ham = ham_data[word].sum()
            count_word_spam = spam_data[word].sum()
            
            p_word_ham = (count_word_ham + alpha) / (ham_count + 2 * alpha)
            p_word_spam = (count_word_spam + alpha) / (spam_count + 2 * alpha)
            
            # Store log probabilities
            log_likelihood_ham[word] = np.log(p_word_ham)
            log_likelihood_spam[word] = np.log(p_word_spam)

        model = {
            'log_p_ham': log_p_ham,
            'log_p_spam': log_p_spam,
            'log_likelihood_ham': log_likelihood_ham,
            'log_likelihood_spam': log_likelihood_spam,
            'vocabulary': vocabulary
        }
        
        return model

    def classify(self, document, model, vectorizer):

        doc_vector = vectorizer.transform([document]).toarray()[0]
        vocabulary = model['vocabulary']
        
        log_prob_ham = model['log_p_ham']
        log_prob_spam = model['log_p_spam']
        
        for i, word in enumerate(vocabulary):
            # Get the log likelihood of the word for ham and spam
            log_word_ham = model['log_likelihood_ham'][word]
            log_word_spam = model['log_likelihood_spam'][word]
            if doc_vector[i] == 1:
                log_prob_ham += log_word_ham
                log_prob_spam += log_word_spam
            else:
                log_prob_ham += np.log(1 - np.exp(log_word_ham))
                log_prob_spam += np.log(1 - np.exp(log_word_spam))
        
        return 1 if log_prob_spam > log_prob_ham else 0

    def get_bernoulli(self, folder_path, label):
        total_lines = []
        file_names = []
        
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, mode='r', encoding='utf-8', errors='replace') as f:
                text = f.read()
                total_lines.append(text)
            file_names.append(filename)
        
        vectorizer = CountVectorizer(binary=True)
        word_counts = vectorizer.fit_transform(total_lines)
        
        valid_words = []
        valid_indices = []
        for i, word in enumerate(vectorizer.get_feature_names_out()):
            if not any(char.isdigit() for char in word):
                valid_words.append(word)
                valid_indices.append(i)
        
        filtered_word_counts = word_counts[:, valid_indices]
        
        df = pd.DataFrame(filtered_word_counts.toarray(), index=file_names, columns=valid_words)
        df['label'] = label
        return df, len(file_names)

    def testing(self, test_folder, model):
        all_docs = []
        ham_path = os.path.join(test_folder, 'ham')
        spam_path = os.path.join(test_folder, 'spam')
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
        
        for filename in os.listdir(ham_path):
            filepath = os.path.join(ham_path, filename)
            with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
                document = f.read()
            prediction = self.classify(document, model, vectorizer)
            if prediction == 0: 
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
            if prediction == 1:  # correctly classified as spam
                correct += 1
                spam_correct += 1
                spam_predicted += 1
            total += 1
        
        spam_total = len(os.listdir(spam_path))
        return {
            'correct': correct,
            'total': total,
            'spam_predicted': spam_predicted,
            'spam_correct': spam_correct,
            'spam_total': spam_total
        }

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

    def test_bernoulli(self):
        # --------------------
        # Training Data Loading
        # --------------------
        # Enron1 training
        folder_path = os.path.join('.', 'enron1_train', 'enron1', 'train')
        ham_df, ham_count = self.get_bernoulli(os.path.join(folder_path, 'ham'), label=0)
        spam_df, spam_count = self.get_bernoulli(os.path.join(folder_path, 'spam'), label=1)
        enron1_df = pd.concat([ham_df, spam_df])
        enron1_df.to_csv('enron1_bernoulli_train.csv', index=False)
        
        # Enron2 training
        folder_path = os.path.join('.', 'enron2_train', 'train')
        ham2_df, ham2_count = self.get_bernoulli(os.path.join(folder_path, 'ham'), label=0)
        spam2_df, spam2_count = self.get_bernoulli(os.path.join(folder_path, 'spam'), label=1)
        enron2_df = pd.concat([ham2_df, spam2_df])
        enron2_df.to_csv('enron2_bernoulli_train.csv', index=False)
        
        # Enron4 training
        folder_path = os.path.join('.', 'enron4_train', 'enron4', 'train')
        ham4_df, ham4_count = self.get_bernoulli(os.path.join(folder_path, 'ham'), label=0)
        spam4_df, spam4_count = self.get_bernoulli(os.path.join(folder_path, 'spam'), label=1)
        enron4_df = pd.concat([ham4_df, spam4_df])
        enron4_df.to_csv('enron4_bernoulli_train.csv', index=False)
        
        print("Training Dataset statistics:")
        print(f"Enron1 - Ham: {ham_count}, Spam: {spam_count}")
        print(f"Enron2 - Ham: {ham2_count}, Spam: {spam2_count}")
        print(f"Enron4 - Ham: {ham4_count}, Spam: {spam4_count}")
        
        # Combine all training datasets and fill missing values with 0
        combined_df = pd.concat([ham_df, spam_df, ham2_df, spam2_df, ham4_df, spam4_df]).fillna(0)
        
        model = self.train_naive_bayes(combined_df)
    
        result1 = self.testing(os.path.join('.', 'enron1_test', 'enron1', 'test'), model)
        result2 = self.testing(os.path.join('.', 'enron2_test', 'test'), model)
        result3 = self.testing(os.path.join('.', 'enron4_test', 'enron4', 'test'), model)
        
        metrics1 = self.calculate_metrics(result1)
        metrics2 = self.calculate_metrics(result2)
        metrics3 = self.calculate_metrics(result3)
        
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
        
        total_correct = result1['correct'] + result2['correct'] + result3['correct']
        total_count = result1['total'] + result2['total'] + result3['total']
        overall_accuracy = total_correct / total_count if total_count else 0
        print(f"\nCombined Test Accuracy: {overall_accuracy:.4f}")
        
        # Save test CSV files using relative paths
        folder_path = os.path.join('.', 'enron1_test', 'enron1', 'test')
        ham_df, ham_count = self.get_bernoulli(os.path.join(folder_path, 'ham'), label=0)
        spam_df, spam_count = self.get_bernoulli(os.path.join(folder_path, 'spam'), label=1)
        enron1_df = pd.concat([ham_df, spam_df])
        enron1_df.to_csv('enron1_bernoulli_test.csv', index=False)
        
        folder_path = os.path.join('.', 'enron2_test', 'test')
        ham_df, ham_count = self.get_bernoulli(os.path.join(folder_path, 'ham'), label=0)
        spam_df, spam_count = self.get_bernoulli(os.path.join(folder_path, 'spam'), label=1)
        enron2_df = pd.concat([ham_df, spam_df])
        enron2_df.to_csv('enron2_bernoulli_test.csv', index=False)
        
        folder_path = os.path.join('.', 'enron4_test', 'enron4', 'test')
        ham_df, ham_count = self.get_bernoulli(os.path.join(folder_path, 'ham'), label=0)
        spam_df, spam_count = self.get_bernoulli(os.path.join(folder_path, 'spam'), label=1)
        enron4_df = pd.concat([ham_df, spam_df])
        enron4_df.to_csv('enron4_bernoulli_test.csv', index=False)

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

        folder_path = os.path.join('.', 'enron1_train', 'enron1', 'train')
        ham_df, ham_count = self.get_bernoulli(os.path.join(folder_path, 'ham'), label=0)
        spam_df, spam_count = self.get_bernoulli(os.path.join(folder_path, 'spam'), label=1)
        enron1_df = pd.concat([ham_df, spam_df])
        
        # Enron2 training
        folder_path = os.path.join('.', 'enron2_train', 'train')
        ham2_df, ham2_count = self.get_bernoulli(os.path.join(folder_path, 'ham'), label=0)
        spam2_df, spam2_count = self.get_bernoulli(os.path.join(folder_path, 'spam'), label=1)
        enron2_df = pd.concat([ham2_df, spam2_df])
        
        # Enron4 training
        folder_path = os.path.join('.', 'enron4_train', 'enron4', 'train')
        ham4_df, ham4_count = self.get_bernoulli(os.path.join(folder_path, 'ham'), label=0)
        spam4_df, spam4_count = self.get_bernoulli(os.path.join(folder_path, 'spam'), label=1)
        enron4_df = pd.concat([ham4_df, spam4_df])
        
        print("Training Dataset statistics:")
        print(f"Enron1 - Ham: {ham_count}, Spam: {spam_count}")
        print(f"Enron2 - Ham: {ham2_count}, Spam: {spam2_count}")
        print(f"Enron4 - Ham: {ham4_count}, Spam: {spam4_count}")
        
        # Combine all training datasets and fill missing values with 0
        combined_df = pd.concat([ham_df, spam_df, ham2_df, spam2_df, ham4_df, spam4_df]).fillna(0)
        
        best_lambda = self.tune_mcap_hyperparameters(combined_df)
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
        
        # Test on each dataset
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
                if prediction == 0:  
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
                if prediction == 1:  
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
            print(f"\n{name} Logistic Regression Results:")
            print(f"Accuracy:   {result['accuracy']:.4f}")
            print(f"Precision: {result['precision']:.4f}")
            print(f"Recall:    {result['recall']:.4f}")
            print(f"F1 Score:  {result['f1']:.4f}")
