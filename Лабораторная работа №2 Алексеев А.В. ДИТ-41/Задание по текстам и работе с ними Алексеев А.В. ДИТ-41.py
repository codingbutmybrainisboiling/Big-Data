import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# 1. Загрузите датасет
# Предполагаем, что ваш датасет находится в файле 'SMSSpamCollection.txt'
df = pd.read_csv('SMSSpamCollection.txt', sep='\t', header=None, names=['label', 'message'])

# 2. Изучите датасет
print(df.head())
print(df.columns)

# 3. Подготовьте списки текстов и меток классов
texts = df['message'].tolist()
labels = [1 if label == 'spam' else 0 for label in df['label'].tolist()]

# 4. Получите матрицу признаков
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 5. Оцените качество классификации с помощью LogisticRegression
model = LogisticRegression(random_state=2)
scores = cross_val_score(model, X, labels, cv=10, scoring='f1')
mean_f1_score = scores.mean()
print(f'Mean F1 Score: {mean_f1_score:.2f}')

# 6. Обучите классификатор на всей выборке и сделайте прогнозы
model.fit(X, labels)
test_messages = [
    "FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! Subscribe6GB",
    "FreeMsg: Txt: claim your reward of 3 hours talk time",
    "Have you visited the last lecture on physics?",
    "Have you visited the last lecture on physics? Just buy this book and you will have all materials! Only 99$",
    "Only 99$"
]
X_test = vectorizer.transform(test_messages)
predictions = model.predict(X_test)
print('Predictions:', ' '.join(map(str, predictions)))

# 7. Измерьте f1-меру для различных ngram_range
for ngram in [(2, 2), (3, 3), (1, 3)]:
    vectorizer = CountVectorizer(ngram_range=ngram)
    X_ngram = vectorizer.fit_transform(texts)
    scores = cross_val_score(model, X_ngram, labels, cv=10, scoring='f1')
    print(f'F1 Score for ngram_range={ngram}: {scores.mean():.2f}')

# 8. Эксперимент с MultinomialNB
for ngram in [(2, 2), (3, 3), (1, 3)]:
    vectorizer = CountVectorizer(ngram_range=ngram)
    X_ngram = vectorizer.fit_transform(texts)
    model_nb = MultinomialNB()
    scores_nb = cross_val_score(model_nb, X_ngram, labels, cv=10, scoring='f1')
    print(f'F1 Score for MultinomialNB with ngram_range={ngram}: {scores_nb.mean():.2f}')

# 9. Используйте TfidfVectorizer и сравните с CountVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(texts)
scores_tfidf = cross_val_score(model, X_tfidf, labels, cv=10, scoring='f1')
print(f'TF-IDF Mean F1 Score: {scores_tfidf.mean():.2f}')

# Сравнение с CountVectorizer
count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(texts)
scores_count = cross_val_score(model, X_count, labels, cv=10, scoring='f1')
print(f'Count Vectorizer Mean F1 Score: {scores_count.mean():.2f}')

# Сравнение результатов
if scores_tfidf.mean() - scores_count.mean() > 0.01:
    print(1)  # повысилось
elif scores_tfidf.mean() - scores_count.mean() < -0.01:
    print(-1)  # понизилось
else:
    print(0)  # не изменилось
