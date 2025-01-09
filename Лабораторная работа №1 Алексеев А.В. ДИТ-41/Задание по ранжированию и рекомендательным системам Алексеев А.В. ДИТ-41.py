from collections import defaultdict

# Функция для разбора строки сессии
def parse_session(session):
    if ';' in session:
        viewed, purchased = session.split(';')
        viewed = viewed.split(',') if viewed else []
        purchased = purchased.split(',') if purchased else []
    else:
        viewed = session.split(',') if session else []
        purchased = []
    return viewed, purchased

# Загрузка данных
with open('sessions_train.txt', 'r') as file:
    train_data = file.readlines()

with open('sessions_test.txt', 'r') as file:
    test_data = file.readlines()

# Построение частот появления id
viewed_freq = defaultdict(int)
purchased_freq = defaultdict(int)

for session in train_data:
    viewed, purchased = parse_session(session.strip())
    for item in viewed:
        viewed_freq[item] += 1
    for item in purchased:
        purchased_freq[item] += 1
        
def recommend_by_viewed_freq(session, viewed_freq, k):
    viewed, _ = parse_session(session.strip())
    viewed_sorted = sorted(viewed, key=lambda x: (-viewed_freq[x], viewed.index(x)))
    return viewed_sorted[:k]

def recommend_by_purchased_freq(session, purchased_freq, k):
    viewed, _ = parse_session(session.strip())
    viewed_sorted = sorted(viewed, key=lambda x: (-purchased_freq[x], viewed.index(x)))
    return viewed_sorted[:k]

def calculate_metrics(recommendations, actual_purchases, k):
    recall = len(set(recommendations[:k]).intersection(actual_purchases)) / len(actual_purchases) if actual_purchases else 0
    precision = len(set(recommendations[:k]).intersection(actual_purchases)) / k if k > 0 else 0
    return recall, precision

def evaluate_algorithm(algorithm, data, k):
    total_recall = 0
    total_precision = 0
    count = 0

    for session in data:
        viewed, purchased = parse_session(session.strip())
        if purchased:
            recommendations = algorithm(session.strip(), k)
            recall, precision = calculate_metrics(recommendations, purchased, k)
            total_recall += recall
            total_precision += precision
            count += 1

    average_recall = total_recall / count if count > 0 else 0
    average_precision = total_precision / count if count > 0 else 0
    return average_recall, average_precision

# Оценка алгоритмов на обучающей выборке
viewed_freq_train_metrics_1 = evaluate_algorithm(lambda session, k: recommend_by_viewed_freq(session, viewed_freq, k), train_data, 1)
purchased_freq_train_metrics_1 = evaluate_algorithm(lambda session, k: recommend_by_purchased_freq(session, purchased_freq, k), train_data, 1)
viewed_freq_train_metrics_5 = evaluate_algorithm(lambda session, k: recommend_by_viewed_freq(session, viewed_freq, k), train_data, 5)
purchased_freq_train_metrics_5 = evaluate_algorithm(lambda session, k: recommend_by_purchased_freq(session, purchased_freq, k), train_data, 5)

# Оценка алгоритмов на тестовой выборке
viewed_freq_test_metrics_1 = evaluate_algorithm(lambda session, k: recommend_by_viewed_freq(session, viewed_freq, k), test_data, 1)
purchased_freq_test_metrics_1 = evaluate_algorithm(lambda session, k: recommend_by_purchased_freq(session, purchased_freq, k), test_data, 1)
viewed_freq_test_metrics_5 = evaluate_algorithm(lambda session, k: recommend_by_viewed_freq(session, viewed_freq, k), test_data, 5)
purchased_freq_test_metrics_5 = evaluate_algorithm(lambda session, k: recommend_by_purchased_freq(session, purchased_freq, k), test_data, 5)

# Вывод результатов
print("Вывод результатов: ")
print(f"Viewed Frequency - Train: AverageRecall@1={viewed_freq_train_metrics_1[0]:.2f}, AveragePrecision@1={viewed_freq_train_metrics_1[1]:.2f}")
print(f"Purchased Frequency - Train: AverageRecall@1={purchased_freq_train_metrics_1[0]:.2f}, AveragePrecision@1={purchased_freq_train_metrics_1[1]:.2f}")
print(f"Viewed Frequency - Train: AverageRecall@5={viewed_freq_train_metrics_5[0]:.2f}, AveragePrecision@5={viewed_freq_train_metrics_5[1]:.2f}")
print(f"Purchased Frequency - Train: AverageRecall@5={purchased_freq_train_metrics_5[0]:.2f}, AveragePrecision@5={purchased_freq_train_metrics_5[1]:.2f}")

print(f"Viewed Frequency - Test: AverageRecall@1={viewed_freq_test_metrics_1[0]:.2f}, AveragePrecision@1={viewed_freq_test_metrics_1[1]:.2f}")
print(f"Purchased Frequency - Test: AverageRecall@1={purchased_freq_test_metrics_1[0]:.2f}, AveragePrecision@1={purchased_freq_test_metrics_1[1]:.2f}")
print(f"Viewed Frequency - Test: AverageRecall@5={viewed_freq_test_metrics_5[0]:.2f}, AveragePrecision@5={viewed_freq_test_metrics_5[1]:.2f}")
print(f"Purchased Frequency - Test: AverageRecall@5={purchased_freq_test_metrics_5[0]:.2f}, AveragePrecision@5={purchased_freq_test_metrics_5[1]:.2f}")

