import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#  Настройки визуализации
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")


#  1. ГЕНЕРАЦИЯ РЕАЛИСТИЧНЫХ ОБУЧАЮЩИХ ДАННЫХ
def prepare_klient_data():
    klienty = []

    for i in range(5000):  # большой датасет
        vozrast = np.random.randint(18, 70)
        pokupki = np.random.randint(0, 40)
        sredniy_chek = np.random.uniform(200, 30000)
        akcii = np.random.choice([0, 1])
        otsenka = np.random.uniform(1, 5)

        #  Реалистичная вероятность лояльности
        prob = (
            0.30 * (pokupki / 40) +
            0.40 * (sredniy_chek / 30000) +
            0.20 * (otsenka / 5) +
            0.10 * akcii
        )

        #  Добавляем шум (реалистичность)
        prob += np.random.normal(0, 0.08)

        #  Ограничиваем диапазон
        prob = max(0, min(1, prob))

        #  Генерация лояльности по вероятности
        loyal = np.random.choice([1, 0], p=[prob, 1 - prob])

        klienty.append({
            "Возраст": vozrast,
            "Покупки": pokupki,
            "Средний чек (₽)": sredniy_chek,
            "Акции": akcii,
            "Оценка": otsenka,
            "Лояльный": loyal
        })

    return pd.DataFrame(klienty)


#  2. ВИЗУАЛИЗАЦИИ
def create_visualizations(results, n_new):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Анализ {n_new} клиентов", fontsize=16, weight='bold')

    # Круговая диаграмма
    counts = results['Статус'].value_counts()
    colors = ['#4CAF50' if s == 'Лояльный' else '#FF9800' for s in counts.index]
    axes[0].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[0].set_title("Распределение клиентов")

    # Гистограмма среднего чека
    axes[1].hist(results['Средний чек (₽)'], bins=10, color='#2196F3', alpha=0.7, edgecolor='black')
    axes[1].axvline(results['Средний чек (₽)'].mean(), color='red', linestyle='dashed')
    axes[1].set_title("Распределение среднего чека (₽)")
    axes[1].set_xlabel("Средний чек (₽)")
    axes[1].set_ylabel("Количество клиентов")

    plt.tight_layout()
    plt.show()


#  3. РЕКОМЕНДАЦИИ
def print_recommendations(results, n_new):
    risk_count = (results['Статус'] == 'В зоне риска').sum()

    print(f"\n ОБЩАЯ СТАТИСТИКА:")
    print(f"Всего проанализировано: {n_new} клиентов")
    print(f"В зоне риска: {risk_count} ({risk_count/n_new:.1%})")

    print(f"\n РЕКОМЕНДАЦИИ:")
    if risk_count == 0:
        print(" Отличные показатели! Все клиенты лояльны.")
        return

    print(f" Требуется внимание к {risk_count} клиентам:")
    risk_clients = results[results['Статус'] == 'В зоне риска']

    for idx, client in risk_clients.iterrows():
        print(f"\n    Клиент {client['ID']}:")
        issues = []

        if client['Покупки'] < 3:
            issues.append("мало покупок")
        if client['Средний чек (₽)'] < 2000:
            issues.append("низкий средний чек")
        if client['Оценка'] < 3:
            issues.append("низкая оценка")

        if issues:
            print(f"      • Проблемы: {', '.join(issues)}")
            print("      • Рекомендуется: скидка 5–10%, бонусы, персональное предложение")
        else:
            print("      • Общая низкая активность")


#  4. ОСНОВНАЯ ПРОГРАММА
print(" ML-СИСТЕМА АНАЛИЗА ЛОЯЛЬНОСТИ КЛИЕНТОВ (₽)")

data = prepare_klient_data()
print(f" Обучающих данных: {len(data)} записей")

X = data[['Возраст', 'Покупки', 'Средний чек (₽)', 'Акции', 'Оценка']]
y = data['Лояльный']

#  Обучение модели
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f" Модель обучена. Реалистичная точность: {accuracy:.1%}")

#  Анализ новых клиентов
try:
    n_new = int(input("\nСколько клиентов вы хотите проанализировать? "))
    new_clients = []

    for i in range(n_new):
        print(f"\n Клиент {i+1}")
        vozrast = int(input("Возраст: "))
        pokupki = int(input("Количество покупок: "))
        sredniy_chek = float(input("Средний чек (₽): "))
        akcii = int(input("Участвовал в акциях? (1 — да, 0 — нет): "))
        otsenka = float(input("Оценка (1-5): "))

        new_clients.append([vozrast, pokupki, sredniy_chek, akcii, otsenka])

    new_df = pd.DataFrame({
        'Возраст': [c[0] for c in new_clients],
        'Покупки': [c[1] for c in new_clients],
        'Средний чек (₽)': [c[2] for c in new_clients],
        'Акции': [c[3] for c in new_clients],
        'Оценка': [c[4] for c in new_clients],
        'ID': [i+1 for i in range(n_new)]
    })

    probs = model.predict_proba(new_df[['Возраст', 'Покупки', 'Средний чек (₽)', 'Акции', 'Оценка']])[:, 1]
    statuses = ["Лояльный" if p >= 0.5 else "В зоне риска" for p in probs]

    results = pd.DataFrame({
        'ID': new_df['ID'],
        'Возраст': new_df['Возраст'],
        'Покупки': new_df['Покупки'],
        'Средний чек (₽)': new_df['Средний чек (₽)'],
        'Акции': new_df['Акции'],
        'Оценка': new_df['Оценка'],
        'Вероятность лояльности': [f"{p:.1%}" for p in probs],
        'Статус': statuses
    })

    print("\n" + " РЕЗУЛЬТАТЫ АНАЛИЗА".center(80))
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)

    create_visualizations(results, n_new)
    print_recommendations(results, n_new)

except Exception as e:
    print(f" Ошибка: {e}")

print("\n" + "=" * 60)
print(" Анализ завершен")
print("=" * 60)
