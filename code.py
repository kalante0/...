import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Настройки визуализации
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")


# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
def load_mall_customer_data(path='Mall_Customers_Enhanced.csv'):
    try:
        df = pd.read_csv(path)

        df = df.dropna(subset=['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 
                               'Estimated Savings (k$)', 'Credit Score', 'Loyalty Years'])

        # Переименование признаков
        df = df.rename(columns={
            'Age': 'Возраст',
            'Loyalty Years': 'Покупки',
            'Annual Income (k$)': 'Средний чек (₽)',
            'Spending Score (1-100)': 'Оценка'
        })

        # Признак "Акции" — участвовал ли в акциях (например, Luxury/Fashion)
        df['Акции'] = df['Preferred Category'].isin(['Luxury', 'Fashion']).astype(int)

        # Усложнённая логика метки "Лояльный"
        df['Лояльный'] = (
            (df['Покупки'] >= 5) &
            (df['Оценка'] >= 50) &
            (df['Средний чек (₽)'] >= 40) &
            (df['Акции'] == 1)
        ).astype(int)

        # Добавим лёгкий шум: 10% случайных инверсий
        flip_mask = np.random.rand(len(df)) < 0.1
        df.loc[flip_mask, 'Лояльный'] = 1 - df.loc[flip_mask, 'Лояльный']

        return df[['Возраст', 'Покупки', 'Средний чек (₽)', 'Акции', 'Оценка', 'Лояльный']]

    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return None


# 2. ВИЗУАЛИЗАЦИИ
def create_visualizations(results, n_new):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Анализ {n_new} клиентов", fontsize=16, weight='bold')

    counts = results['Статус'].value_counts()
    colors = ['#4CAF50' if s == 'Лояльный' else '#FF9800' for s in counts.index]
    axes[0].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[0].set_title("Распределение клиентов")

    axes[1].hist(results['Средний чек (₽)'], bins=10, color='#2196F3', alpha=0.7, edgecolor='black')
    axes[1].axvline(results['Средний чек (₽)'].mean(), color='red', linestyle='dashed')
    axes[1].set_title("Распределение среднего чека (₽)")
    axes[1].set_xlabel("Средний чек (₽)")
    axes[1].set_ylabel("Количество клиентов")

    plt.tight_layout()
    plt.show()


# 3. РЕКОМЕНДАЦИИ
def print_recommendations(results, n_new):
    risk_count = (results['Статус'] == 'В зоне риска').sum()

    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"Всего проанализировано: {n_new} клиентов")
    print(f"В зоне риска: {risk_count} ({risk_count/n_new:.1%})")

    print(f"\n🎯 РЕКОМЕНДАЦИИ:")
    if risk_count == 0:
        print("✅ Отличные показатели! Все клиенты лояльны.")
        return

    risk_clients = results[results['Статус'] == 'В зоне риска']
    for idx, client in risk_clients.iterrows():
        print(f"\n👤 Клиент {client['ID']}:")
        issues = []
        if client['Покупки'] < 3:
            issues.append("мало покупок")
        if client['Средний чек (₽)'] < 30:
            issues.append("низкий средний чек")
        if client['Оценка'] < 40:
            issues.append("низкая оценка")

        if issues:
            print(f"   • Проблемы: {', '.join(issues)}")
            print("   • Рекомендуется: скидка 5–10%, бонусы, персональное предложение")
        else:
            print("   • Общая низкая активность")


# 4. ОСНОВНАЯ ПРОГРАММА
print("💼 ML-СИСТЕМА АНАЛИЗА ЛОЯЛЬНОСТИ КЛИЕНТОВ")

data = load_mall_customer_data('Mall_Customers_Enhanced.csv')
if data is None or data.empty:
    print("❌ Не удалось загрузить данные.")
    exit()

print(f"📊 Обучающих данных: {len(data)} записей")

X = data[['Возраст', 'Покупки', 'Средний чек (₽)', 'Акции', 'Оценка']]
y = data['Лояльный']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print(f"✅ Модель обучена. Точность: {model.score(X_test, y_test):.1%}")

# Анализ новых клиентов
try:
    n_new = int(input("\nСколько клиентов вы хотите проанализировать? "))
    new_clients = []

    for i in range(n_new):
        print(f"\n👤 Клиент {i+1}")
        vozrast = int(input("Возраст: "))
        pokupki = int(input("Количество покупок: "))
        sredniy_chek = float(input("Средний чек (₽): "))
        akcii = int(input("Участвовал в акциях? (1 — да, 0 — нет): "))
        otsenka = float(input("Оценка (1–100): "))

        new_clients.append([vozrast, pokupki, sredniy_chek, akcii, otsenka])

    new_df = pd.DataFrame(new_clients, columns=['Возраст', 'Покупки', 'Средний чек (₽)', 'Акции', 'Оценка'])
    new_df['ID'] = range(1, n_new + 1)

    probs = model.predict_proba(new_df.drop(columns='ID'))[:, 1]
    statuses = ["Лояльный" if p >= 0.5 else "В зоне риска" for p in probs]

    results = new_df.copy()
    results['Вероятность лояльности'] = [f"{p:.1%}" for p in probs]
    results['Статус'] = statuses

    print("\n📋 РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)

    create_visualizations(results, n_new)
    print_recommendations(results, n_new)

except Exception as e:
    print(f"❌ Ошибка: {e}")

print("\n" + "=" * 60)
print("Анализ завершен")
print("=" * 60)
