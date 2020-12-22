# for basic mathematics operation
import numpy as np
import pandas as pd
from pandas import plotting

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# for interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff
import streamlit as st

import warnings
import math
import random

from sklearn.cluster import KMeans

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv('Mall_Customers.csv')
dat = ff.create_table(data.head())
# py.iplot(dat)
desc = ff.create_table(data.describe())
py.iplot(desc)

todo_selectbox = st.sidebar.selectbox(
    "Что вы хотите сделать?",
    ("Ознакомиться с данными", "Кластеризовать")
)

if todo_selectbox == "Ознакомиться с данными":
    visualize_selectbox = st.sidebar.selectbox(
        "Выберите метод визуализции",
        ("Общие сведения", "Andrew Curves for Gender", "Распределение по возрасту и доходу", "По gender",
         "Распределение по возрасту", "Распределение по доходу",
         "Распредленеие по оценке расходов", "Есть ли кореляция",
         "Пол и оценка расходов")
    )
    if visualize_selectbox == "Andrew Curves for Gender":
        plt.rcParams['figure.figsize'] = (15, 10)
        plotting.andrews_curves(data.drop("CustomerID", axis=1), "Gender", color=list(["lightsteelblue", "pink"]))
        plt.title('Andrew Curves for Gender', fontsize=20)
        st.markdown('Можно заметить, что линии соответствующие похожим **значениям** также имеют и схожую **форму**.')
        st.pyplot(plt)
        # plt.show()
    elif visualize_selectbox =="Общие сведения":
        st.markdown('_CustomerID_ - уникальный идентификатор пользователя')
        st.markdown('_Gender_ - пол')
        st.markdown('_Age_ - возраст')
        st.markdown('_Annual Income_ - годовой доход клиента')
        st.markdown(
            '_Spending Score_ - оценка, присвоенная торговым центром на основе поведения клиентов и характера расходов')

        st.plotly_chart(dat)
    elif visualize_selectbox == "Распределение по возрасту и доходу":
        warnings.filterwarnings('ignore')

        plt.rcParams['figure.figsize'] = (18, 8)

        plt.subplot(1, 2, 1)
        sns.set(style='darkgrid')
        sns.distplot(data['Annual Income (k$)'])
        plt.title('Distribution of Annual Income', fontsize=20)
        plt.xlabel('Range of Annual Income')
        plt.ylabel('Count')

        plt.subplot(1, 2, 2)
        sns.set(style='whitegrid')
        sns.distplot(data['Age'], color='red')
        plt.title('Distribution of Age', fontsize=20)
        plt.xlabel('Range of Age')
        plt.ylabel('Count')

        st.markdown('Визуализируется картина распределения годового дохода и возраста.'
                    'Можем сделать один вывод: **мало кто зарабатывает больше 100 долларов**. '
                    'Большинство людей зарабатывают около **50-75** долларов США. '
                    'Кроме того, мы можем сказать, что **наименьший** доход составляет около **20** долларов США. '
                    'Большинство постоянных клиентов торгового центра имеют возраст около **30-35 лет**. '
                    'В то время как возрастная группа **пожилых** граждан является **наименее частым** посетителем торгового центра. '
                    'Молодежь-это меньшее число по сравнению с людьми среднего возраста')
        st.pyplot(plt)
    elif visualize_selectbox=="По gender":
        labels = ['Female', 'Male']
        size = data['Gender'].value_counts()
        colors = ['pink', 'lightblue']
        explode = [0, 0.1]

        plt.rcParams['figure.figsize'] = (18, 8)
        plt.pie(size, colors=colors, explode=explode, labels=labels, shadow=True, autopct='%.2f%%')
        plt.title('Gender', fontsize=10)
        plt.axis('off')
        # plt.legend()
        st.markdown('Круговая диаграмма, объясняет распределение полов в торговом центре'
                    'Женщины лидируют с долей 56%, в то время как мужчины имеют долю 44%, '
                    'что является **огромным разрывом**, особенно когда популяция мужчин сравнительно выше, чем женщин.')
        st.pyplot(plt, use_container_width=True)
    elif visualize_selectbox =="Распределение по возрасту":
        plt.rcParams['figure.figsize'] = (15, 8)
        sns.countplot(data['Age'], palette='coolwarm')
        plt.title('Distribution of Age', fontsize=20)
        st.markdown('Этот график показывает более интерактивную диаграмму о распределении каждой возрастной группы в торговом центре '
                    'для получения более подробной информации о возрастной группе посетителя в торговом центре.'
                    'Возраст от **27 до 39 лет** очень часто встречается, но четкой закономерности нет, можно найти только некоторые групповые закономерности, '
                    'такие как старшие возрастные группы менее часты по сравнению с ними. '
                    'Интересный факт: в торговом центре **одинаково много** посетителей в возрасте **от 18 до 67 лет**. '
                    'Люди в возрасте **55, 56, 69, 64** лет встречаются в торговых центрах гораздо **реже**. '
                    'Люди в возрасте **32 лет** самые **частые** посетители торгового центра.')
        st.pyplot(plt)
    elif visualize_selectbox=="Распределение по доходу":
        plt.rcParams['figure.figsize'] = (20, 8)
        sns.countplot(data['Annual Income (k$)'], palette='coolwarm')
        plt.title('Distribution of Annual Income', fontsize=20)
        st.markdown('Опять же, это также диаграмма, чтобы лучше объяснить распределение каждого уровня дохода, '
                    'интересно, что в торговом центре есть клиенты c очень сопоставимой частотой с их годовым доходом '
                    'в диапазоне **от 15 до 137 тысяч** долларов США. В торговом центре больше клиентов, чей годовой'
                    ' доход составляет 54 тысячи долларов США или 78 долларов США.')
        st.pyplot(plt)

    elif visualize_selectbox=="Распредленеие по оценке расходов":
        plt.rcParams['figure.figsize'] = (20, 8)
        sns.countplot(data['Spending Score (1-100)'], palette = 'coolwarm')
        plt.title('Distribution of Spending Score', fontsize = 20)
        st.markdown('Это самая важная диаграмма с точки зрения торгового центра, так как очень важно иметь некоторую интуицию и представление о расходах клиентов, '
                    'посещающих торговый центр. На общем уровне можно сделать вывод, что большинство клиентов '
                    'имеют свой показатель расходов в диапазоне 35-60. Интересно, что есть клиенты,'
                    'у которых также есть оценка расходов 1 и оценка расходов 99, которая показывает, '
                    'что торговый центр обслуживает множество клиентов с различными потребностями и '
                    'требованиями, доступными в торговом центре.')
        st.pyplot(plt)
    elif visualize_selectbox=="Есть ли кореляция":
        plt.rcParams['figure.figsize'] = (15, 8)
        sns.heatmap(data.corr(), cmap='PuBu', annot=True)
        plt.title('Heatmap for the Data', fontsize=20)
        st.markdown('Корреляции нет')
        st.pyplot(plt)
    elif visualize_selectbox=="Пол и оценка расходов":
        plt.rcParams['figure.figsize'] = (18, 7)
        sns.boxenplot(data['Gender'], data['Spending Score (1-100)'], palette='PuBu')
        plt.title('Gender vs Spending Score', fontsize=20)
        st.markdown('**Двухвариантный анализ между гендером и оценкой расходов**,'
                    'Ясно видно, что большинство **мужчин** имеют показатель расходов **от 25 до 70** тысяч долларов США,'
                    'в то время как **женщины** имеют показатель расходов **от 35 до 75** тысяч долларов США. '
                    'что еще раз указывает на то, что _женщины-лидеры шопинга_.')
        st.pyplot(plt)
elif todo_selectbox == "Кластеризовать":
    methods_selectbox = st.sidebar.selectbox(
        "Выберите метод кластеризации",
        ("Кластеризовать методом K-means по доходу", "Кластеризовать Сетью Кохонена по доходу",
         "Кластеризовать методом K-means по возрасту", "Кластеризовать Сетью Кохонена по возрасту",
         "Иерархическая кластеризация", "K-means 3D")
    )
    if methods_selectbox=="Кластеризовать методом K-means по доходу":
        x_income = data.iloc[:, [3, 4]].values
        km = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_means = km.fit_predict(x_income)

        plt.scatter(x_income[y_means == 0, 0], x_income[y_means == 0, 1], s=100, c='pink', label='скупой')
        plt.scatter(x_income[y_means == 1, 0], x_income[y_means == 1, 1], s=100, c='yellow', label='общий')
        plt.scatter(x_income[y_means == 2, 0], x_income[y_means == 2, 1], s=100, c='cyan', label='целевой')
        plt.scatter(x_income[y_means == 3, 0], x_income[y_means == 3, 1], s=100, c='magenta', label='расточительный')
        plt.scatter(x_income[y_means == 4, 0], x_income[y_means == 4, 1], s=100, c='orange', label='осторожный')
        plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, c='blue', label='centeroid')

        plt.style.use('fivethirtyeight')
        plt.title('K Means Clustering', fontsize=20)
        plt.xlabel('Annual Income')
        plt.ylabel('Spending Score')
        plt.legend()
        plt.grid()
        st.markdown('Этот кластерный анализ дает нам очень четкое представление о различных сегментах клиентов в торговом центре. '
                    'Существует 5 сегментов клиента:')
        st.markdown('1. скупой')
        st.markdown('2. общий')
        st.markdown('3. целевой')
        st.markdown('4. расточительный ')
        st.markdown('5. осторожный')
        st.markdown( 'Эти распредления на кластеры основаны на Годовом доходе (Annual Score) и '
                    'Оценке расходов (Spending Score)')
        st.pyplot(plt)

    elif methods_selectbox=="Кластеризовать Сетью Кохонена по доходу":
        x_income = data.iloc[:, [3, 4]].values
        # случайные значения
        classes_selectbox=st.sidebar.selectbox(
            "Какое количество классов", (1,2,3,4,5,6)
        )
        epochs_selectbox=st.sidebar.slider("Epochs", 10,100)
        W = []
        K = classes_selectbox
        M = len(x_income)
        N = len(x_income[0])

        la = 0.3  # коэффициент обучения
        dla = 0.05  # уменьшение коэффициента обучения
        epochs = epochs_selectbox
        arr_for_paint = []
        run_button = st.sidebar.button(label='Run Optimization')

        # получить случайное значение для инициализирования весов
        def get_w():
            z = random.random() * (2.0 / math.sqrt(M))
            return 0.5 - (1 / math.sqrt(M)) + z


        # инициализировать веса
        for i in range(K):
            W.append([])
            for j in range(N):
                W[i].append(get_w() * 0.5)


        # расстояние между векторами
        def rho(w, x, dest):
            r = 0
            # эвклидово расстояние
            if dest == 0:
                for i in range(len(w)):
                    r = r + (w[i] - x[i]) * (w[i] - x[i])
                r = math.sqrt(r)
            # квадрат эвклидова расстояния
            elif dest == 1:
                for i in range(len(w)):
                    r = r + (w[i] - x[i]) * (w[i] - x[i])
                r = r * r
            # манхэтэнское расстояния
            elif dest == 2:
                for i in range(len(w)):
                    r = r + abs((w[i] - x[i]) * (w[i] - x[i]))
            # чэбышева
            elif dest == 3:
                for i in range(len(w)):
                    max = 0
                    r = abs(w[i] - x[i]) * (w[i] - x[i])
                    if r > max:
                        max = r
                r = max
            return r


        # поиск ближайшего вектора
        def FindNear(W, x):
            wm = W[0]
            r = rho(wm, x, 2)
            i = 0
            i_n = i
            for w in W:
                if rho(w, x, 2) < r:
                    r = rho(w, x, 2)
                    wm = w
                    i_n = i
                i = i + 1
            return (wm, i_n)


        def fit(la, x_income, W):
            Wk = []
            # начать процесс обучения
            while la >= 0:
                for k in range(epochs):
                    for x in x_income:
                        wm = FindNear(W, x)[0]
                        for i in range(len(wm)):
                            wm[i] = wm[i] + la * (x[i] - wm[i])  # корректировка весов
                            arr_for_paint.append(wm)
                la = la - dla  # уменьшение коэффициента обучения
            return wm
        fit(la, x_income, W)
        # создать классы
        Data = list()
        for i in range(len(W)):
            Data.append(list())

        # отнести исходные данные к своему классу
        DS = list()
        i = 0
        y_labels_kohonen = []
        for x in x_income:
            i_n = FindNear(W, x)[1]
            Data[i_n].append(x)
            DS.append([i_n, x_income[i]])
            y_labels_kohonen.append(i_n)
            i = i + 1
        y_means = np.array(y_labels_kohonen)
        plt.scatter(x_income[y_means == 0, 0], x_income[y_means == 0, 1], s=100, c='pink', label='скупой')
        plt.scatter(x_income[y_means == 1, 0], x_income[y_means == 1, 1], s=100, c='yellow', label='общий')
        plt.scatter(x_income[y_means == 2, 0], x_income[y_means == 2, 1], s=100, c='cyan', label='целевой')
        plt.scatter(x_income[y_means == 3, 0], x_income[y_means == 3, 1], s=100, c='magenta', label='расточительный')
        plt.scatter(x_income[y_means == 4, 0], x_income[y_means == 4, 1], s=100, c='orange', label='осторожный')

        plt.style.use('fivethirtyeight')
        plt.title('Kohonen Networks Clusterization', fontsize=20)
        plt.xlabel('Annual Income')
        plt.ylabel('Spending Score')
        plt.legend()
        plt.grid()

        st.markdown('Сеть Кохонена выделила всего 4 класса, в отличии от K-means.')
        st.markdown('1. Целевые клиенты совпадают с k-means')
        st.markdown('2. К расточительный клиентам сеть добавила еще осторожных и общих')
        st.markdown('3. К осторожным сеть отнесла скупых')
        st.markdown('4. К общим сеть отнесла 3х клиентов, которые входили в скупых в k-means')
        st.markdown('5. К скупым сеть отнесла 3х клиентов, которые входили в целевые в k-means')
        st.pyplot(plt)

    elif methods_selectbox=="Кластеризовать методом K-means по возрасту":
        x_age = data.iloc[:, [2, 4]].values
        kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
        ymeans = kmeans.fit_predict(x_age)

        plt.rcParams['figure.figsize'] = (10, 10)
        plt.title('Cluster of Ages Kmeans', fontsize=30)

        plt.scatter(x_age[ymeans == 0, 0], x_age[ymeans == 0, 1], s=100, c='pink', label='Обычные клиенты')
        plt.scatter(x_age[ymeans == 1, 0], x_age[ymeans == 1, 1], s=100, c='orange', label='Приоритетные клиенты')
        plt.scatter(x_age[ymeans == 2, 0], x_age[ymeans == 2, 1], s=100, c='lightgreen',
                    label='Молодые целевые клиенты')
        plt.scatter(x_age[ymeans == 3, 0], x_age[ymeans == 3, 1], s=100, c='red', label='Пожилые целевые клиенты')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='black')

        plt.style.use('fivethirtyeight')
        plt.xlabel('Age')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.grid()
        st.markdown('Можно разработать различные маркетинговые стратегии и политику для оптимизации расходов клиента в торговом центре.')
        st.pyplot(plt)
    elif methods_selectbox=="Кластеризовать Сетью Кохонена по возрасту":
        x_age = data.iloc[:, [2, 4]].values

        classes_selectbox = st.sidebar.selectbox(
            "Какое количество классов", (1, 2, 3, 4, 5, 6)
        )
        epochs_selectbox = st.sidebar.slider("Epochs", 10, 100)
        W = []
        K = classes_selectbox
        M = len(x_age)
        N = len(x_age[0])

        la = 0.3  # коэффициент обучения
        dla = 0.05  # уменьшение коэффициента обучения
        epochs = epochs_selectbox
        arr_for_paint = []
        run_button = st.sidebar.button(label='Run Optimization')

        # получить случайное значение для инициализирования весов
        def get_w():
            z = random.random() * (2.0 / math.sqrt(M))
            return 0.5 - (1 / math.sqrt(M)) + z


        # инициализировать веса
        for i in range(K):
            W.append([])
            for j in range(N):
                W[i].append(get_w() * 0.5)


        # расстояние между векторами
        def rho(w, x, dest):
            r = 0
            # эвклидово расстояние
            if dest == 0:
                for i in range(len(w)):
                    r = r + (w[i] - x[i]) * (w[i] - x[i])
                r = math.sqrt(r)
            # квадрат эвклидова расстояния
            elif dest == 1:
                for i in range(len(w)):
                    r = r + (w[i] - x[i]) * (w[i] - x[i])
                r = r * r
            # манхэтэнское расстояния
            elif dest == 2:
                for i in range(len(w)):
                    r = r + abs((w[i] - x[i]) * (w[i] - x[i]))
            # чэбышева
            elif dest == 3:
                for i in range(len(w)):
                    max = 0
                    r = abs(w[i] - x[i]) * (w[i] - x[i])
                    if r > max:
                        max = r
                r = max
            return r


        # поиск ближайшего вектора
        def FindNear(W, x):
            wm = W[0]
            r = rho(wm, x, 2)
            i = 0
            i_n = i
            for w in W:
                if rho(w, x, 2) < r:
                    r = rho(w, x, 2)
                    wm = w
                    i_n = i
                i = i + 1
            return (wm, i_n)


        def fit(la, x_age, W):
            Wk = []
            # начать процесс обучения
            while la >= 0:
                for k in range(epochs):
                    for x in x_age:
                        wm = FindNear(W, x)[0]
                        for i in range(len(wm)):
                            wm[i] = wm[i] + la * (x[i] - wm[i])  # корректировка весов
                            arr_for_paint.append(wm)
                la = la - dla  # уменьшение коэффициента обучения
            return wm


        fit(la, x_age, W)

        # создать классы
        Data = list()

        for i in range(len(W)):
            Data.append(list())

        # отнести исходные данные к своему классу
        DS = list()
        i = 0
        y_labels_kohonen = []
        for x in x_age:
            i_n = FindNear(W, x)[1]
            Data[i_n].append(x)
            DS.append([i_n, x_age[i]])
            y_labels_kohonen.append(i_n)
            i = i + 1
        ymeans = np.array(y_labels_kohonen)
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.title('Cluster of Ages Kohonen Networks', fontsize=30)

        plt.scatter(x_age[ymeans == 0, 0], x_age[ymeans == 0, 1], s=100, c='pink', label='Обычные клиенты')
        plt.scatter(x_age[ymeans == 1, 0], x_age[ymeans == 1, 1], s=100, c='orange', label='Приоритетные клиенты')
        plt.scatter(x_age[ymeans == 2, 0], x_age[ymeans == 2, 1], s=100, c='lightgreen',
                    label='Молодые целевые клиенты')
        plt.scatter(x_age[ymeans == 3, 0], x_age[ymeans == 3, 1], s=100, c='red', label='Пожилые целевые клиенты')

        plt.style.use('fivethirtyeight')
        plt.xlabel('Age')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.grid()
        st.markdown('Сеть Кохонена более грамотно распределила группы клиентов')
        st.pyplot(plt)
    elif methods_selectbox=="Иерархическая кластеризация":
        x = data[['Age', 'Spending Score (1-100)', 'Annual Income (k$)']].values
        km = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km.fit(x)
        labels = km.labels_
        centroids = km.cluster_centers_

        hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
        y_hc = hc.fit_predict(x)

        plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s=100, c='pink', label='Обычные клиенты')
        plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s=100, c='orange', label='Приоритетные клиенты')
        plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s=100, c='lightgreen',
                    label='Молодые целевые клиенты')
        plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s=100, c='red', label='Пожилые целевые клиенты')
        plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, c='blue', label='centeroid')

        plt.style.use('fivethirtyeight')
        plt.title('Hierarchial Clustering', fontsize=20)
        plt.xlabel('Age')
        plt.ylabel('Spending Score')
        plt.legend()
        plt.grid()
        st.pyplot(plt)
    elif methods_selectbox=="K-means 3D":
        x = data[['Age', 'Spending Score (1-100)', 'Annual Income (k$)']].values
        km = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km.fit(x)
        labels = km.labels_
        centroids = km.cluster_centers_
        data['labels'] = labels
        trace1 = go.Scatter3d(
            x=data['Age'],
            y=data['Spending Score (1-100)'],
            z=data['Annual Income (k$)'],
            mode='markers',
            marker=dict(
                color=data['labels'],
                size=10,
                line=dict(
                    color=data['labels'],
                    width=12
                ),
                opacity=0.8
            )
        )
        df = [trace1]

        layout = go.Layout(
            title='Character vs Gender vs Alive or not',
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
            scene=dict(
                xaxis=dict(title='Age'),
                yaxis=dict(title='Spending Score'),
                zaxis=dict(title='Annual Income')
            )
        )

        fig = go.Figure(data=df, layout=layout)
        st.markdown('x-age,y-spending score, z-annual income]')
        st.plotly_chart(fig)
        # py.iplot(fig)
        # fig.write_html("3d_cluster.html")