#импорт библиотек
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import numpy as np
from sklearn.preprocessing import OrdinalEncoder #энкодинг
from sklearn.preprocessing import StandardScaler #масштабирование
from sklearn.tree import DecisionTreeClassifier #дерево решений
from sklearn.ensemble import RandomForestClassifier #случайный лес
from sklearn.linear_model import LogisticRegression #логистическое решение
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
import altair as alt
import time

import warnings
warnings.filterwarnings('ignore')

st.session_state['model'] = None
st.session_state['true_uploaded'] = False
st.session_state['data'] = None

 

STATE = np.random.RandomState(12345)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

st.set_page_config(
    page_title="Общий анализ",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",

)

#Чтение и обработка данных

st.title('Общий анализ прогнозирования оттока клиентов в сети отелей «Как в гостях»')

file_train = st.file_uploader("Данные для обучения модели", type = 'csv')
file_test = st.file_uploader("Данные для тестирования модели", type = 'csv')

if (file_train is not None) & (file_test is not None):

    if (file_train.name == 'hotel_train.csv') & (file_test.name == 'hotel_test.csv'):
        st.session_state['true_uploaded'] = True
        with st.spinner('Wait for it...'):
            time.sleep(5)
        st.success('Files uploaded!')

        data_train = pd.read_csv(file_train)
        data_test = pd.read_csv(file_test)

        columns = ['id', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month']
        for column in columns:
            data_train[column] = data_train[column].astype('str')
            data_test[column] = data_test[column].astype('str')
        columns = ['lead_time', 'adults', 'children', 'babies']
        for column in columns:
            data_train[column] = data_train[column].astype('int')
            data_test[column] = data_test[column].astype('int')

        d_train = data_train.drop(['id'], axis=1)
        d_test = data_test.drop(['id'], axis=1)

        d_train = d_train.drop_duplicates().reset_index(drop=True)
        d_test = d_test.drop_duplicates().reset_index(drop=True)

        def delete_spaces(df, columns):
            for column in columns:
                df[column] = df[column].str.strip()
            return df

        column_spaces = ['meal', 'reserved_room_type']
        d_train = delete_spaces(d_train, column_spaces)
        d_test = delete_spaces(d_test, column_spaces)

        # def out_boxplot(dfs, columns):
        #     count_df = len(dfs)
        #     count_columns = len(columns)
        #     fig, axs = plt.subplots(nrows= count_df, ncols= count_columns, figsize = (20,15))
        #     fig.tight_layout(h_pad=2)
        #     for row in range(count_df):
        #         for column in range(count_columns):
        # #             print(columns[column])
        #             axs[row][column].boxplot(x=columns[column], data=dfs[row])
        # #             axs[row][column].hist(x=columns[column], data=dfs[row])
        #             axs[0][column].set_xlabel(columns[column])
        #             if row == 0:
        #                 axs[row][0].set_ylabel('data_train')
        #             elif row == 1:
        #                 axs[row][0].set_ylabel('data_test')
        #     st.pyplot(fig)

        columns_numeric = d_test.columns.tolist()
        columns_object = ['arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'arrival_date_month', 'meal', 'country', 'distribution_channel', 'reserved_room_type', 'customer_type', 'is_canceled', 'is_repeated_guest']
        for column in columns_object:
            columns_numeric.remove(column)
        dfs = [d_train, d_test]

        # out_boxplot(dfs, columns_numeric)

        def df_process(df, columns):
            df_processed = df
            for column in columns:
                q25 = df_processed[column].quantile(.25)
                q75 = df_processed[column].quantile(.75)        
                q_min = q25 - 1.5*(q75-q25)
                q_max = q75 + 1.5*(q75-q25)
                df_processed = df_processed[df_processed[column] >= q_min]
                df_processed = df_processed[df_processed[column] <= q_max]
            return df_processed

        need_columns = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'total_of_special_requests', 'total_nights']

        df_train = df_process(d_train, need_columns).reset_index(drop=True)
        df_test = df_process(d_test, need_columns).reset_index(drop=True)

        dfs = [df_train, df_test]

        # out_boxplot(dfs, columns_numeric)

        hotel_train = df_train.query('adults > 0')
        hotel_test = df_test.query('adults > 0')

        columns_delete = ['arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_week_nights', 'stays_in_weekend_nights']
        hotel_train_0 = hotel_train.drop(columns_delete, axis=1)
        hotel_test_0 = hotel_test.drop(columns_delete, axis=1)

        analis_columns = ['arrival_date_month', 'is_canceled', 'reserved_room_type', 'total_nights']
        hotel_profit = hotel_test_0[analis_columns]

        # month = st.selectbox('Выбор месяца для анализа', hotel_profit['arrival_date_month'].unique())
        # st.table(hotel_profit.query('arrival_date_month == @month').head())

        koeff_df = pd.DataFrame({'arrival_date_month' : [month for month in hotel_profit['arrival_date_month'].unique()],
                                'koeff' : [1, 1, 1.2, 1.2, 1.2, 1.4, 1.4, 1.4]})
        hotel_for_profit = hotel_profit.merge(koeff_df, on='arrival_date_month', how='left')
        room_price_df = pd.DataFrame({'reserved_room_type' : [type_room for type_room in sorted(hotel_profit['reserved_room_type'].unique())],
                                'price_per_night' : [1000, 800, 600, 550, 500, 450, 350],
                                'service_price' : [400, 350, 350, 150, 150, 150, 150]})
        hotel_for_profit = hotel_for_profit.merge(room_price_df, on='reserved_room_type', how='left')

        def profit_column(row):    
            if row['is_canceled'] == 1:
                income = 0
                expenses = row['service_price'] + (row['price_per_night'] * row['koeff'])
            elif row['is_canceled'] == 0:
                income = row['price_per_night'] * row['total_nights'] * row['koeff']
                expenses = row['service_price'] * (1 + row['total_nights'] // 2)
            profit = income - expenses
            return profit

        hotel_for_profit['profit'] = hotel_for_profit.apply(profit_column, axis=1)

        hotel_for_profit['profit'].sum()
        st.header('Модель машинного обучения - Decision Tree (Дерево решений)')
        


        #кодирование признаков
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        print('до кодирования тренировочных данных')
        display(hotel_train_0.head())
        hotel_train_ordinal = pd.DataFrame(encoder.fit_transform(hotel_train_0), 
                                            columns=hotel_train_0.columns)
        print('после')
        display(hotel_train_ordinal.head())
        # hotel_test_ordinal = encoder.fit_transform(hotel_test_0)
        print('до кодирования тестовых данных')
        display(hotel_test_0.head())
        hotel_test_ordinal = pd.DataFrame(encoder.transform(hotel_test_0), columns=hotel_test_0.columns)
        print('после')
        display(hotel_test_ordinal.head())

        #деление обучающих и тестовых данных на признаки и ответы
        features_train = hotel_train_ordinal.drop(['is_canceled'], axis=1)
        target_train = hotel_train_ordinal['is_canceled']

        features_test = hotel_test_ordinal.drop(['is_canceled'], axis=1)
        target_test = hotel_test_ordinal['is_canceled']

        #масштабирование признаков
        column_delete_numeric = ['stays_in_weekend_nights', 'stays_in_week_nights',]
        for column in column_delete_numeric:
            columns_numeric.remove(column)
        numeric = columns_numeric

        scaler = StandardScaler()
        numeric = columns_numeric
        scaler.fit(features_train[numeric])

        features_train[numeric] = scaler.transform(features_train[numeric])
        features_test[numeric] = scaler.transform(features_test[numeric])

        #Обучение лучшей модели ("дерево решений") с лучшими параметрами
        model = DecisionTreeClassifier(
                        max_depth=7, random_state=STATE, 
                        class_weight='balanced', min_samples_split = 4, min_samples_leaf = 1)

        model.fit(features_train, target_train)

        st.session_state['model'] = model 

        predicted_test_tree = model.predict(features_test)
        predicted_test_tree = pd.Series(predicted_test_tree)

        predicts = [predicted_test_tree]

        models_name = ['Decision Tree']

        for i in range(len(models_name)):
            FP = 0
            FN = 0
            TP = 0
            TN = 0
            predict_need = predicts[i]
            for j in range(len(target_test)):
                if predict_need[j] == target_test[j]:
                    if predict_need[j] == 1:
                        TP += 1
                    elif predict_need[j] == 0:
                        TN += 1
                if predict_need[j] != target_test[j]:
                    if predict_need[j] == 1:
                        FP += 1
                    elif predict_need[j] == 0:
                        FN += 1
            
            sum_pt = FP+FN+TP+TN
            predict_map = {'predict_0': [TN, FN], 'predict_1': [FP, TP]}
            df_predict_map = pd.DataFrame(data=predict_map, index=['target_0', 'target_1'])
            
            predict_map_per = (
                {'predict_0': ['{:.2%}'.format(TN/sum_pt), '{:.2%}'.format(FN/sum_pt)], 'predict_1': ['{:.2%}'.format(FP/sum_pt), '{:.2%}'.format(TP/sum_pt)]}
            )
            df_predict_map_per = pd.DataFrame(data=predict_map_per, index=['target_0', 'target_1'])
            print(models_name[i])

        st.subheader('Матрица ошибок')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Абсолютные значения')
            st.table(df_predict_map)
        with col2:
            st.write('Относительные значения')
            st.table(df_predict_map_per)

        i_num = 0
        predicts_columns = ['is_canceled_tree']
        for column in predicts_columns:
            hotel_for_profit[column] = predicts[i_num]
            i_num += 1
        display(hotel_for_profit.head())

        def predicted_profit(row, col):    
            if row['is_canceled'] == 0:
                income = row['price_per_night'] * row['total_nights'] * row['koeff']
                expenses = row['service_price'] * (1 + row['total_nights'] // 2)
            elif row['is_canceled'] == 1:
                if row[col] == 0:
                    income = 0
                    expenses = row['service_price'] + (row['price_per_night'] * row['koeff'])            
                elif row[col] == 1:
                    income = 0.8 * (row['service_price'] + (row['price_per_night'] * row['koeff']))
                    expenses = row['service_price'] + (row['price_per_night'] * row['koeff'])
            profit = income - expenses
            return profit

        profit_columns = ['profit_tree']
        predicts_columns = ['is_canceled_tree']
        for i in range(len(profit_columns)):
            col = profit_columns[i]
            hotel_for_profit[col] = hotel_for_profit.apply(lambda x: predicted_profit(x, col=predicts_columns[i]), axis=1)
        #st.table(hotel_for_profit.head())

        st.write(f'Прибыль отеля - {int(hotel_for_profit["profit"].sum())} руб.')
        st.write(f"Прибыль после обучения DT - {int(hotel_for_profit['profit_tree'].sum())} руб.")
        st.write(f"Полученная выгода от обучения DT - {int(hotel_for_profit['profit_tree'].sum() - hotel_for_profit['profit'].sum() - 400000)} руб.")
        st.subheader('Полученная выгода по месяцам')
        hotel_for_profit_bymonth = hotel_for_profit.groupby('arrival_date_month')['profit_tree'].sum() - hotel_for_profit.groupby('arrival_date_month')['profit'].sum() - 400000/hotel_for_profit.groupby('arrival_date_month')['profit'].sum().count()
        hotel_for_profit_bymonth_df = pd.DataFrame(hotel_for_profit_bymonth.sort_values(ascending=False), columns=['profit'])
        st.dataframe(hotel_for_profit_bymonth_df)
        st.bar_chart(hotel_for_profit_bymonth_df)      

        hotel_test_0['is_canceled_tree'] = predicted_test_tree

        st.markdown('**портрет "ненадежного" клиента**: месяц заезда - май, без изменений параметров заказа, гражданство - Португалия,')
        st.markdown('без договора на бронирование номера, с включенным завтраком в стоимость номера,')
        st.markdown('постояльцы самих дорогих номеров, бронь номера на 1-4 ночи')

        hotel_test_0['profit'] = hotel_for_profit['profit']
        hotel_test_0['profit_tree'] = hotel_for_profit['profit_tree']

        st.session_state['data'] = hotel_test_0
    else:
        st.error('Загружены не те файлы: ' + "обучающий файл - 'hotel_train.csv', тестовый - 'hotel_test.csv'" )


#streamlit run e:/Диплом/diplom.py   
