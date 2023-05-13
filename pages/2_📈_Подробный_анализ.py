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

from IPython.core.display import display, HTML
from IPython.display import display_html
import warnings
warnings.filterwarnings('ignore')

STATE = np.random.RandomState(12345)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

st.set_page_config(
    page_title="Подробный анализ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",

)

st.title('Подробный анализ прогнозирования оттока клиентов в сети отелей «Как в гостях»')

if 'model' not in st.session_state:
    st.session_state['model'] = None

if 'true_uploaded' not in st.session_state:
    st.session_state['true_uploaded'] = False

if 'data' not in st.session_state:
    st.session_state['data'] = None

def do_analis(data, columns):
    st.header('Детальный анализ')
    # for column in columns:
    # st.subheader(f'столбец - {column}')
    n = pd.DataFrame(st.session_state['data']).shape[0]
    k = data.shape[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'**Анализ прибыли до внедрения модели Decision Tree**')
        st.write(f'Прибыль отеля - {int(data["profit"].sum())} руб.')
        st.write('')
    with col2:
        st.markdown(f'**Анализ прибыли после внедрения модели Decision Tree**')
        st.write(f"Прибыль после обучения DT - {int(data['profit_tree'].sum())} руб.")            
        st.write('')
    with col3:
        st.markdown(f'**Анализ прибыли после внедрения модели Decision Tree**')
        st.write(f"Полученная выгода от обучения DT - {int(data['profit_tree'].sum() - data['profit'].sum() - 400000*k/n)} руб.")
        st.write('')
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for ind in data.index:
        row = data[ind:ind+1]
        predict_t = row['is_canceled_tree'].to_list()[0]
        target_t = row['is_canceled'].to_list()[0]
        if predict_t == target_t:
            if predict_t == 1:
                TP += 1
                profit_pred = row['profit_tree'].to_list()[0]
                profit_true = row['profit'].to_list()[0]
            elif predict_t  == 0:
                TN += 1
        if predict_t != target_t:
            if predict_t == 1:
                FP += 1
            elif predict_t == 0:
                FN += 1   
    sum_pt = FP+FN+TP+TN
    predict_map = {'predict_0': [TN, FN], 'predict_1': [FP, TP]}
    df_predict_map = pd.DataFrame(data=predict_map, index=['target_0', 'target_1'])    
    predict_map_per = (
        {'predict_0': ['{:.2%}'.format(TN/sum_pt), '{:.2%}'.format(FN/sum_pt)], 'predict_1': ['{:.2%}'.format(FP/sum_pt), '{:.2%}'.format(TP/sum_pt)]}
    )
    df_predict_map_per = pd.DataFrame(data=predict_map_per, index=['target_0', 'target_1'])
    st.write(f'Вероятность отказа от брони - {(FN+TP)/sum_pt}')
    st.markdown('**Матрица ошибок в выбранной брони**')
    col1, col2 = st.columns(2)
    with col1:
        st.write('Абсолютные значения')
        st.table(df_predict_map)
    with col2:
        st.write('Относительные значения')
        st.table(df_predict_map_per)

    for column in columns:
        st.subheader(f'столбец - "{column}"')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'**Анализ прибыли по различным разделам "{column}" до внедрения модели Decision Tree**')
            before_bycolumn =  data.groupby(column)['profit'].sum().astype('int')
            st.dataframe(before_bycolumn)
            st.bar_chart(before_bycolumn)
            st.write('')
        with col2:
            st.markdown(f'**Анализ прибыли по различным разделам "{column}"  после внедрения модели Decision Tree**')
            after_bycolumn = data.groupby(column)['profit_tree'].sum().astype('int')
            st.dataframe(after_bycolumn)
            st.bar_chart(after_bycolumn)            
            st.write('')
        with col3:
            st.markdown(f'**Анализ полученной выгоды по различным разделам "{column}" после внедрения модели Decision Tree**')
            st.dataframe(data.groupby(column)['profit'].сount())
            profit_bycolumn = int(data.groupby(column)['profit_tree'].sum().astype('int') - data.groupby(column)['profit'].sum().astype('int') - 400000/n*data.groupby(column)['profit'].сount())
            profit_bycolumn_df = pd.DataFrame(profit_bycolumn.sort_values(ascending=False), columns=['profit_from_DT'])
            st.dataframe(profit_bycolumn_df)
            st.bar_chart(profit_bycolumn_df)
            st.write('')


if st.session_state['true_uploaded']:
    data = st.session_state['data']
    selected_columns = ['arrival_date_month', 'country', 'customer_type', 'total_nights', 'reserved_room_type', 'meal']
    column = st.radio('Выберите столбец', selected_columns)
    if column is not None:
        if column == 'arrival_date_month':
            chosen_month = st.selectbox('Месяц заезда', data['arrival_date_month'].unique())
            new_data = data.query('arrival_date_month == @chosen_month').reset_index(drop=True)
            non_selected_columns = pd.Series(selected_columns)
            non_selected_columns = non_selected_columns[non_selected_columns != column].to_list()
        elif column == 'country':
            citizen = st.selectbox('Гражданство постояльца', data['country'].unique())
            new_data = data.query('country == @citizen').reset_index(drop=True)
            non_selected_columns = pd.Series(selected_columns)
            non_selected_columns = non_selected_columns[non_selected_columns != column].to_list()
        elif column == 'customer_type':
            customer = st.selectbox('Тип заказчика', data['customer_type'].unique())
            new_data = data.query('customer_type == @customer').reset_index(drop=True)
            non_selected_columns = pd.Series(selected_columns)
            non_selected_columns = non_selected_columns[non_selected_columns != column].to_list()
        elif column == 'total_nights':
            nights = st.selectbox('Количество ночей', data['total_nights'].unique())
            new_data = data.query('total_nights == @nights').reset_index(drop=True)
            non_selected_columns = pd.Series(selected_columns)
            non_selected_columns = non_selected_columns[non_selected_columns != column].to_list()
        elif column == 'reserved_room_type':            
            room_type = st.selectbox('Тип комнаты', data['reserved_room_type'].unique())
            new_data = data.query('reserved_room_type == @room_type').reset_index(drop=True)
            non_selected_columns = pd.Series(selected_columns)
            non_selected_columns = non_selected_columns[non_selected_columns != column].to_list()
        elif column == 'meal':            
            book_type = st.selectbox('Опция заказа', data['meal'].unique())
            new_data = data.query('meal == @book_type').reset_index(drop=True)
            non_selected_columns = pd.Series(selected_columns)
            non_selected_columns = non_selected_columns[non_selected_columns != column].to_list()
            
        button = st.button('Провести детальный анализ')

        if button:
            do_analis(new_data, non_selected_columns)
else:
    st.error('Файлы не загружены или загружены некорректно; ещё раз загрузите их во вкладке "Общий анализ"')
