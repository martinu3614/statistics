import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
from pylab import rcParams
import seaborn as sns
import sys
import scipy.stats as sts

#データ補完、変換用WebAPP
#サイトタイトル
st.title('データ補完&変換を簡単に...')

#ファイルアップロード
uploaded_file = st.sidebar.file_uploader("ファイルアップロード", type='csv')


if uploaded_file is not None:
    #アップロードファイルをDataFrameに変換
    df = pd.read_csv(uploaded_file)
    #データ型
    data_type = df.dtypes.to_frame()
    data_type.columns = ['データ型']
    #カラムのリスト
    column = df.columns.values
    column_list = column.tolist()
    #数値データのカラムリスト
    numerical_df = data_type[data_type['データ型'] != object]
    numerical_column = numerical_df.index.values
    numerical_column_list = numerical_column.tolist()
    #二値データのカラムリスト
    data_unique = df.nunique().to_frame()
    data_unique.columns = ['ユニークな要素数']
    twovalues_df = data_unique[data_unique['ユニークな要素数'] == 2]
    twovalues_column = twovalues_df.index.values
    twovalues_column_list = twovalues_column.tolist()
    #カテゴリ変数のカラムリスト（昇順）
    data_type_unique = df.dtypes.to_frame()
    data_type_unique['ユニークな要素数'] = data_unique['ユニークな要素数']
    categorical_column = data_type_unique[data_type_unique['ユニークな要素数'] < 10].index.values
    categorical_column_list = categorical_column.tolist()
    if len(categorical_column_list) != 0:
        categorical_data = df[categorical_column]  
        categorical_data_unique = categorical_data.nunique().to_frame()
        categorical_data_unique.columns = ['値の種類']
        #カラムリストを要素の種類数でソート
        categorical_data_unique = categorical_data_unique.sort_values('値の種類')
        categorical_column = categorical_data_unique.index.values
        categorical_column_list = categorical_column.tolist()
    #連続変数のカラムリスト
    continuous_column = np.intersect1d(numerical_column, data_unique[data_unique['ユニークな要素数'] >= 10].index.values)
    continuous_column_list = continuous_column.tolist()
    #連続変数とカテゴリ変数を合わせたカラムリスト
    con_cate_column = np.union1d(categorical_column, continuous_column)
    con_cate_column_list = con_cate_column.tolist()
else:
    pass


#______________________________________________________________________________________
#アップロードデータの確認
if uploaded_file is not None:
    data_display = st.checkbox('アップロードしたデータを表示する')
    if data_display == True:
        st.header('読み込みデータ（100行目まで）')
        st.dataframe(df.head(100))
    else:
        pass
else:
    st.header('csvファイルを選択して下さい')

#______________________________________________________________________________________
#データの概要
if uploaded_file is not None:
    st.sidebar.title('データの概要')
else:
    pass


#______________________________________________________________________________________
#データ型の確認
if uploaded_file is not None:
    dtype_check = st.sidebar.checkbox('データ型の確認')
    if dtype_check == True:
        st.title('データの概要')
        st.header('データ型の確認')
        st.table(data_type)
        st.write('(int : 整数 , float : 小数 , object : 文字 , bool : 真偽)')
    else:
        pass
else:
    pass


#要約統計量の確認
if uploaded_file is not None:
    summary_check = st.sidebar.checkbox('要約統計量の確認')
    summary_index = ['データ数', '平均値', '標準偏差', '最小値', '第一四分位数', '中央値', '第二四分位数', '最大値']
    if summary_check == True:
        if dtype_check == False:
            st.title('データの概要')
        st.header('要約統計量の確認')
        if len(continuous_column_list) != 0:
            summary_data = df.describe()
            summary_data.index = summary_index
            st.table(summary_data)
            st.write('※数値で表されたカテゴリ変数が含まれている場合があります')
        else:
            st.write('数値型の変数がありません')
    else:
        pass
else:
    pass

#カテゴリ変数の値の種類
if uploaded_file is not None:
    unique_check = st.sidebar.checkbox('カテゴリ変数の値の種類')
    if unique_check == True:
        if dtype_check == False and summary_check == False:
            st.title('データの概要')
        else:
            pass
        st.header('カテゴリ変数の値の種類')
        if len(categorical_column_list) != 0:
            st.table(categorical_data_unique)
        else:
            st.write('カテゴリ変数がありません')
    else:
        pass
else:
    pass

#欠損数の確認
if uploaded_file is not None:
    null_check = st.sidebar.checkbox('欠損数の確認')
    if null_check == True:
        if dtype_check == False and summary_check == False and unique_check == False:
            st.title('データの概要')
        else:
            pass
        st.header('欠損数の確認')
        null_count = df.isnull().sum().to_frame()
        null_count.columns = ['欠損数']
        null_count['欠損割合'] = null_count['欠損数'] / len(df.index)
        st.table(null_count)
    else:
        pass
else:
    pass

#変数の相関行列
if uploaded_file is not None:
    corr_matrix_check = st.sidebar.checkbox('相関行列の確認')
    if corr_matrix_check == True:
        if dtype_check == False and summary_check == False and unique_check == False and null_check == False:
            st.title('データの概要')
        else:
            pass
        st.header('相関行列の確認')
        corr_matrix_method = 'pearson'
        corr_matrix_method = st.radio('相関係数の算出方法（相関行列）', ('pearson', 'spearman', 'kendall'))
        corr_matrix = df.corr(method=corr_matrix_method)
        st.table(corr_matrix)
        st.write('※注目したい変数に合わせて算出方法を適切に選択して下さい')
    else:
        pass
else:
    pass


#______________________________________________________________________________________
#欠損値の扱い

#欠損地のまま扱う
