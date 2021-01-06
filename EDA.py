import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import sys

#EDA用WebApp
#サイトタイトル
st.title('EDA（探索的データ解析）を簡単に...')

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
#グラフの作成
if uploaded_file is not None:
    st.sidebar.title('グラフの作成')
    title = None
    border = None
    grid_check = st.sidebar.checkbox('グラフにグリッドをいれる')
    if grid_check == True:
        sns.set()
    else:
        sns.set_style('white')
    file_type = ('png', 'jpg', 'pdf', 'jpeg', 'eps', 'pgf', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff')
else:
    pass


#棒グラフの作成
if uploaded_file is not None and len(categorical_column_list) >= 1:
    bar_check = st.sidebar.checkbox('棒グラフ')
    if bar_check == True:
        border = not None
        title = st.title('グラフの作成')
        st.header('棒グラフ')
        st.write('（※値の種類が多いとグラフの作成に時間がかかることがあります）')
        bar_x = st.sidebar.selectbox('棒グラフのx軸を選択してください', categorical_column)
        bar_hue_check = st.checkbox('カテゴリ毎に棒グラフを分ける')
        if bar_hue_check == True:
            bar_hue = st.selectbox('棒グラフを分けるカテゴリ変数', categorical_column)
            bar_fig = plt.figure()
            sns.countplot(x=bar_x, data=df, hue=bar_hue, )
        else:
            bar_fig = plt.figure()
            sns.countplot(x=bar_x, data=df)
        st.pyplot(bar_fig)
        #グラフの保存
        bar_name = st.text_input('ファイル名（棒グラフ）')
        bar_file_type = st.selectbox('拡張子（棒グラフ）', file_type)
        bar_out_put = st.button('棒グラフを出力する')
        if bar_out_put == True and bar_name != '':
            if f'.{bar_file_type}' in bar_name:
                bar_fig.savefig(bar_name)
            else:
                bar_fig.savefig(bar_name + f'.{bar_file_type}')
        elif bar_out_put == True and bar_name == '':
            st.write('※ファイル名を入力して下さい※')
        else:
            pass
    else:
        st.write('カテゴリ変数がありません')
else:
    pass


#ヒストグラムの作成
#スタージェス
sturges = lambda n: math.ceil(math.log2(n*2))
if uploaded_file is not None and len(continuous_column_list) >= 1:
    hist_check = st.sidebar.checkbox('ヒストグラム')
    if hist_check == True:
        if border is not None:
            st.write('_________________________________________________________________')
        else:
            border = not None
        if title is None:
            title = st.title('グラフの作成')
        st.header('ヒストグラム')
        hist_x = st.sidebar.selectbox('ヒストグラムのx軸を選択してください', continuous_column)
        hist_bin = sturges(len(df[hist_x]))
        hist_hue_check = st.checkbox('カテゴリ毎にヒストグラムを分ける')
        if hist_hue_check == True and len(categorical_column_list) >= 1:
            hist_hue = st.selectbox('ヒストグラムを分けるカテゴリ変数', categorical_column)
            hist_fig = sns.FacetGrid(df, col=hist_hue)
            hist_fig.map_dataframe(sns.histplot, x=hist_x, ec='white')
            hist_fig.set_axis_labels(hist_x, "Count")
            hist_fig_stack = plt.figure()
            sns.histplot(data=df, x=hist_x, hue=hist_hue, edgecolor='white')
            st.pyplot(hist_fig_stack)
        else:
            hist_fig = plt.figure()
            sns.histplot(data=df, x=df[hist_x], bins=hist_bin, ec='white')
        st.pyplot(hist_fig)
        #グラフの保存
        hist_name = st.text_input('ファイル名（ヒストグラム）')
        hist_file_type = st.selectbox('拡張子（ヒストグラム）', file_type)
        hist_out_put = st.button('ヒストグラムを出力する')
        if hist_out_put == True and hist_name != '':
            if f'.{hist_file_type}' in hist_name:
                hist_fig.savefig(hist_name)
            else:
                hist_fig.savefig(hist_name + f'.{hist_file_type}')
        elif hist_out_put == True and hist_name == '':
            st.write('※ファイル名を入力して下さい※')
        else:
            pass
    else:
        pass
else:
    pass

#箱ひげ図andバイオリンプロットの作成
if uploaded_file is not None and len(categorical_column_list) >= 1 and len(continuous_column_list) >= 1:
    box_check = st.sidebar.checkbox('箱ひげ図とバイオリンプロット')
    if box_check == True:
        if border is not None:
            st.write('_________________________________________________________________')
        else:
            border = not None
        if title is None:
            title = st.title('グラフの作成')
        st.header('箱ひげ図')
        box_x = st.sidebar.selectbox('箱ひげ図のx軸を選択してください', categorical_column)
        box_y = st.sidebar.selectbox('箱ひげ図のy軸を選択してください', continuous_column)
        box_df = pd.concat([df[box_x], df[box_y]], axis=1, join='outer')
        box_fig = plt.figure()
        box_fig.add_subplot(111)
        sns.boxplot(x=box_x, y=box_y, data=box_df)
        st.pyplot(box_fig)
        #グラフの保存
        box_name = st.text_input('ファイル名（箱ひげ図）')
        select_file_type = st.selectbox('拡張子（箱ひげ図）', file_type)
        box_out_put = st.button('箱ひげ図を出力する')
        if box_out_put == True and box_name != '':
            if f'.{box_file_type}' in box_name:
                box_fig.savefig(box_name)
            else:
                box_fig.savefig(box_name + f'.{box_file_type}')
        elif box_out_put == True and box_name == '':
            st.write('※ファイル名を入力して下さい※')
        else:
            pass
        st.header('バイオリンプロット')
        violin_fig = plt.figure()
        sns.violinplot(x=df[box_x], y=df[box_y])
        st.pyplot(violin_fig)
        #グラフの保存
        violin_name = st.text_input('ファイル名（バイオリンプロット）')
        violin_file_type = st.selectbox('拡張子（バイオリンプロット）', file_type)
        violin_out_put = st.button('バイオリンプロットを出力する')
        if violin_out_put == True and violin_name != '':
            if f'.{violin_file_type}' in violin_name:
                violin_fig.savefig(violin_name)
            else:
                violin_fig.savefig(violin_name + f'.{violin_file_type}')
        elif violin_out_put == True and violin_name == '':
            st.write('※ファイル名を入力して下さい※')
        else:
            pass
    else:
        pass
else:
    pass

#散布図の作成
if uploaded_file is not None and len(numerical_column_list) >= 1:
    scatt_check = st.sidebar.checkbox('散布図')
    if scatt_check == True:
        if border is not None:
            st.write('_________________________________________________________________')
        else:
            border = not None
        if title is None:
            title = st.title('グラフの作成')
        st.header('散布図')
        scatt_x = st.sidebar.selectbox('散布図のx軸を選択してください', numerical_column)
        scatt_y = st.sidebar.selectbox('散布図のy軸を選択してください', numerical_column)
        scatt_df = pd.concat([(df[scatt_x]), (df[scatt_y])], axis=1, join='outer')
        if scatt_x == scatt_y:
            st.write('x軸とy軸には別の変数を選択して下さい')
        else:
            scatt_color_check = st.checkbox('カテゴリ毎に散布図を分ける')
            if scatt_color_check == False:
                scatt_hist_list = ['普通', '六角形', '等高線', '回帰直線付き']
                scatt_hist_type = st.selectbox('散布図のタイプを指定して下さい', scatt_hist_list)
                if scatt_hist_type == '普通':
                    scatt_hist_fig = sns.jointplot(scatt_x, scatt_y, data=df)
                elif scatt_hist_type == '六角形':
                    scatt_hist_fig = sns.jointplot(scatt_x, scatt_y, data=df, kind='hex')
                elif scatt_hist_type == '等高線':
                    scatt_hist_fig = sns.jointplot(scatt_x, scatt_y, data=df, kind='kde')
                else:
                    scatt_hist_fig = sns.jointplot(scatt_x, scatt_y, data=df, kind='reg')
                st.pyplot(scatt_hist_fig)
                #グラフの保存
                scatt_name = st.text_input('ファイル名（散布図）')
                scatt_file_type = st.selectbox('拡張子（散布図）', file_type)
                scatt_out_put = st.button('散布図を出力する')
                if scatt_out_put == True and scatt_name != '':
                    if f'.{scatt_file_type}' in scatt_name:
                        scatt_fig.savefig(scatt_name)
                    else:
                        scatt_hist_fig.savefig(scatt_name + f'.{scatt_file_type}')
                elif scatt_out_put == True and scatt_name == '':
                    st.write('※ファイル名を入力して下さい※')
                else:
                    pass
            else:
                scatt_color = st.selectbox('散布図を分けるカテゴリ変数', con_cate_column)
                scatt_df = pd.concat([scatt_df, df[scatt_color]], axis=1, join='outer')
                scatt_fig = sns.lmplot(x=scatt_x, y=scatt_y, data=scatt_df, hue=scatt_color)
                st.pyplot(scatt_fig)
                #グラフの保存
                scatt_name = st.text_input('散布図のファイル名を入力して下さい（Enterで確定）')
                scatt_file_type = st.selectbox('散布図の拡張子を選択して下さい', file_type)
                scatt_out_put = st.button('散布図を出力する')
                if scatt_out_put == True and scatt_name != '':
                    if f'.{scatt_file_type}' in scatt_name:
                        scatt_fig.savefig(scatt_name)
                    else:
                        scatt_fig.savefig(scatt_name + f'.{scatt_file_type}')
                elif scatt_out_put == True and scatt_name == '':
                    st.write('※ファイル名を入力して下さい※')
                else:
                    pass
    else:
        pass
else:
    pass

#散布図行列の作成
if uploaded_file is not None and len(continuous_column_list) >= 1:
    scatt_matrix_check = st.sidebar.checkbox('散布図行列')
    if scatt_matrix_check == True:
        if border is not None:
            st.write('_________________________________________________________________')
        else:
            border = not None
        if title is None:
            title = st.title('グラフの作成')
        scatt_matrix_list = st.sidebar.multiselect('散布図行列に使用する変数を選択してください', continuous_column_list, (continuous_column_list[0], continuous_column_list[1]))
        st.header('散布図行列')
        scatt_matrix_df = df[scatt_matrix_list]
        scatt_matrix_fig = sns.pairplot(scatt_matrix_df)
        st.pyplot(scatt_matrix_fig)
        #グラフの保存
        scatt_matrix_name = st.text_input('ファイル名（散布図行列）')
        scatt_matrix_file_type = st.selectbox('拡張子（散布図行列）', file_type)
        scatt_matrix_out_put = st.button('散布図行列を出力する')
        if scatt_matrix_out_put == True and scatt_matrix_name != '':
            if f'.{scatt_matrix_file_type}' in scatt_matrix_name:
                scatt_matrix_fig.savefig(scatt_matrix_name)
            else:
                scatt_matrix_fig.savefig(scatt_matrix_name + f'.{scatt_matrix_file_type}')
        elif scatt_matrix_out_put == True and scatt_matrix_name == '':
            st.write('※ファイル名を入力して下さい※')
        else:
            pass
    else:
        pass
else:
    pass

#ヒートマップ
if uploaded_file is not None and len(numerical_column_list) >= 1:
    heat_check = st.sidebar.checkbox('ヒートマップ')
    if heat_check == True:
        if border is not None:
            st.write('_________________________________________________________________')
        else:
            border = not None
        if title is None:
            title = st.title('グラフの作成')
        st.header('ヒートマップ')
        heat_list = st.sidebar.multiselect('ヒートマップに使用する変数を選択してください', numerical_column_list, (numerical_column_list[0], numerical_column_list[1]))
        if len(heat_list) <= 1:
            st.write('変数を複数選択して下さい')
        else:
            heat_df = df[heat_list]
            #相関行列の算出
            heat_method = 'pearson'
            heat_method = st.radio('相関係数の算出方法（ヒートマップ）', ('pearson', 'spearman', 'kendall'))
            heat_corr = heat_df.corr(method=heat_method)
            #ヒートマップの作成
            value_display = True
            value_display_radio = st.radio('相関係数の表示有無', ('表示する', '表示しない'))
            if value_display_radio == '表示する':
                pass
            else:
                value_display = False
            heat_fig = plt.figure()
            sns.heatmap(heat_corr,  vmin=-1.0, vmax=1.0, center=0, annot=value_display, fmt='.3f', xticklabels=heat_corr.columns.values, yticklabels=heat_corr.columns.values)
            st.pyplot(heat_fig)
            #グラフの保存
            heat_name = st.text_input('ファイル名（ヒートマップ）')
            heat_file_type = st.selectbox('拡張子（ヒートマップ）', file_type)
            heat_out_put = st.button('ヒートマップを出力する')
            if heat_out_put == True and heat_name != '':
                if f'.{heat_file_type}' in heat_name:
                    heat_fig.savefig(heat_name)
                else:
                    heat_fig.savefig(heat_name + f'.{heat_file_type}')
            elif heat_out_put == True and heat_name == '':
                st.write('※ファイル名を入力して下さい※')
            else:
                pass
    else:
        pass
else:
    pass

#折れ線グラフ
if uploaded_file is not None and len(numerical_column_list) >= 1:
    line_chart_check = st.sidebar.checkbox('折れ線グラフ')
    if line_chart_check == True:
        if border is not None:
            st.write('_________________________________________________________________')
        else:
            border = not None
        if title is None:
            title = st.title('グラフの作成')
        st.header('折れ線グラフ')
        line_chart_x = st.sidebar.selectbox('折れ線グラフのx軸を選択してください', numerical_column)
        line_chart_y = st.sidebar.selectbox('折れ線グラフのy軸を選択してください', numerical_column)
        line_chart_color_check = st.checkbox('カテゴリ毎に折れ線グラフを分ける')
        if line_chart_color_check == False:
            line_chart_fig = plt.figure()
            sns.lineplot(x=line_chart_x, y=line_chart_y, data=df)
        else:
            line_chart_color = st.selectbox('折れ線グラフを分けるカテゴリ変数', categorical_column)
            line_chart_fig = plt.figure()
            sns.lineplot(x=line_chart_x, y=line_chart_y, hue=line_chart_color, data=df)
        st.pyplot(line_chart_fig)
        #グラフの保存
        line_chart_name = st.text_input('ファイル名（折れ線グラフ）')
        line_chart_file_type = st.selectbox('拡張子（折れ線グラフ）', file_type)
        line_chart_out_put = st.button('折れ線グラフを出力する')
        if line_chart_out_put == True and line_chart_name != '':
            if f'.{line_chart_file_type}' in line_chart_name:
                line_chart_fig.savefig(line_chart_name)
            else:
                line_chart_fig.savefig(line_chart_name + f'.{line_chart_file_type}')
        elif line_chart_out_put == True and heat_name == '':
            st.write('※ファイル名を入力して下さい※')
        else:
            pass
    else:
        pass
else:
    pass