import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import time
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
from labelling import pelabelan
from preprocessing import preprocessing_text
import cv2
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.metrics import classification_report


def intro():

    st.sidebar.success(list(pages_multi.keys())[0])
    st.write("Sentimen opini ulasan di Tokopedia")

    st.markdown(
        """
       Ulasan pelanggan di Tokopedia
       Sentimen Elektronik
       Sentimen Pakaian
    """
    )


def fn_wordcloud():
    originalImage = cv2.imread('img\cloud.jpg')
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    (thresh, cloud_mask) = cv2.threshold(
        grayImage, 100, 255, cv2.THRESH_BINARY)
    st.title(list(pages_multi.keys())[4])
    
    st.sidebar.success(list(pages_multi.keys())[4])
    negatif_string_elektronik = []
    positif_string_elektronik = []
    netral_string_elektronik = []

    st.header("""
    Wordcloud Elektronik
    """)
    df_elektronik = pd.read_csv("assets\data_elektronik.csv")
    elektronik_positif = df_elektronik[df_elektronik["kelas"] == "positif"]
    for t in elektronik_positif["komentar_bersih"]:
        positif_string_elektronik.append(t)

    positif_string_elektronik = pd.Series(
        positif_string_elektronik).str.cat(sep=' ')

    wordcloud_positif_elektronik = WordCloud(width=1600, height=800, margin=10,
                                             background_color='white', colormap='Dark2',
                                             max_font_size=200, min_font_size=25,
                                             mask=cloud_mask, contour_width=10, contour_color='firebrick',
                                             max_words=100).generate(positif_string_elektronik)
    plt.figure(figsize=(10, 8))
    st.markdown(""" Wordcloud label positif """)

    st.image(wordcloud_positif_elektronik.to_array())

    st.write("wordcloud negatif")
    pakaian_negatif = df_elektronik[df_elektronik["kelas"] == "negatif"]

    for t in pakaian_negatif["komentar_bersih"]:
        negatif_string_elektronik.append(t)

    negatif_string_elektronik = pd.Series(
        negatif_string_elektronik).str.cat(sep=' ')

    wordcloud_negatif_elektronik = WordCloud(width=1600, height=800, margin=10,
                                             background_color='white', colormap='Dark2',
                                             max_font_size=200, min_font_size=25,
                                             mask=cloud_mask, contour_width=10, contour_color='firebrick',
                                             max_words=100).generate(negatif_string_elektronik)
    plt.figure(figsize=(10, 8))
    st.image(wordcloud_negatif_elektronik.to_array())

    st.write("wordcloud netral")
    pakaian_netral = df_elektronik[df_elektronik["kelas"] == "netral"]

    for t in pakaian_netral["komentar_bersih"]:
        netral_string_elektronik .append(t)

    netral_string_elektronik = pd.Series(
        netral_string_elektronik).str.cat(sep=' ')

    (thresh, cloud_mask) = cv2.threshold(
        grayImage, 100, 255, cv2.THRESH_BINARY)
    wordcloud_netral_elektronik = WordCloud(width=1600, height=800, margin=10,
                                            background_color='white', colormap='Dark2',
                                            max_font_size=200, min_font_size=25,
                                            mask=cloud_mask, contour_width=10, contour_color='firebrick',
                                            max_words=100).generate(netral_string_elektronik)
    plt.figure(figsize=(10, 8))
    st.image(wordcloud_netral_elektronik.to_array())
    st.header("""
    Wordcloud Pakaian
    """)
    
    negatif_string_pakaian = []
    positif_string_pakaian = []
    netral_string_pakaian = []

    df_pakaian = pd.read_csv("assets\data_pakaian.csv")
    pakaian_positif = df_pakaian[df_pakaian["kelas"] == "positif"]
    for t in pakaian_positif["komentar_bersih"]:
        positif_string_pakaian.append(t)

    positif_string_pakaian = pd.Series(positif_string_pakaian).str.cat(sep=' ')

    wordcloud_positif_pakaian = WordCloud(width=1600, height=800, margin=10,
                                          background_color='white', colormap='Dark2',
                                          max_font_size=200, min_font_size=25,
                                          mask=cloud_mask, contour_width=10, contour_color='firebrick',
                                          max_words=100).generate(positif_string_pakaian)
    plt.figure(figsize=(10, 8))
    st.markdown(""" Wordcloud label positif """)
    st.image(wordcloud_positif_pakaian.to_array())

    st.write("wordcloud negatif")
    pakaian_negatif = df_pakaian[df_pakaian["kelas"] == "negatif"]

    for t in pakaian_negatif["komentar_bersih"]:
        negatif_string_pakaian.append(t)

    negatif_string_pakaian = pd.Series(negatif_string_pakaian).str.cat(sep=' ')

    wordcloud_negatif_pakaian = WordCloud(width=1600, height=800, margin=10,
                                          background_color='white', colormap='Dark2',
                                          max_font_size=200, min_font_size=25,
                                          mask=cloud_mask, contour_width=10, contour_color='firebrick',
                                          max_words=100).generate(negatif_string_pakaian)
    plt.figure(figsize=(10, 8))
    st.image(wordcloud_negatif_pakaian.to_array())

    st.write("wordcloud netral")
    pakaian_netral = df_pakaian[df_pakaian["kelas"] == "netral"]

    for t in pakaian_netral["komentar_bersih"]:
        netral_string_pakaian .append(t)

    netral_string_pakaian = pd.Series(netral_string_pakaian).str.cat(sep=' ')

    (thresh, cloud_mask) = cv2.threshold(
        grayImage, 100, 255, cv2.THRESH_BINARY)
    wordcloud_netral_pakaian = WordCloud(width=1600, height=800, margin=10,
                                         background_color='white', colormap='Dark2',
                                         max_font_size=200, min_font_size=25,
                                         mask=cloud_mask, contour_width=10, contour_color='firebrick',
                                         max_words=100).generate(netral_string_pakaian)
    plt.figure(figsize=(10, 8))
    st.image(wordcloud_netral_pakaian.to_array())

    
def sentimen_elektronik():
    import pandas as pd

    st.markdown(f'# {list(pages_multi.keys())[2]}')
    st.sidebar.success(list(pages_multi.keys())[2])

    url_elektronik = st.text_input('Masukkan Link url')
    if not url_elektronik:
        st.warning("please input url")
        st.stop()
    else:
        tombol_ambil_data_elektronik = st.button("Ambil data")
        if not tombol_ambil_data_elektronik:
            st.stop()
        else:
            data_scrap = []
            rating_scrap = []
            options = webdriver.ChromeOptions()
            options.add_argument("--start-maximized")
            driver = webdriver.Chrome(options=options)
            driver.get(url_elektronik)
            for i in range(0, 10):
                soup = BeautifulSoup(driver.page_source, 'lxml')
                time.sleep(2)
                time.sleep(3)
                contain_file = soup.find_all(
                    'article', attrs={'class': 'css-72zbc4'})
                for contain in contain_file:
                    review = contain.find(
                        'span', attrs={"data-testid": 'lblItemUlasan'})

                    review = review.text if review is not None else " "
                    ratings = contain.find(
                        'div', attrs={"data-testid": 'icnStarRating'})
                    data_scrap.append(review)
                    rating_scrap.append(ratings)

                time.sleep(2)
                driver.find_element(
                    By.CSS_SELECTOR, "li:nth-child(11) .unf-icon").click()
                time.sleep(3)

    df = pd.DataFrame({
        "komentar": data_scrap,
        "rating": rating_scrap,
    })

    df.to_csv("assets\data_elektronik.csv")
    df_elektronik = pd.read_csv("assets\data_elektronik.csv")
    df_elektronik.dropna(inplace=True)
    # regex
    df_elektronik["komentar"] = df_elektronik["komentar"].str.replace(
        r'<[^<>]*>', '', regex=True)
    df_elektronik['rating'] = df_elektronik['rating'].str.get(25)

    df_elektronik = df_elektronik[["komentar", "rating"]]
    df_elektronik["kelas"] = df_elektronik["rating"].apply(pelabelan)
    df_elektronik["komentar_bersih"] = df_elektronik["komentar"].apply(
        preprocessing_text)
    df_elektronik.to_csv("assets\data_elektronik.csv")


def sentimen_pakaian():
    import streamlit as st
    import time

    st.markdown(f'# {list(pages_multi.keys())[1]}')
    st.sidebar.success(list(pages_multi.keys())[1])

    st.write(
        """
        Sentimen Pakaian di Tokopedia
"""
    )
    url_pakaian = st.text_input('Masukkan Link url', key="urlpakaian")
    if not url_pakaian:
        st.warning("please input url")
        st.stop()
    else:
        tombol_ambil_data_pakaian = st.button("Ambil data")
        if not tombol_ambil_data_pakaian:
            st.stop()
        else:
            data_scrap = []
            rating_scrap = []
            options = webdriver.ChromeOptions()
            options.add_argument("--start-maximized")
            driver = webdriver.Chrome(options=options)
            driver.get(url_pakaian)
            for i in range(0, 10):
                soup = BeautifulSoup(driver.page_source, 'lxml')
                time.sleep(2)
                time.sleep(3)
                contain_file = soup.find_all(
                    'article', attrs={'class': 'css-72zbc4'})
                for contain in contain_file:
                    review = contain.find(
                        'span', attrs={"data-testid": 'lblItemUlasan'})

                    review = review.text if review is not None else " "
                    ratings = contain.find(
                        'div', attrs={"data-testid": 'icnStarRating'})
                    data_scrap.append(review)
                    rating_scrap.append(ratings)

                time.sleep(2)
                driver.find_element(
                    By.CSS_SELECTOR, "li:nth-child(11) .unf-icon").click()
                time.sleep(3)

    df_pakaian = pd.DataFrame({
        "komentar": data_scrap,
        "rating": rating_scrap,
    })

    df_pakaian.to_csv("assets\data_pakaian.csv")
    df_pakaian = pd.read_csv("assets\data_pakaian.csv")
    df_pakaian.dropna(inplace=True)
    # regex
    df_pakaian["komentar"] = df_pakaian["komentar"].str.replace(
        r'<[^<>]*>', '', regex=True)
    df_pakaian['rating'] = df_pakaian['rating'].str.get(25)

    df_pakaian = df_pakaian[["komentar", "rating"]]
    df_pakaian["kelas"] = df_pakaian["rating"].apply(pelabelan)
    df_pakaian["komentar_bersih"] = df_pakaian["komentar"].apply(
        preprocessing_text)
    df_pakaian.to_csv("assets\data_pakaian.csv")


def data_frame_result():
    import streamlit as st
    import pandas as pd
    st.title(list(pages_multi.keys())[3])
    st.sidebar.success(list(pages_multi.keys())[3])

    st.write("Data Elektronik")
    df_elektronik = pd.read_csv("assets\data_elektronik.csv")
    st.dataframe(df_elektronik[["komentar","komentar_bersih","rating", "kelas"]])

    st.write("Data Pakaian")
    df_pakaian = pd.read_csv("assets\data_pakaian.csv")

    st.dataframe(df_pakaian[["komentar","komentar_bersih","rating", "kelas"]])

    prediksi = st.button("Prediksi")
    if not prediksi:
        st.stop()
    else:

        st.title("Prediksi data pakaian")
        data_pakaian = df_pakaian.dropna()

        loaded_vec_pakaian = TfidfVectorizer(decode_error="replace", vocabulary=set(
            pickle.load(open(os.path.join("model-tfidf\idf-pakaian.pkl"), 'rb'))))

        predictor_load_dec_pakaian = pickle.load(open(
            "content\model_tree\model_pakaian\model_dec_pakaian_10persen.pkl", 'rb'))

        tfidf_pakaian = loaded_vec_pakaian.fit_transform(
            data_pakaian["komentar_bersih"])
        prediction_pakaian_tree = predictor_load_dec_pakaian.predict(
            tfidf_pakaian)

        st.dataframe({"komentar":  data_pakaian["komentar_bersih"],
                      "prediksi": prediction_pakaian_tree})

        test_label_pakaian = data_pakaian["kelas"]
        NewprediksiBenar_pakaian = (
            prediction_pakaian_tree == test_label_pakaian).sum()
        NewprediksiSalah_pakaian = (
            prediction_pakaian_tree != test_label_pakaian).sum()

        st.title("Prediksi data elektronik")
        data_elektronik = df_elektronik.dropna()

        loaded_vec_elektronik = TfidfVectorizer(decode_error="replace", vocabulary=set(
            pickle.load(open(os.path.join("model-tfidf\idf-elektronik.pkl"), 'rb'))))

        predictor_load_dec_elektronik = pickle.load(open(
            "content\model_tree\model_elektronik\model_dec_elektronik_40persen.pkl", 'rb'))

        tfidf_elektronik = loaded_vec_elektronik.fit_transform(
            data_elektronik["komentar_bersih"])
        prediction_elektronik_tree = predictor_load_dec_elektronik.predict(
            tfidf_elektronik)

        st.dataframe({"komentar":  data_elektronik["komentar_bersih"],
                      "prediksi": prediction_elektronik_tree})

        test_label_elektronik = data_elektronik["kelas"]
        NewprediksiBenar_elektronik = (
            prediction_elektronik_tree == test_label_elektronik).sum()
        NewprediksiSalah_elektronik = (
            prediction_elektronik_tree != test_label_elektronik).sum()

        st.header("Evaluasi")
        st.markdown(""" Evaluasi prediksi data pakaian""")
        st.write(f"Prediksi benar: {NewprediksiBenar_pakaian}")
        st.write(f"Prediksi salah: {NewprediksiSalah_pakaian}")
        st.dataframe(classification_report(test_label_pakaian,
                     prediction_pakaian_tree, output_dict=True))

        st.markdown(""" Evaluasi prediksi data elektronik""")
        st.write(f"Prediksi benar: {NewprediksiBenar_elektronik}")
        st.write(f"Prediksi salah: {NewprediksiSalah_elektronik}")
        st.dataframe(classification_report(test_label_elektronik,
                     prediction_elektronik_tree, output_dict=True))


pages_multi = {
    "—": intro,
    "Sentimen Pakaian": sentimen_pakaian,
    "Sentimen Elektronik": sentimen_elektronik,
    "Hasil data": data_frame_result,
    "Wordlcoud": fn_wordcloud
}

page_choose = st.sidebar.selectbox("Pilih halaman ", pages_multi.keys())
pages_multi[page_choose]()
