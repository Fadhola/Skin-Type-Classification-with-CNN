import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

def describe_image_shape(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    shape = img_array.shape

    if len(shape) == 2:
        mode = "Grayscale"
    elif shape[2] == 3:
        mode = "RGB"
    elif shape[2] == 4:
        mode = "RGBA"
    else:
        mode = f"{shape[2]}-channel (unknown)"

    return shape, mode

def plot_class_images(path, class_name, random_state):
    folder_path = os.path.join(path, class_name)
    fig = plt.figure(figsize=(16, 10))
    columns = 5
    rows = 3

    images = os.listdir(folder_path)
    random.seed(random_state)
    sampled_images = random.sample(images, min(columns * rows, len(images)))

    for index, img_name in enumerate(sampled_images):
        img_path = os.path.join(folder_path, img_name)
        image = plt.imread(img_path)
        fig.add_subplot(rows, columns, index + 1)
        plt.imshow(image)
        plt.title(f"{image.shape}")
        plt.axis("off")
    st.pyplot(fig)

def plot_class_images_pseudocolor(path, class_name, random_state=42):
    folder_path = os.path.join(path, class_name)
    fig = plt.figure(figsize=(16, 10))
    columns = 5
    rows = 3

    images = os.listdir(folder_path)
    random.seed(random_state)
    sampled_images = random.sample(images, min(columns * rows, len(images)))

    for idx, img_name in enumerate(sampled_images):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path).convert('L')
        img_np = np.array(img)

        ax = fig.add_subplot(rows, columns, idx + 1)
        imgplot = plt.imshow(img_np)
        imgplot.set_cmap('nipy_spectral')
        plt.axis("off")
    plt.tight_layout()
    st.pyplot(fig)

def run():
    st.title('Skin Type Recognition - Exploratory Data Analysis')
    st.write('Page ini dibuat oleh *Fadhola Asandi*')

    classes = ['oily', 'dry', 'normal']
    main_path = './src/Oily-Dry-Skin-Types'
    train_path = os.path.join(main_path, 'train')
    val_path = os.path.join(main_path, 'valid')
    test_path = os.path.join(main_path, 'test')

    st.header('General Visualisation')
    for cls in classes:
        st.subheader(f'Class: {cls}')
        plot_class_images(train_path, cls, random_state=61)
    st.write("""
    Di general visualisasi terlihat bahwa beberapa gambar tampaknya **tidak konsisten** dengan label yang diberikan.
    Yaitu:
    - Pada kelas **oily**, terdapat gambar wajah dengan tekstur kulit kering atau normal.
    - Di kelas **normal**, muncul gambar yang tampak jelas berminyak atau terlalu kering.
    - Bahkan ada gambar **iklan skincare** yang tidak menunjukkan kulit asli, sehingga sulit diidentifikasi.

    Langkah kedepannya bersihkan datanya dulu, pisahkan gambar dengan benar dan hapus yang tidak menunjukkan kulit.
    """)

    st.header('1. Color and Shape')
    st.write("""
    Terlihat Pada kelas **oily**, kulit cenderung lebih mengilap atau reflektif. Kelas **dry** tampak merah dan tidak mengilap, juga memiliki tekstur kasar, sedangkan kelas **normal** terlihat sedikit mengilap.
    tapi banyak gambar yang tidak menunjukkan wajah secara penuh, ada yang hanya sebagian wajah, bahkan ada yang memperlihatkan hingga bahu atau bagian tubuh lainnya dll.
    """)

    st.header('2. Size')
    for cls in classes:
        st.subheader(f'Class: {cls}')
        image_path = os.path.join(train_path, cls, os.listdir(os.path.join(train_path, cls))[0])
        shape, mode = describe_image_shape(image_path)
        st.write(f"Shape: {shape} → Mode: {mode}")
    st.write("Semua gambar sudah seragam dalam hal ukuran dan mode warna, yaitu **640x640 RGB**, sehingga tidak memerlukan preprocessing tambahan untuk penyesuaian resolusi.")

    st.header('3. Room Color using Nipy Spectral Colormap')
    for cls in classes:
        st.subheader(f'Class: {cls}')
        plot_class_images_pseudocolor(train_path, cls, random_state=61)
    st.write(
        """
        Secara umum pola warna tidak terlalu jelas terlihat karena banyak gambar yang bercampur antar kelas, shape dan angle juga tidak teratur. Namun untuk sebagian gambar yang sesuai labelnya terlihat:

        - **Oily**: warna pseudocolor cenderung lebih terang dan intens, mencerminkan area kulit yang memantulkan cahaya.
        - **Dry**: distribusi warna cenderung lebih gelap, menunjukkan tekstur kulit yang kering atau kasar.
        - **Normal**: warna terlihat lebih merata dan tidak terlalu ekstrem dalam luminansi.
        """
    )

    st.header('4. Angle')
    st.write("""
    Banyak gambar tidak memiliki sudut pengambilan yang seragam. Ada yang diambil dari depan, samping, atas, bahkan dari kejauhan. 
    Beberapa hanya memperlihatkan sebagian wajah atau tubuh, yang berpotensi menjadi gangguan bagi model dalam memahami fitur kulit secara konsisten.
    """)

    st.header('5. Another Character')
    st.write("""
    Karakteristik lain yang menonjol di antaranya:
    - **Tangan atau benda** menutupi sebagian wajah.
    - Banyak gambar memiliki **noise visual**, seperti watermark, logo brand, atau teks iklan.
    - **Latar belakang** gambar sangat beragam, dari yang polos hingga sangat ramai, yang bisa menjadi distraktor saat training model.
    """)



if __name__ == '__main__':
    run()