import boto3
import tempfile
import traceback
from keras.models import load_model
from keras.optimizers import Adam
import librosa
import numpy as np
import tensorflow as tf
import mysql.connector as connector

def lambda_handler(event, context):
    try:
        # S3에서 파일 다운로드
        s3 = boto3.client('s3')
        bucket_name = 'noiroze-noisefile'
        s3_file_key = event['Records'][0]['s3']['object']['key']
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        s3.download_file(bucket_name, s3_file_key, tmp_file.name)
        filename = tmp_file.name

        # 모델 로드
        model = load_model("all_batch32_dense(224,224).hdf5", custom_objects={".Adam": Adam})

        # 파일 로드
        y, sr = librosa.load(filename)


        class_names = ['1-1어른발걸음', '1-2아이발걸음', '1-3망치질', '2-1가구끄는', '2-2문여닫는','2-3.런닝머신','2-4골프퍼팅','3-1화장실',
                    '3-2샤워','3-3드럼세탁기','3-4통돌이세탁기','3-5진공청소기','3-6식기세척기','4-1바이올린','4-2피아노','5-1개','5-2고양이']

        check = [0] * len(class_names)  # 가장 많이 나온 것 판단

        n_fft = 2048
        hop_length = 512
        n_mels = 64
        fmin = 20
        fmax = 8000
        duration = 15 

        # 이미지화 -> 판단
        for j in range(0, len(y)-duration*sr+1, duration*sr):

            y_interval = y[j:j+duration*sr]

            S = librosa.feature.melspectrogram(y=y_interval, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)

            S_dB = librosa.power_to_db(S, ref=np.max)
            S_dB_norm = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB)) * 255

            S_dB_norm_resized = tf.image.resize(tf.expand_dims(tf.convert_to_tensor(S_dB_norm), axis=-1), [224, 224])
            S_dB_norm_resized_4d = tf.expand_dims(S_dB_norm_resized, axis=0)
            S_dB_norm_resized_4d = tf.repeat(S_dB_norm_resized_4d, 3, axis=-1).numpy()

            preds = model.predict(S_dB_norm_resized_4d) #판단하는 부분

            check[np.argmax(preds[0])]+=1

        # 결과값
        result_class = class_names[np.argmax(check)]
        print(result_class)

        # MySQL db에 결과값 저장
        mydb = connector.connect(
            host="3.36.141.220",
            user="root",
            password="password",
            database="mysql",
            port=3307
        )

        mycursor = mydb.cursor()
        # MySQL db에서 place, created_at, dong, ho 값 가져오기
        mycursor.execute("SELECT place, created_at, dong, ho, value FROM main_sound_file WHERE file_name = %s", (filename,))
        result = mycursor.fetchone()
        if result:
            place, created_at, dong, ho = result
        else:
            print("No record found for the given file_name")
            exit()

        # main_sound_level_verified 테이블에 값을 저장

        sql = "INSERT INTO main_sound_level_verified (file_name, sound_type, place, created_at, dong, ho, value) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = (filename, result_class, place, created_at, dong, ho)

        mycursor.execute(sql, val)

        mydb.commit()

        print(mycursor.rowcount, "record inserted.")

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()