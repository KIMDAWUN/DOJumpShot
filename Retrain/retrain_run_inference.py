# -*- coding: utf-8 -*-

"""Inception v3 architecture 모델을 retraining한 모델을 이용해서 이미지에 대한 추론(inference)을 진행하는 예제"""

import numpy as np
import tensorflow as tf

imagePath = 'C:\\tmp\\jj.jpg'                                      # 추론을 진행할 이미지 경로
modelFullPath = '/tmp/opt_graph.pb'                                      # 읽어들일 graph 파일 경로
labelsFullPath = '/tmp/labels.txt'                                   # 읽어들일 labels 파일 경로


#사전 학습된 파라미터와 그래프 구조를 저장하고 있는 
#output_graph.pb파일을 읽어서 그래프를 생성하는 create_graph()함수
def create_graph():

    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = None

#만약 경로에 이미지 파일이 없을 경우 오류 로그를 출력
    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

#이미지 파일을 읽는다
    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    create_graph()


# 'softmax:0': 1000개의 레이블에 대한 정규화된 예측결과값(normalized prediction)을 포함하고 있는 텐서   
# 'pool_3:0': 2048개의 이미지에 대한 float 묘사를 포함하고 있는 next-to-last layer를 포함하고 있는 텐서
# 'DecodeJpeg/contents:0': 제공된 이미지의 JPEG 인코딩 문자를 포함하고 있는 텐서
#세션을 열고 그래프를 실행
    with tf.Session() as sess:

        #최종 소프트 맥스 행렬의 출력 층을 지정
        #softmax_tensor == 이미지에 대한 추론값인 소프트맥스 행렬을 출력하는 최종 출력층
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        #추론할 이미지를 DecodeJpeg/contents:0이라는 이름의 플레이스홀더에 넣고
        #softmax_tensor를 실행해서 네트워크의 출력값 predictions을 구함
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        #np.squeeze API를 사용해서 predictions의 불필요한 차원을 제거
        predictions = np.squeeze(predictions)

        # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
        #[0 3 2 4 1]
        top_k = predictions.argsort()[-5:][::-1]  
        
        #output_label.txt.파일로부터 정답 레이블들을 list형태로 가져옴
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        
        #가장 높은 확률을 가진 인덱스들부터 추론 결과(top 5)를 출력
        for node_id in top_k:
            label_name = labels[node_id]
            probability = predictions[node_id]
            print('%s (확률 = %.5f)' % (label_name, probability))

        #가장 높은 확률을 가진 Top-1 추론 결과를 출력한다
        answer = labels[top_k[0]]
        probability = predictions[top_k[0]]
        print('%s(확률=%.5f)'&(answer, probability))



if __name__ == '__main__':
    run_inference_on_image()
