# 이미지를 빌드할 베이스 이미지 선택
FROM public.ecr.aws/lambda/python:3.8

# 필요한 라이브러리 및 패키지 설치
COPY requirements.txt .

RUN python -m pip install -r requirements.txt

# 딥 러닝 모델 파일 복사 (옵션, 필요에 따라 주석 처리)
COPY all_batch32_dense(224,224).hdf5 /var/task/

# Lambda 함수 코드 복사
COPY lambda_function.py /var/task/

# Lambda 함수 핸들러 설정
CMD ["lambda_function.lambda_handler"]
