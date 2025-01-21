# Import các thư viện cần thiết
from pathlib import Path
import argparse
import sys
import os
import pandas as pd
from xgboost import XGBClassifier
from joblib import dump, load
import numpy as np
from sklearn.metrics import f1_score
from utils import *
import json
import numpy as np
from hpo import *

# Hàm để chạy quá trình huấn luyện
def run_train(train_dir, dev_dir, model_dir):
    set_seed(42)
    # Tạo thư mục cho mô hình nếu nó chưa tồn tại
    os.makedirs(model_dir, exist_ok=True)

    # Đường dẫn đến các tệp dữ liệu
    train_file = os.path.join(train_dir, 'train.json')
    dev_file = os.path.join(dev_dir, 'dev.json')

    # Đọc dữ liệu huấn luyện và phát triển
    train_data = pd.read_json(train_file, lines=True)
    dev_data = pd.read_json(dev_file, lines=True)

    # # Biến đổi dữ liệu và thực hiện feature selection
    X_train, y_train = preprocess(train_data, split='train')
    X_dev, y_dev = preprocess(dev_data, split='validation')

    # Tạo và huấn luyện mô hình

    X_train.to_csv('preprocessed_X_train.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    X_dev.to_csv('preprocessed_X_dev.csv', index=False)
    y_dev.to_csv('y_dev.csv', index=False)

    create_study('Voting    ')

    # model_1 = train(X_train, y_train, X_dev, y_dev, 'xgb5')
    # predictions, probas_xgb1 = predict(model_1, X_dev, y_dev, 'XGBoost1')
    # #
    # # model_2, features = train(X_train, Y_train, X_dev, Y_dev, 'xgb2')
    # # predictions, probas_xgb2 = predict(model_2, X_dev[features], Y_dev, 'XGBoost2')
    # #
    # # model_3, features = train(X_train, Y_train, X_dev, Y_dev, 'xgb3')
    # # predictions, probas_xgb3 = predict(model_3, X_dev[features], Y_dev, 'XGBoost3')
    # #
    # # model_4, features = train(X_train, Y_train, X_dev, Y_dev, 'xgb4')
    # # predictions, probas_xgb4 = predict(model_4, X_dev[features], Y_dev, 'XGBoost4')
    # #
    # predictions = model_1.predict(X_dev)
    #
    # print(f1_score(y_dev, predictions))
    #
    # # Lưu mô hình
    # model_path = os.path.join(model_dir, 'trained_model.joblib')
    #
    # dump(model_1, model_path)
    # print(f'Saving models into {model_path}')
    # #
    # pd.DataFrame(predictions, columns=['two_year_recid']).to_json('2_compas_dev.json', orient='records', lines=True)


# Hàm để chạy quá trình dự đoán
def run_predict(model_dir, input_dir, output_path):
    # Đường dẫn đến mô hình và dữ liệu đầu vào
    model_path = Path(model_dir) / 'trained_model.joblib'
    input_file = Path(input_dir) / 'test.json'

    # Tải mô hình
    model = load(model_path)

    # Đọc dữ liệu kiểm tra
    test_data = pd.read_json(input_file, lines=True)
    # Chuẩn bị dữ liệu kiểm tra
    X_test, _ = preprocess(test_data, split='test')

    # Thực hiện dự đoán
    predictions = model.predict(X_test)

    # Lưu kết quả dự đoán
    pd.DataFrame(predictions, columns=['two_year_recid']).to_json(output_path, orient='records', lines=True)

# Hàm chính để xử lý lệnh từ dòng lệnh
def main():
    # Tạo một parser cho các lệnh từ dòng lệnh
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Tạo parser cho lệnh 'train'
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--train_dir', type=str)
    parser_train.add_argument('--dev_dir', type=str)
    parser_train.add_argument('--model_dir', type=str)

    # Tạo parser cho lệnh 'predict'
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('--model_dir', type=str)
    parser_predict.add_argument('--input_dir', type=str)
    parser_predict.add_argument('--output_path', type=str)

    # Xử lý các đối số nhập vào
    args = parser.parse_args()

    # Chọn hành động dựa trên lệnh
    if args.command == 'train':
        run_train(args.train_dir, args.dev_dir, args.model_dir)
    elif args.command == 'predict':
        run_predict(args.model_dir, args.input_dir, args.output_path)
    else:
        parser.print_help()
        sys.exit(1)

# Điểm khởi đầu của chương trình
if __name__ == "__main__":
    main()
