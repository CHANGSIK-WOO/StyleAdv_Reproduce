#!/usr/bin/env python3

import os
from pathlib import Path
from test_function_bscdfsl_benchmark_ViT import test_bestmodel_bscdfsl_ViT


def main():
    # 학습된 모델 경로 (학습 시 --output으로 지정한 경로)
    model_name = "ViT_5WAY_1SHOTS"
    n_shot = 1  # 학습 시 --nSupport와 동일하게 설정

    # 테스트할 데이터셋 목록
    test_datasets = ['ISIC', 'EuroSAT', 'CropDisease', 'ChestX']

    # 결과 저장할 파일 경로
    output_dir = Path(f"output/{model_name}")
    acc_file_path = output_dir / 'test_results.txt'

    print("=" * 50)
    print("Starting testing on BS-CDFSL datasets...")
    print(f"Model: {model_name}")
    print(f"n_shot: {n_shot}")
    print("=" * 50)

    # 결과 파일 생성
    with open(acc_file_path, 'w') as acc_file:
        print('Testing Results:', file=acc_file)
        print('Model:', model_name, file=acc_file)
        print('n_shot:', n_shot, file=acc_file)
        print('-' * 50, file=acc_file)

        # 각 데이터셋에 대해 테스트 수행
        for dataset in test_datasets:
            print(f"\nTesting on {dataset}...")
            print(f'Testing on {dataset}:', file=acc_file)
            try:
                test_bestmodel_bscdfsl_ViT(acc_file, model_name, dataset, n_shot, save_epoch=-1)
                print(f"Completed testing on {dataset}")
            except Exception as e:
                print(f"Error testing on {dataset}: {e}")
                print(f"Error testing on {dataset}: {e}", file=acc_file)

    print(f"\nAll testing completed. Results saved to {acc_file_path}")


if __name__ == '__main__':
    main()