import numpy as np

# 파일 불러오기
data = np.load('data.npz')
# data = np.load('converted_all_data.npz')
# 키 목록 출력
print("📦 파일에 저장된 항목들:")
print(data.files)

# 각 항목 내용 확인
for key in data.files:
    print(f"\n🔑 [{key}]")
    print(f"shape: {data[key].shape}")
    print(f"dtype: {data[key].dtype}")
    
    if key in ['lb', 'ub','X_test','Y_test']:
        # lb, ub는 전체 다 출력
        print(f"values: {data[key]}")
    else:
        # 나머지는 앞 3개만 미리보기
        print(f"sample data (first 3 rows):\n{data[key][:3]}")
    if key == 'X_test':
        X_test = data['X_test']
        subset = X_test[0:40,:]
        print(subset)
        subset = X_test[40:80,:]
        print(subset)
        subset = X_test[80:120,:]
        print(subset)
        subset = X_test[120:160,:]
        print(subset)
        subset = X_test[160:200,:]
        print(subset)

        subset = X_test[720:760,:]
        print(subset)
        subset = X_test[760:800,:]
        print(subset)
        # u1_min = X_test[:,5].min()
        # u2_min = X_test[:,6].min()
        # u1_max = X_test[:,5].max()
        # u2_max = X_test[:,6].max()
        # print(u1_min)
        # print(u2_min)
        # print(u1_max)
        # print(u2_max)

    elif key == 'Y':
        X_test = data['Y']
        subset = X_test[0:40,:]
        # print(subset)
        # subset = X_test[40:80,:]
        # print(subset)
        # subset = X_test[80:120,:]
        # print(subset)
        # subset = X_test[120:160,:]
        # print(subset)
        # subset = X_test[160:200,:]
        # print(subset)
        # subset = X_test[200:240,:]
        # print(subset)
        # subset = X_test[240:280,:]
        # print(subset)
        # subset = X_test[280:320,:]
        # print(subset)
        # subset = X_test[320:360,:]
        # print(subset)
        # subset = X_test[360:400,:]
        # print(subset)
        # subset = X_test[400:440,:]
        # print(subset)
        # subset = X_test[440:480,:]
        # print(subset)
        # subset = X_test[480:520,:]
        # print(subset)
        # subset = X_test[520:560,:]
        # print(subset)
        # subset = X_test[560:600,:]
        # print(subset)
        # subset = X_test[600:640,:]
        # print(subset)
        # subset = X_test[640:680,:]
        # print(subset)
        # subset = X_test[680:720,:]
        # print(subset)
        # subset = X_test[720:760,:]
        # print(subset)
        # subset = X_test[760:800,:]
        # print(subset)

        # subset = X_test[720:760,:]
        # print(subset)
        # subset = X_test[760:800,:]
        # print(subset)

