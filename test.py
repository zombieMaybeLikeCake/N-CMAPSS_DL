import numpy as np
import os
# def load_part_array_merge(npz_units):
#     sample_array_lst = []
#     label_array_lst = []
#     for npz_unit in npz_units:
#       # loaded = np.load(npz_unit)
#       loaded = np.load(os.path.join(r'.\N-CMAPSS\Samples_whole',npz_unit))
#       sample_array_lst.append(loaded['sample'])
#       label_array_lst.append(loaded['label'])
#     sample_array = np.dstack(sample_array_lst)
#     label_array = np.concatenate(label_array_lst)
#     sample_array = sample_array.transpose(2, 0, 1)
#     return sample_array, label_array
# filelist=os.listdir(r'.\N-CMAPSS\Samples_whole')
# npz_units=[]
# for file in filelist:
#     datas = np.load(os.path.join(r'.\N-CMAPSS\Samples_whole',file))
#     npz_units.append(datas)
# sample_array,label_array=load_part_array_merge(filelist)
# print("sample_array")
# print(sample_array.shape)
# print("label_array")
# print(label_array)

# import os
# import numpy as np

# def ds022txt(npz_unit):
#     # 讀取 npz 檔案
#     loaded = np.load(os.path.join(r'.\N-CMAPSS\Samples_whole', npz_unit))
    
#     # 打印文件內部的數據鍵（例如：'sample', 'label'等）
#     print(loaded.files)
    
#     # 取得 'sample' 和 'label' 的資料
#     sample_data = loaded['sample']
#     label_data = loaded['label']
    
#     # 取得 'sample' 的形狀
#     sample_shape = sample_data.shape
#     label_shape = label_data.shape

#     # 生成文字檔案名
#     txt_filename = os.path.splitext(npz_unit)[0] + '_sample_label.txt'
    
#     # 將 'label' 和 'sample' 寫入文字檔案
#     with open(txt_filename, 'w') as f:
#         # 先寫入 'label' 和 'sample'
#         i=0
#         for rowrul, row in zip(label_data, sample_data.reshape(-1, sample_data.shape[-1])):  # 重塑為一行行輸出
#             # 將 label 和 sample 合併並寫入
#             f.write(' '.join(map(str, [rowrul] + list(row))) + '\n')
#             i+=1
#             if i>2:
#               break

#     print(f"Saved {txt_filename} with sample shape {sample_shape} and label shape {label_shape}")
    

# # 假設 filelist 是你已經列出的文件列表
# filelist = os.listdir(r'.\N-CMAPSS\Samples_whole')
# for npz_file in filelist:
#     if npz_file.endswith('.npz'):
#         ds022txt(npz_file)
#         break
for i in range(6, 10):
  print(2**i)


