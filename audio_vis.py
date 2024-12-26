import matplotlib.pyplot as plt
import librosa
import librosa.display

audio_path = "/mnt/data/datasets/UrbanSound8K-AV/Dataset_v3_sound/street_music/126153-9-0-6.wav"

# 2. 使用librosa加载音频数据
y, sr = librosa.load(audio_path, sr=None)

# 3. 使用matplotlib可视化音频波形
plt.figure(figsize=(10, 4))

# --- plot 单个的 ------
librosa.display.waveshow(y, sr=sr, color='green')

# for i in range(1, 10):
#     plt.axvline(i, color='lightgray', linestyle='--')

plt.title('Waveform')
plt.tight_layout()


waveform_svg_path = 'waveform_plot.svg'
plt.savefig(waveform_svg_path, format='svg')

plt.clf()  # 清除当前图形


# ax = plt.gca()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

## -------plot 梅尔频谱---------

# 提取梅尔频谱
melspec = librosa.feature.melspectrogram(y, sr)

# 将梅尔频率数据做对数变换
logmelspec = librosa.power_to_db(melspec)

# 绘制图像
fig, ax = plt.subplots(1, 1)
# x轴是时间（单位：秒），y轴是梅尔尺度的频率值（单位：Hz）
img = librosa.display.specshow(logmelspec, y_axis='mel', sr=sr, x_axis='time')
plt.title('audio.wav', fontproperties="SimSun")
fig.colorbar(img, ax=ax, format="%+2.f dB")

# 保存梅尔频谱图为SVG文件
melspec_svg_path = 'melspectrogram_plot.svg'
plt.savefig(melspec_svg_path, format='svg')

plt.show()




# # --- plot 多个的 ---
# # 计算每个分段的样本数量
# segment_samples = len(y) // 10
#
# # 3. 使用matplotlib分别可视化每个部分的音频波形
# for i in range(10):
#     plt.figure(figsize=(10, 4))
#
#     segment_start = i * segment_samples
#     segment_end = (i + 1) * segment_samples
#     librosa.display.waveshow(y[segment_start:segment_end], sr=sr)
#
#     plt.title(f'Waveform Segment {i + 1}')
#     plt.tight_layout()
#
#     # 去掉边框
#     ax = plt.gca()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#
#     plt.show()