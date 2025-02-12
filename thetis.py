import torch
import math
import matplotlib.pyplot as plt
import PIL

snr_list = [0, 10, 20, 30]

# /mnt/data/datasets/CREMA-D/processed_dataset/train/visual/NEU/1001_TIE_NEU_XX.jpg; /mnt/data/datasets/CREMA-D/processed_dataset/train/visual/HAP/1001_IEO_HAP_HI.jpg; /mnt/data/datasets/CREMA-D/processed_dataset/train/visual/DIS/1001_IEO_DIS_HI.jpg
for snr in snr_list:
    # Example values, modify according to your actual settings
    args = {
        'dataset': 'CREMAD',
        'modality': 'audio-visual',
        'snr': snr,  # Signal-to-noise ratio
        'step': 5  # Example step, adjust as necessary
    }

    # Loading the image
    file_path = "/mnt/data/datasets/CREMA-D/processed_dataset/train/visual/DIS/1001_IEO_DIS_HI.jpg"
    visual_context = PIL.Image.open(file_path).convert("RGB")

    # 1) 把图像转换为张量 (H, W, 3) -> (3, H, W) -> (1, 3, H, W)
    W, H = visual_context.size
    pixels_list = list(visual_context.getdata())  # [(R, G, B), ...] length = H*W
    pixels_tensor = torch.tensor(pixels_list, dtype=torch.float32).view(H, W, 3)  # [H, W, 3]
    pixels_tensor = pixels_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # 2) Repeat for step size => [step, 3, H, W]
    pixels_tensor = pixels_tensor.repeat(args['step'], 1, 1, 1)  # [step, 3, H, W]

    if args['snr'] >= -10:
        # 3) 选取其中一张加噪
        image = pixels_tensor[1]  # shape [3, H, W]
        power_signal = torch.mean(image ** 2)
        # 根据 SNR 计算噪声尺度：SNR(dB) = 10 * log10( signal_power / noise_power )
        noise_scale = math.sqrt(power_signal / (10 ** (args['snr'] / 10)))
        noise = torch.randn_like(image) * noise_scale

        pixels_tensor[1] = image + noise

    # 4) 把加过噪声的图像转回 PIL.Image 进行可视化
    #    pixels_tensor[1] => shape [3, H, W]
    noisy_image_tensor = pixels_tensor[1].cpu().clamp(0, 255)  # 先 clamp 到 [0, 255]
    noisy_image_numpy = noisy_image_tensor.numpy().transpose(1, 2, 0).astype('uint8')  # [H, W, 3]
    noisy_image = PIL.Image.fromarray(noisy_image_numpy)

    # 5) 可视化
    plt.figure()
    plt.imshow(noisy_image)
    plt.axis('off')

    # 保存加噪图为SVG文件（可选）
    visual_svg_path = 'visual_plot_snr_{}.svg'.format(snr)
    plt.savefig(visual_svg_path, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.show()


    import math
    import torch
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    # 1. 设置音频路径、以及所需的SNR
    audio_path = "/mnt/data/datasets/CREMA-D/processed_dataset/train/audio/DIS/1001_IEO_DIS_HI.wav"

    # 2. 使用librosa加载音频数据
    y, sr = librosa.load(audio_path, sr=None)

    # 3. 把numpy音频信号转换为PyTorch张量，以便计算噪声
    y_torch = torch.from_numpy(y)

    # 4. 计算原始音频的信号功率
    signal_power = torch.mean(y_torch ** 2)  # mean(y^2)

    # 5. 根据SNR(dB)计算噪声的标准差，并生成噪声
    #    公式： SNR(dB) = 10 * log10(signal_power / noise_power)
    #    => noise_power = signal_power / (10^(SNR/10))
    #    => noise_std = sqrt(noise_power)
    noise_std = math.sqrt(signal_power / (10 ** (snr / 10)))
    noise = torch.randn_like(y_torch) * noise_std

    # 6. 将噪声加入原音频
    y_noisy_torch = y_torch + noise

    # 7. 转回 numpy 数组，供后续librosa操作使用
    y_noisy = y_noisy_torch.numpy()

    # 8. 使用matplotlib可视化音频波形 / 梅尔频谱
    # 例如，这里直接可视化“加过噪声后的” 梅尔频谱

    # 提取梅尔频谱
    melspec = librosa.feature.melspectrogram(y=y_noisy, sr=sr)
    logmelspec = librosa.power_to_db(melspec)

    # 绘制梅尔频谱
    fig, ax = plt.subplots(1, 1)
    img = librosa.display.specshow(logmelspec, y_axis='mel', x_axis='time', sr=sr, ax=ax)
    # ax.set_title('Noisy Audio Mel-Spectrogram')
    # fig.colorbar(img, ax=ax, format="%+2.f dB")

    ax.axis('off')

    # 保存梅尔频谱图为SVG文件（可选）
    melspec_svg_path = 'melspectrogram_plot_snr_{}.svg'.format(snr)
    plt.savefig(melspec_svg_path, format='svg', bbox_inches='tight', pad_inches=0)

    # plt.show()
